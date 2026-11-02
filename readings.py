"""What a drawing reads as: the reader's best reading and the likeliest others, so that the writer can say which is right."""
import re

import numpy as np

from layout import MAIN_LINE, TOKENS, WORDS, to_text
from reader import segment
from solver import read_as_written

MOST_READINGS = 4     # the most offered, the best reading included
LEAST_SHARE = 0.02    # a second guess with less of the CNN's belief than this is not worth offering
RARELY_ALONE = 0.2    # a letter that seldom stands alone (it is kept for function words) counts for less as a second guess
LABEL_OF = {"+": "plus", "-": "minus", "×": "x", "÷": "slash", "(": "lparen", ")": "rparen", "→": "arrow", "∞": "inf",
            "fraction bar": "minus", "∫": "integral", "√": "sqrt", "π": "pi", "[": "lbracket", "]": "rbracket", "∂": "partial",
            "!": "fact", "λ": "lambda", "μ": "mu", "θ": "theta"}  # what the reader wrote -> CNN class
IN_LATEX = {"×": r"\times ", "÷": r"\div ", "·": r"\cdot ", "−": "-", "π": r"\pi ", "λ": r"\lambda ", "μ": r"\mu ", "θ": r"\theta ", "∂": r"\partial ", "∞": r"\infty ",
            "→": r"\to ", "∫": r"\int ", " ": r"\ "}
COUNTING = re.compile(r"(\d+)([cp])(\d+)")


def score(symbols, model):
    """Let the CNN score every symbol the shape rules did not decide."""
    todo = [s for s in symbols if s["kind"] is None]
    if todo:
        probs = model(np.stack([s["image"] for s in todo])[..., None], training=False).numpy()
        for s, p in zip(todo, probs):
            s["probs"] = p


def read_board(mask, strokes, model, labels):
    """(the symbols, the text they read as)."""
    symbols = segment(mask, strokes)
    score(symbols, model)
    return symbols, to_text(symbols, labels)


def can_read(text):
    """Would the solver understand this line? Commands such as rank[[1,2],[3,4]] count as understood."""
    if "[[" in text:
        return True
    try:
        for line in text.split(";"):
            read_as_written(line)
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _after_group(text, start):
    """Where the bracket group opening at start ends."""
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "(") - (text[i] == ")")
        if depth == 0:
            return i + 1
    raise ValueError("A bracket is not closed")


def _as_latex(text):
    """A line as the reader wrote it, in LaTeX: nothing moved and nothing worked out."""
    out, i = [], 0
    while i < len(text):
        word = next((w for w in WORDS if text.startswith(w, i)), None)
        counting = COUNTING.match(text, i)
        if text[i] == "(":
            end = _after_group(text, i)
            if text.startswith("/(", end):
                stop = _after_group(text, end + 1)
                out.append(rf"\frac{{{_as_latex(text[i + 1:end - 1])}}}{{{_as_latex(text[end + 2:stop - 1])}}}")
                end = stop
            else:
                out.append(rf"\left({_as_latex(text[i + 1:end - 1])}\right)")
            i = end
        elif text[i] in "^_√" and text.startswith("(", i + 1):
            end = _after_group(text, i + 1)
            inside = _as_latex(text[i + 2:end - 1])
            out.append(rf"\sqrt{{{inside}}}" if text[i] == "√" else f"{text[i]}{{{inside}}}")
            i = end
        elif text[i] in "^_" and text[i + 1:i + 2].isdigit():
            digits = re.match(r"\d+", text[i + 1:])[0]
            out.append(f"{text[i]}{{{digits}}}")
            i += 1 + len(digits)
        elif word:
            out.append(rf"\{word} " if word in ("lim", "log", "sin", "cos", "tan", "ln", "cot", "sec", "exp", "det") else rf"\operatorname{{{word}}} ")
            i += len(word)
        elif counting and (i == 0 or not text[i - 1].isdigit()):
            out.append(rf"\binom{{{counting[1]}}}{{{counting[3]}}}" if counting[2] == "c" else rf"{{}}^{{{counting[1]}}}P_{{{counting[3]}}}")
            i = counting.end()
        else:
            out.append(IN_LATEX.get(text[i], text[i]))
            i += 1
    return "".join(out)


def to_latex(text):
    """A reading as a formula to show, or None if it is a command or cannot be read."""
    if "[[" in text or not can_read(text):
        return None
    try:
        return r" \quad ".join(_as_latex(line.strip()) for line in text.split(";"))
    except ValueError:  # the solver read it, but its brackets are not how a formula is shown
        return None


def _with_symbol_changed(symbols, k, label, labels):
    """The whole reading if symbol k were this label."""
    trial = [{key: value for key, value in s.items() if key != "read_as"} for s in symbols]
    probs = np.zeros(len(labels), np.float32)
    probs[labels.index(label)] = 1
    trial[k].update(probs=probs, forced=TOKENS.get(label, label))
    return to_text(trial, labels)


def _second_guesses(symbols, labels):
    """(how close, which symbol, what else it could be), closest first. A letter of a function word stays: 1og is never meant."""
    guesses = []
    for k, s in enumerate(symbols):
        if s["kind"] is not None or s.get("in_word"):
            continue
        read = LABEL_OF.get(s.get("read_as"), s.get("read_as"))
        believed = s["probs"][labels.index(read)] if read in labels else s["probs"].max()
        for i in np.argsort(s["probs"])[::-1][:3]:
            if labels[i] != read and s["probs"][i] >= LEAST_SHARE:
                weight = RARELY_ALONE if labels[i] in MAIN_LINE else 1
                guesses.append((weight * s["probs"][i] / max(believed, 1e-6), k, labels[i]))
    return sorted(guesses, key=lambda guess: -guess[0])


def other_readings(symbols, best, labels, mask, strokes, model):
    """The best reading first, then the drawing grouped from its pixels alone, then the closest second guesses of single symbols."""
    readings = [best]
    if strokes is not None:  # the record of pen strokes can split what the pixels show as one symbol, and the other way round
        from_pixels = segment(mask)
        score(from_pixels, model)
        text = to_text(from_pixels, labels)
        if text and text not in readings and can_read(text):
            readings.append(text)
    for _, k, label in _second_guesses(symbols, labels):
        if len(readings) >= MOST_READINGS:
            break
        text = _with_symbol_changed(symbols, k, label, labels)
        if text and text not in readings and can_read(text):
            readings.append(text)
    return readings
