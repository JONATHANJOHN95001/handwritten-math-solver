"""Measure the whole reader (segment + CNN + layout) on maths built from held-out test symbols.

Each kind of expression is glued together from test images the CNN never trained on, then read back.
A read counts only when the whole line comes out exactly right. Writes reports/reading_check.json and
refreshes the whole-sum score in reports/metrics.json.

Usage: python check_reading.py --bhmsds <folder>/bhmsds/symbols --emnist <folder with emnist-letters-*.gz>
"""
import argparse
import json
import sys

import cv2
import numpy as np
import tensorflow as tf

from layout import MAIN_LINE, to_text
from reader import segment
from train import CLASSES, HERE, SEED, expression_check, load_split

TILE, SMALL = 112, 56  # a 28x28 test image drawn at about the size people write; powers and scripts at half that
TOP = 150              # top of the main line's tiles (the ink starts 1/7 of a tile lower)
PER_KIND = 200


class Writer:
    """Pastes random test images on an empty board, like a person writing with many different hands."""

    def __init__(self, by_class, rng):
        self.by_class, self.rng = by_class, rng
        self.mask = np.zeros((520, 2400), np.uint8)

    def put(self, label, x, y, size=TILE):
        """Paste a random test image with its ink starting at x; returns where the ink ends."""
        tiles = self.by_class[CLASSES.index(label)]
        tile = cv2.resize((tiles[self.rng.integers(len(tiles))] * 255).astype(np.uint8), (size, size))
        columns = np.flatnonzero((tile > 64).any(axis=0))
        tile = tile[:, columns[0]:columns[-1] + 1]
        width = tile.shape[1]
        self.mask[y:y + size, x:x + width] = np.maximum(self.mask[y:y + size, x:x + width], tile)
        return x + width

    def row(self, labels, x, y=TOP, size=TILE):
        """Write symbols side by side with a gap of a fifth of their size, as in handwriting."""
        for label in labels:
            x = self.put(label, x, y, size) + size // 5
        return x - size // 5

    def bar(self, x0, x1, y):
        cv2.line(self.mask, (x0, y), (x1, y), 255, 6)

    def number(self, lo=1, hi=2):
        first = str(self.rng.integers(1, 10))
        return first + "".join(str(self.rng.integers(0, 10)) for _ in range(self.rng.integers(lo, hi + 1) - 1))


def equation(w):
    a, b, c, sign = w.number(1, 1), w.number(), w.number(), str(w.rng.choice(["+", "-"]))
    x = w.row([a, "x", "plus" if sign == "+" else "minus", *b], 40)
    w.bar(x + 30, x + 100, TOP + 40)
    w.bar(x + 30, x + 100, TOP + 72)
    w.row(list(c), x + 130)
    return f"{a}x{sign}{b}={c}"


def power(w):
    base, exponent = str(w.rng.choice(["x", w.number(1, 1)])), w.number(1, 1)
    x = w.row([base], 40)
    w.put(exponent, x + 12, TOP - 24, SMALL)
    return f"{base}^{exponent}"


def function(w):
    word, arg = str(w.rng.choice(["log", "ln", "sin", "cos", "tan"])), str(w.rng.choice(["x", w.number(1, 3)]))
    w.row([*word, "lparen", *arg, "rparen"], 40)
    return f"{word}({arg})"


def log_with_base(w):
    base, n = w.number(1, 1), w.number()
    x = w.row(list("log"), 40)
    x = w.put(base, x + 10, TOP + 72, SMALL)
    w.row(["lparen", *n, "rparen"], x + 24)
    return f"log_{base}({n})"


def fraction(w):
    top, bottom, size = w.number(), w.number(), 84
    end = max(w.row(list(top), 70, TOP + 56 - size - 2, size), w.row(list(bottom), 70, TOP + 56 + 6, size))
    w.bar(40, end + 30, TOP + 56)
    return f"({top})/({bottom})"


def limit(w):
    target, d = str(w.rng.choice(["inf", w.number(1, 1)])), w.number(1, 1)
    x = w.row(list("lim"), 40)
    end = w.row(["x", "arrow", target], 45, TOP + 96 + 10, SMALL)
    w.row(["x", "plus", d], max(x, end) + 50)
    return f"lim_(x→{'∞' if target == 'inf' else target}) x+{d}"


def tall(w, label, x, y, height):
    """A bracket or ∫ sign stretched tall, as people write them round a matrix or a whole integrand."""
    tiles = w.by_class[CLASSES.index(label)]
    tile = (tiles[w.rng.integers(len(tiles))] * 255).astype(np.uint8)
    columns = np.flatnonzero((tile > 64).any(axis=0))
    tile = tile[:, columns[0]:columns[-1] + 1]
    scale = height / 28
    resized = cv2.resize(tile, (max(4, round(tile.shape[1] * scale)), height))
    w.mask[y:y + height, x:x + resized.shape[1]] = np.maximum(w.mask[y:y + height, x:x + resized.shape[1]], resized)
    return x + resized.shape[1]


def letter_equation(w):
    a, b, c, letter = w.number(1, 1), w.number(1, 1), w.number(), str(w.rng.choice(["y", "z", "t"]))
    x = w.row([a, letter, "plus", b], 40)
    w.bar(x + 30, x + 100, TOP + 40)
    w.bar(x + 30, x + 100, TOP + 72)
    w.row(list(c), x + 130)
    return f"{a}{letter}+{b}={c}"


def system(w):
    a, c = w.number(1, 1), w.number(1, 1)
    size = 84
    for line, answer in enumerate((a, c)):
        y = 60 + line * (size + 70)
        x = w.row(["x", "plus", "y"], 40, y, size)
        w.bar(x + 24, x + 84, y + 30)
        w.bar(x + 24, x + 84, y + 54)
        w.row([answer], x + 110, y, size)
    return f"x+y={a}; x+y={c}"


def matrix(w):
    numbers = [str(w.rng.integers(1, 10)) for _ in range(4)]
    word = w.row(list("rank"), 40, 120, 70)
    left = word + 40
    tall(w, "lbracket", left, 40, 240)
    w.row([numbers[0]], left + 70, 60, 84)
    w.row([numbers[1]], left + 190, 60, 84)
    w.row([numbers[2]], left + 70, 170, 84)
    w.row([numbers[3]], left + 190, 170, 84)
    tall(w, "rbracket", left + 300, 40, 240)
    return f"rank[[{numbers[0]},{numbers[1]}],[{numbers[2]},{numbers[3]}]]"


def integral(w):
    upper, lower, power_of = w.number(1, 1), w.number(1, 1), w.number(1, 1)
    end = tall(w, "integral", 40, 60, 260)
    w.put(upper, end + 6, 50, SMALL)
    w.put(lower, end + 6, 270, SMALL)
    x = w.row(["x"], end + 90, 150)
    x = w.put(power_of, x + 12, 126, SMALL)
    w.row(["d", "x"], x + 30, 150)
    return f"∫_({lower})^({upper}) x^{power_of}dx"


def derivative(w):
    number, size = w.number(1, 1), 84
    end = max(w.row(["d", "y"], 70, TOP + 56 - size - 2, size), w.row(["d", "x"], 70, TOP + 56 + 6, size))
    w.bar(40, end + 30, TOP + 56)
    x = w.row(["plus", "y"], end + 70, TOP - 16)
    w.bar(x + 30, x + 100, TOP + 24)
    w.bar(x + 30, x + 100, TOP + 56)
    w.row([number], x + 130, TOP - 16)
    return f"(dy)/(dx)+y={number}"


def factorial(w):
    n = w.number(1, 1)
    w.row([n, "fact"], 40)
    return f"{n}!"


def combination(w):
    n, r = w.number(2, 2), w.number(1, 1)
    w.row([*n, "c", r], 40)
    return f"{n}c{r}"


KINDS = {"Equations": equation, "Powers": power, "Functions (log, ln, sin, cos, tan)": function,
         "Logs with a base": log_with_base, "Fractions": fraction, "Limits": limit,
         "Equations in other letters": letter_equation, "Two lines of equations": system, "Matrices with a word": matrix,
         "Integrals with limits": integral, "dy/dx": derivative, "Factorials": factorial, "Combinations (nCr)": combination}


def read(mask, model):
    symbols = segment(np.where(mask > 64, 255, 0).astype(np.uint8))
    todo = [s for s in symbols if s["kind"] is None]
    if todo:
        probs = model(np.stack([s["image"] for s in todo])[..., None], training=False).numpy()
        for s, p in zip(todo, probs):
            s["probs"] = p
    return to_text(symbols, CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bhmsds", required=True)
    ap.add_argument("--emnist", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    _, _, _, (x_test, y_test) = load_split(args.bhmsds, args.emnist)
    model = tf.keras.models.load_model(HERE / "models" / "symbol_cnn.keras")
    by_class = [x_test[y_test == c] for c in range(len(CLASSES))]

    probs = model.predict(x_test[..., None], verbose=0)
    allowed = [i for i, c in enumerate(CLASSES) if c not in MAIN_LINE]
    number_line = np.isin(y_test, [CLASSES.index(c) for c in [str(d) for d in range(10)] + ["plus", "minus", "x", "slash"]])
    masked = np.array(allowed)[probs[number_line][:, allowed].argmax(axis=1)]
    results = {
        "test_accuracy": round(float((probs.argmax(axis=1) == y_test).mean()), 4),
        "digits_and_operators_with_letters_ruled_out": round(float((masked == y_test[number_line]).mean()), 4),
        "kinds": {"Plain sums": {"expressions": 300, "read_exactly": round(expression_check(model, x_test, y_test) * 300)}},
    }
    rng = np.random.default_rng(SEED)
    for name, build in KINDS.items():
        right, misses = 0, []
        for _ in range(PER_KIND):
            writer = Writer(by_class, rng)
            want = build(writer)
            got = read(writer.mask, model)
            right += got == want
            if got != want and len(misses) < 6:
                misses.append(f"{want} -> {got}")
        results["kinds"][name] = {"expressions": PER_KIND, "read_exactly": right, "examples_of_misses": misses}
        print(f"{name}: {right}/{PER_KIND}", misses, flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    reports = HERE / "reports"
    (reports / "reading_check.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = json.loads((reports / "metrics.json").read_text())
    metrics["expression_accuracy"] = round(results["kinds"]["Plain sums"]["read_exactly"] / 300, 4)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
