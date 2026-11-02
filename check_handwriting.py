"""Score models on real handwriting that was kept out of training, so that the figures can be compared with each other.

Symbols: the test images of each source. Expressions: the MathWriting test split, each real expression redrawn stroke by
stroke as the board does and read by the whole reader. Only expressions the board has every symbol for can count.

Usage: python check_handwriting.py --data <folder made by prepare_handwriting.py> --model new=models/symbol_cnn.keras [--model old=<file>]
"""
import argparse
import copy
import gzip
import json
import pathlib
import re
import sys

import cv2
import numpy as np
import tensorflow as tf

from layout import MAIN_LINE, WORDS, to_text
from reader import segment
from train import CLASSES, HERE, REAL, SHOWN

SAME_SHAPE = [{"1", "l", "i"}, {"0", "o"}]  # only where a symbol stands can tell these apart
PLAIN = set("0123456789abcdefghijklmnopqrstuvwxyz+-=()[]/.,!")
COMMANDS = {"\\times": "x", "\\cdot": ".", "\\div": "/", "\\pi": "π", "\\theta": "θ", "\\lambda": "λ", "\\mu": "μ", "\\infty": "∞",
            "\\partial": "∂", "\\to": "→", "\\rightarrow": "→", "\\int": "∫", "\\lim": "lim"}
MAX_STROKES = 30
STROKE_HEIGHT, PEN = 100, 5  # drawn so that a tall stroke is this high in px and the pen this wide, as in real boards people wrote


def _group(tokens, i):
    """The text of the {...} starting at tokens[i], and where it ends."""
    if tokens[i] != "{":
        raise ValueError
    text, i = _sequence(tokens, i + 1)
    return text, i + 1


def _sequence(tokens, i):
    out = ""
    while i < len(tokens) and tokens[i] != "}":
        token = tokens[i]
        if token == "\\frac":
            top, i = _group(tokens, i + 1)
            bottom, i = _group(tokens, i)
            out += f"({top})/({bottom})"
        elif token == "\\sqrt":
            inside, i = _group(tokens, i + 1)
            out += f"√({inside})"
        elif token in ("^", "_"):
            script, i = _group(tokens, i + 1)
            out += f"{token}({script})"
        elif token in COMMANDS:
            out += COMMANDS[token]
            i += 1
        elif token.lower() in PLAIN and token != "I":  # the board reads a capital as its small letter
            out += token.lower()
            i += 1
        else:
            raise ValueError
    return out, i


def expected_text(latex):
    """What the reader should write for a LaTeX expression, or None if the board cannot write all of it.

    That also rules out letters the reader keeps for function words (layout.MAIN_LINE): alone on a line they are never read.
    """
    tokens = re.findall(r"\\[a-zA-Z]+|\S", latex)
    try:
        text, end = _sequence(tokens, 0)
    except (ValueError, IndexError):
        return None
    if end != len(tokens) or MAIN_LINE & set(re.sub("|".join(WORDS), "", text)):
        return None
    return comparable(text)


def comparable(text):
    """The same reading written the same way: no spaces, one mark for times, divide and dot, no brackets round one symbol."""
    for shown, plain in (("×", "x"), ("÷", "/"), ("·", "."), ("−", "-"), (" ", "")):
        text = text.replace(shown, plain)
    return re.sub(r"\(([^()])\)", r"\1", text)


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
    return row[-1]


def stroke_layers(strokes):
    """Real pen strokes -> (board mask, one mask per stroke), as reader.stroke_masks makes them from the canvas."""
    points = [np.array(s, float) for s in strokes]
    heights = sorted(np.ptp(p[:, 1]) for p in points)
    scale = STROKE_HEIGHT / max(heights[3 * len(heights) // 4], 1e-6)
    low = np.min([p.min(axis=0) for p in points], axis=0)
    size = np.ceil((np.max([p.max(axis=0) for p in points], axis=0) - low) * scale).astype(int) + 4 * PEN
    if size.max() > 4000:  # a stray far-away point: not something a board could hold
        return None, None
    layers = []
    for p in points:
        layer = np.zeros((size[1], size[0]), np.uint8)
        pts = np.rint((p - low) * scale + 2 * PEN).astype(np.int32)
        if len(pts) == 1:
            pts = np.vstack([pts, pts + 1])
        cv2.polylines(layer, [pts], False, 255, PEN - 1)
        layers.append(layer)
    return np.max(layers, axis=0), layers


def check_symbols(models, data):
    report = {}
    with np.load(pathlib.Path(data) / "symbols.npz") as cache:
        tests = {name: (cache[f"{key}_test_x"].astype(np.float32) / 255, cache[f"{key}_test_y"]) for name, key in REAL.items()}
    groups = {CLASSES.index(c): k for k, shapes in enumerate(SAME_SHAPE) for c in shapes}
    together = np.array([groups.get(c, 10 + c) for c in range(len(CLASSES))])
    predictions = {name: {source: model.predict(x[..., None], batch_size=512, verbose=0).argmax(axis=1) for source, (x, _) in tests.items()}
                   for name, model in models.items()}
    y_all = np.concatenate([y for _, y in tests.values()])
    for name in models:
        by_source = {source: round(float((predictions[name][source] == y).mean()), 4) for source, (_, y) in tests.items()}
        pred_all = np.concatenate([predictions[name][source] for source in tests])
        per_class = {SHOWN.get(c, c): round(float((pred_all[y_all == k] == k).mean()), 4) for k, c in enumerate(CLASSES)}
        report[name] = {"images": len(y_all), "accuracy": round(float((pred_all == y_all).mean()), 4),
                        "accuracy_same_shapes_together": round(float((together[pred_all] == together[y_all]).mean()), 4),
                        "by_source": by_source, "per_class": per_class}
    report["images_by_source"] = {source: len(y) for source, (_, y) in tests.items()}
    return report


def check_expressions(models, data):
    with gzip.open(pathlib.Path(data) / "real_expressions.json.gz", "rt", encoding="utf-8") as f:
        expressions = json.load(f)
    usable = [(e, expected_text(e["label"])) for e in expressions if len(e["strokes"]) <= MAX_STROKES]
    usable = [(e, want) for e, want in usable if want]
    totals = {name: {"exact": 0, "errors": 0, "misses": []} for name in models}
    length, read = 0, 0
    for done, (entry, want) in enumerate(usable):
        mask, layers = stroke_layers(entry["strokes"])
        if mask is None:
            continue
        symbols = segment(mask, layers)
        length += len(want)
        read += 1
        for name, model in models.items():
            mine = copy.deepcopy(symbols)
            todo = [s for s in mine if s["kind"] is None]
            if todo:
                probs = model(np.stack([s["image"] for s in todo])[..., None], training=False).numpy()
                for s, p in zip(todo, probs):
                    s["probs"] = p
            got = comparable(to_text(mine, CLASSES)) if mine else ""
            totals[name]["exact"] += got == want
            totals[name]["errors"] += edit_distance(got, want)
            if got != want and len(totals[name]["misses"]) < 12:
                totals[name]["misses"].append(f"{want} -> {got}")
        if (done + 1) % 100 == 0:
            print(done + 1, {name: t["exact"] for name, t in totals.items()}, flush=True)
    return {"in_test_split": len(expressions), "board_can_write": len(usable), "read": read,
            **{name: {"read_exactly": t["exact"], "exact_share": round(t["exact"] / max(read, 1), 4),
                      "character_error_rate": round(t["errors"] / max(length, 1), 4), "examples_of_misses": t["misses"]}
               for name, t in totals.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path of a .keras file with the same classes; repeat to compare")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        models[name] = tf.keras.models.load_model(path)
    results = {"symbols": check_symbols(models, args.data)}
    print(json.dumps(results["symbols"], indent=1, ensure_ascii=False), flush=True)
    results["expressions"] = check_expressions(models, args.data)
    print(json.dumps(results["expressions"], indent=1, ensure_ascii=False), flush=True)
    (HERE / "reports" / "real_handwriting_check.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
