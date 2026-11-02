"""Does writing wider or narrower than usual hurt the CNN? The real test symbols of MathWriting+, redrawn stretched sideways.

Usage: python check_wide_writing.py --plus <MathWriting+ zip start> --plus-files <directory.json> --model new=models/symbol_cnn.keras [--model old=<file>]
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

import prepare_handwriting as ph
from train import CLASSES, HERE

STRETCHES = (1.0, 1.25, 1.5, 1.8, 0.75)
MOST_OF_A_CLASS = 300


def test_symbols(blob, files):
    """Up to MOST_OF_A_CLASS real symbols of each class, as (class, pen strokes), in a fixed random order of the test inks."""
    entries = [e for e in json.loads(pathlib.Path(files).read_text(encoding="utf-8"))["entries"] if "/Test/" in e[0] and e[0].endswith(".inkml")]
    order = np.random.default_rng(3).permutation(len(entries))
    have = collections.Counter()
    symbols = []
    with open(blob, "rb") as f:
        for k in order:
            for label, traces in ph._ink_symbols(ph._member(f, entries[k])):
                target = ph.class_index(label)
                if target is not None and have[target] < MOST_OF_A_CLASS:
                    symbols.append((target, [ph._points(t) for t in traces]))
                    have[target] += 1
    return symbols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plus", required=True)
    ap.add_argument("--plus-files", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path of a .keras file with the same classes; repeat to compare")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    symbols = test_symbols(args.plus, args.plus_files)
    y = np.array([target for target, _ in symbols])
    images = {stretch: np.array([ph.as_image(ph.draw_strokes([s * (stretch, 1) for s in strokes], np.random.default_rng(i))) for i, (_, strokes) in enumerate(symbols)], np.float32) / 255
              for stretch in STRETCHES}
    results = {"symbols": len(symbols)}
    for item in args.model:
        name, path = item.split("=", 1)
        model = tf.keras.models.load_model(path)
        results[name] = {str(stretch): round(float((model.predict(images[stretch][..., None], batch_size=512, verbose=0).argmax(axis=1) == y).mean()), 4) for stretch in STRETCHES}
    print(json.dumps(results, indent=1))
    (HERE / "reports" / "wide_writing_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
