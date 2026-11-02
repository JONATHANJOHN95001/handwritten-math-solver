"""Turn the downloaded handwriting sets into one cache of CNN inputs, so train.py need not read the archives again.

All of it is real handwriting, and each image goes through reader.normalize just as a symbol on the board does:
- EMNIST ByClass (NIST): digits and letters from filled-in forms, with capitals kept apart from small letters.
- MathWriting+ (Google's MathWriting pen strokes, with each symbol's strokes marked by J. Seitz): symbols as people write them
  inside real maths. It holds every symbol the board can read, in the same form the board records: pen strokes.
- HASYv2 (M. Thoma): ∫ √ π ∂ ∞ λ μ θ → [ ] and more as black and white images.
- BHMSDS: digits and operators.
Each source keeps its own test split, or holds one tenth back when it has none. The expressions of the full MathWriting test
split also go into real_expressions.json.gz, for check_handwriting.py.

Usage: python prepare_handwriting.py --bhmsds <folder> --emnist <folder with emnist-byclass-*.gz> --hasy <HASYv2.tar.bz2>
           --plus <start of MathWriting+.zip> --plus-files <its file list> --mathwriting <mathwriting-2024.tgz> --out <folder>
"""
import argparse
import collections
import csv
import gzip
import io
import json
import pathlib
import struct
import sys
import tarfile
import xml.etree.ElementTree as ET
import zlib

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from reader import normalize
from train import CLASSES, SEED

LATEX = {"+": "plus", "-": "minus", "/": "slash", "\\times": "x", "(": "lparen", ")": "rparen", "[": "lbracket", "]": "rbracket",
         "\\rightarrow": "arrow", "\\to": "arrow", "\\longrightarrow": "arrow", "\\infty": "inf", "\\int": "integral",
         "\\sqrt": "sqrt", "\\sqrt{}": "sqrt", "\\pi": "pi", "\\partial": "partial", "!": "fact", "\\lambda": "lambda",
         "\\mu": "mu", "\\theta": "theta"}
EMNIST_SHARE = {"train": {"digit": 5000, "small": 3000, "capital": 1000}, "test": {"digit": 400, "small": 300, "capital": 100}}
PLUS_CAP = {"Train": 6000, "Val": 300, "Test": 1000}  # most images of one class
PER_INK = 3  # most of one symbol from one expression, so that no single writer's habits fill a class
WIDEST = (0.8, 1.8)  # how much narrower and wider training draws a symbol's strokes
KEEP_WIDTH = ("1", "l", "i", "slash")  # a plain stroke stretched sideways only leans more: stretched 1s taught the CNN to call slashes 1
INK = "{http://www.w3.org/2003/InkML}"


def class_index(label):
    """The CNN class of a LaTeX symbol, or None. Capitals join their small letter, as the board has one class for a and A.
    Not I: as a plain bar it is what 1 and l look like, and it would only teach the CNN that a bar may be an i."""
    if label in LATEX:
        return CLASSES.index(LATEX[label])
    if len(label) == 1 and label.isascii() and label.isalnum() and label != "I":
        return CLASSES.index(label.lower())
    return None


def as_image(mask):
    return np.rint(normalize(mask) * 255).astype(np.uint8)


def binarize(gray):
    """Grayscale image -> 255 ink on 0, whatever the original polarity."""
    if gray.mean() > 127:  # dark ink on light paper
        gray = 255 - gray
    return np.where(gray > 127, 255, 0).astype(np.uint8)


def hold_out(images, labels):
    """One tenth of each class for the test set, the same tenth on every run."""
    return train_test_split(np.array(images), np.array(labels), test_size=0.1, stratify=labels, random_state=SEED)


def load_bhmsds(folder):
    images, labels = [], []
    for path in sorted(pathlib.Path(folder).glob("*.png")):
        name = path.stem.rsplit("-", 1)[0]
        if name in CLASSES:
            mask = binarize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            if mask.any():
                images.append(as_image(mask))
                labels.append(CLASSES.index(name))
    return hold_out(images, labels)


def _read_idx(path):
    """Images or labels from an EMNIST .gz file in the MNIST idx format."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    dims = data[3]
    shape = struct.unpack(">" + "I" * dims, data[4:4 + 4 * dims])
    return np.frombuffer(data, np.uint8, offset=4 + 4 * dims).reshape(shape)


def load_emnist(folder, split):
    """A sample of ByClass, whose labels run 0-9, then A-Z, then a-z."""
    folder, rng = pathlib.Path(folder), np.random.default_rng(SEED)
    images = _read_idx(folder / f"emnist-byclass-{split}-images-idx3-ubyte.gz")
    labels = _read_idx(folder / f"emnist-byclass-{split}-labels-idx1-ubyte.gz")
    xs, ys = [], []
    for code, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
        target = class_index(char)
        if target is not None:
            kind = "digit" if char.isdigit() else "small" if char.islower() else "capital"
            where = np.flatnonzero(labels == code)
            pick = rng.choice(where, min(len(where), EMNIST_SHARE[split][kind]), replace=False)
            xs += [as_image(binarize(img.T)) for img in images[pick]]  # EMNIST stores images transposed
            ys += [target] * len(pick)
    return np.array(xs), np.array(ys)


def load_hasy(archive):
    labels, pictures = [], {}
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            if member.name.endswith("hasy-data-labels.csv"):
                labels = list(csv.DictReader(io.StringIO(tar.extractfile(member).read().decode("utf-8"))))
            elif member.name.endswith(".png"):
                pictures[member.name.rsplit("/", 1)[1]] = tar.extractfile(member).read()
    xs, ys = [], []
    for row in labels:
        target = class_index(row["latex"])
        if target is not None:
            data = np.frombuffer(pictures[row["path"].rsplit("/", 1)[1]], np.uint8)
            mask = binarize(cv2.imdecode(data, cv2.IMREAD_GRAYSCALE))
            if mask.any():
                xs.append(as_image(mask))
                ys.append(target)
    return hold_out(xs, ys)


def _points(text):
    """A trace 'x y t, x y t, ...' -> its points (x, y)."""
    return np.array(text.replace(",", " ").split(), float).reshape(-1, 3)[:, :2]


def draw_strokes(strokes, rng, vary_width=False):
    """Pen strokes -> mask, at a size and pen width picked at random from those seen on the board.

    vary_width redraws half of the symbols wider or narrower, as people's writing differs. It is for training only: tests
    keep symbols as they were written.
    """
    if vary_width and rng.random() < 0.5:
        stretch = rng.uniform(WIDEST[0], WIDEST[1])
        strokes = [s * (stretch, 1) for s in strokes]
    low = np.min([s.min(axis=0) for s in strokes], axis=0)
    span = np.max([s.max(axis=0) for s in strokes], axis=0) - low
    size = int(rng.integers(50, 131))
    pen = int(rng.integers(3, max(4, size // 12) + 1))
    scale = size / max(span.max(), 1e-6)
    height, width = (np.ceil(span * scale).astype(int) + 2 * pen + 1)[::-1]
    mask = np.zeros((height, width), np.uint8)
    for stroke in strokes:
        pts = np.rint((stroke - low) * scale + pen).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(mask, tuple(int(v) for v in pts[0]), pen // 2 + 1, 255, -1)
        else:
            cv2.polylines(mask, [pts], False, 255, pen)
    return mask


def _ink_symbols(text):
    """(label, its strokes as text) for each symbol MathWriting+ marks in one ink."""
    root = ET.fromstring(text)
    traces = {t.get("id"): t.text for t in root.iter(INK + "trace")}
    for group in root.find(INK + "traceGroup").findall(INK + "traceGroup"):
        yield group.find(INK + "annotation").text, [traces[v.get("traceDataRef")] for v in group.findall(INK + "traceView")]


def _member(f, entry):
    """One file of the ZIP from its row in the saved file list."""
    _, method, packed, _, local = entry
    f.seek(local)
    name_and_extra = sum(struct.unpack("<HH", f.read(30)[26:30]))
    data = f.read(name_and_extra + packed)[name_and_extra:]
    return zlib.decompress(data, -15) if method == 8 else data


def load_plus(blob, file_list, folders, cap, seed, vary_width=False):
    """Symbols from the given MathWriting+ folders, in random file order, up to cap images of a class.

    blob holds only the start of the ZIP, so files that end past it are skipped. The inks of Symbols and Train come from one
    folder of the original set, so together they count as training, which also varies the width of the writing.
    """
    rng, have = np.random.default_rng(seed), collections.Counter()
    training = "Train" in folders
    xs, ys = [], []
    size = pathlib.Path(blob).stat().st_size
    entries = [e for e in file_list if e[0].endswith(".inkml") and e[0].split("/")[-2] in folders and e[4] + e[2] + 1024 < size]
    with open(blob, "rb") as f:
        for k in rng.permutation(len(entries)):
            from_this_ink = collections.Counter()
            for label, traces in _ink_symbols(_member(f, entries[k])):
                target = class_index(label)
                if target is None or have[target] >= cap or from_this_ink[target] >= PER_INK:
                    continue
                strokes = [_points(t) for t in traces]
                xs.append(as_image(draw_strokes(strokes, rng, vary_width=vary_width and training and CLASSES[target] not in KEEP_WIDTH)))
                ys.append(target)
                have[target] += 1
                from_this_ink[target] += 1
    print(folders, "read", len(entries), "inks", flush=True)
    return np.array(xs), np.array(ys)


def save_real_expressions(archive, path):
    """Every expression of the MathWriting test split: its LaTeX and pen strokes. All of it, so no filter decides what gets tested."""
    kept = []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if "/test/" in member.name and member.name.endswith(".inkml"):
                root = ET.fromstring(tar.extractfile(member).read())
                label = next(a.text for a in root.findall(INK + "annotation") if a.get("type") == "normalizedLabel")
                strokes = [np.round(_points(t.text), 1).tolist() for t in root.iter(INK + "trace")]
                kept.append({"label": label, "strokes": strokes})
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(kept, f)
    print(len(kept), "real expressions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for name in ("bhmsds", "emnist", "hasy", "plus", "plus-files", "mathwriting", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--vary-width", action="store_true", help="also draw half of MathWriting's training symbols wider or narrower; the model in models/ was trained without this")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "real_expressions.json.gz").exists():
        save_real_expressions(args.mathwriting, out / "real_expressions.json.gz")

    file_list = json.loads(pathlib.Path(args.plus_files).read_text(encoding="utf-8"))["entries"]
    arrays = {}
    arrays["bhmsds_train_x"], arrays["bhmsds_test_x"], arrays["bhmsds_train_y"], arrays["bhmsds_test_y"] = load_bhmsds(args.bhmsds)
    print("BHMSDS done", flush=True)
    arrays["hasy_train_x"], arrays["hasy_test_x"], arrays["hasy_train_y"], arrays["hasy_test_y"] = load_hasy(args.hasy)
    print("HASYv2 done", flush=True)
    for split in ("test", "train"):
        arrays[f"emnist_{split}_x"], arrays[f"emnist_{split}_y"] = load_emnist(args.emnist, split)
        print("EMNIST", split, "done", flush=True)
    for split, folders in (("test", ("Test",)), ("val", ("Val",)), ("train", ("Train", "Symbols"))):
        arrays[f"plus_{split}_x"], arrays[f"plus_{split}_y"] = load_plus(args.plus, file_list, folders, PLUS_CAP[folders[0]], SEED, args.vary_width)
    np.savez_compressed(out / "symbols.npz", **arrays)

    sources = sorted({key.rsplit("_", 1)[0] for key in arrays})
    counts = {source: np.bincount(arrays[f"{source}_y"], minlength=len(CLASSES)).tolist() for source in sources}
    (out / "counts.json").write_text(json.dumps({"classes": CLASSES, "counts": counts}))
    print("class".ljust(10), *(source.rjust(13) for source in sources))
    for i, name in enumerate(CLASSES):
        print(name.ljust(10), *(str(counts[source][i]).rjust(13) for source in sources))
    print("total".ljust(10), *(str(sum(counts[source])).rjust(13) for source in sources))


if __name__ == "__main__":
    main()
