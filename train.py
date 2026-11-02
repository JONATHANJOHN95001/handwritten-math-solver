"""Train the symbol CNN on real handwriting (see prepare_handwriting.py) and a few symbols drawn by this script,
then save the model, labels and report figures.

Usage: python train.py --data <folder made by prepare_handwriting.py>
"""
import argparse
import json
import pathlib
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from layout import to_text
from reader import normalize, segment

# No dot or = here: the reader knows them by their shape, which the CNN never sees. x is apart from the letters as it is × too.
LETTERS = list("abcdefghijklmnopqrstuvwyz")
DRAWN = ["lparen", "rparen", "arrow", "inf", "integral", "sqrt", "pi", "lbracket", "rbracket", "partial", "fact",
         "lambda", "mu", "theta"]  # also drawn by this script, so that the plain textbook form is always among the real ones
CLASSES = [str(d) for d in range(10)] + ["plus", "minus", "x", "slash"] + LETTERS + DRAWN
SHOWN = {"plus": "+", "minus": "−", "slash": "÷ (/)", "lparen": "(", "rparen": ")", "arrow": "→", "inf": "∞", "integral": "∫",
         "sqrt": "√", "pi": "π", "lbracket": "[", "rbracket": "]", "partial": "∂", "fact": "!", "lambda": "λ", "mu": "μ", "theta": "θ"}
REAL = {"BHMSDS": "bhmsds", "EMNIST ByClass": "emnist", "HASYv2": "hasy", "MathWriting+": "plus"}  # names in the cache
PER_DRAWN = 500
SEED = 7
HERE = pathlib.Path(__file__).parent


def _arc(cx, cy, rx, ry, start, end):
    t = np.radians(np.linspace(start, end, 40))
    return np.stack([cx + rx * np.cos(t), cy - ry * np.sin(t)], axis=1)


def _turn(v, degrees):
    a = np.radians(degrees)
    return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])


def _new_symbol_strokes(name, rng):
    """Strokes for ∫ √ π [ ] ∂ ! λ μ θ, with the corners moved a little each time."""
    def at(x, y):
        return (x + rng.normal(0, 3), y + rng.normal(0, 3))

    if name == "integral":
        height, lean, hook = rng.uniform(90, 118), rng.uniform(1, 8), rng.uniform(10, 17)  # big hooks: it must not look like /
        top = _arc(64 + lean + hook, 64 - height / 2 + hook, hook, hook, 10, 180)
        bottom = _arc(64 - lean - hook, 64 + height / 2 - hook, hook, hook, 0, -170)
        return [np.concatenate([top, bottom])]
    if name == "sqrt":
        left, bar, top, low = rng.uniform(14, 30), rng.uniform(25, 70), rng.uniform(22, 36), rng.uniform(95, 108)
        return [np.array([at(left - 8, 70), at(left, 64), at(left + 13, low), at(left + 28, top), (left + 28 + bar, top + rng.normal(0, 2))])]
    if name == "pi":
        return [np.array([at(28, 42), at(100, 38)]), np.array([at(50, 40), at(46, 100)]), np.array([at(80, 40), at(80, 88), at(94, 98)])]
    if name in ("lbracket", "rbracket"):
        side, high = rng.uniform(24, 42), rng.uniform(36, 52)
        x = 64 - side / 2 if name == "lbracket" else 64 + side / 2
        out = side if name == "lbracket" else -side
        return [np.array([at(x + out, 64 - high), (x, 64 - high), (x, 64 + high), at(x + out, 64 + high)])]
    if name == "partial":
        bowl = _arc(60, 84, 25, 24, 40, 400)
        return [np.concatenate([bowl, np.array([at(86, 62), at(84, 40), at(70, 24), at(52, 22), at(40, 30)])])]
    if name == "fact":
        return [np.array([at(64, 10), at(64, 82)]), _arc(64, 112, 4, 4, 0, 360)]
    if name == "lambda":
        return [np.array([at(42, 18), at(62, 58), at(92, 110)]), np.array([at(62, 56), at(34, 110)])]
    if name == "mu":
        return [np.array([at(40, 22), at(40, 112)]), np.array([at(40, 66), at(52, 84), at(72, 84), at(84, 68), at(84, 24)]), np.array([at(84, 68), at(94, 84)])]
    return [_arc(64, 64, rng.uniform(22, 32), rng.uniform(40, 50), 0, 360), np.array([at(36, 64), at(92, 64)])]  # theta


def draw_symbol(name, rng):
    """A hand-drawn-looking '(', ')', '→', '∞', curly x ( a ')' and a '(' back to back) or new symbol on a 128x128 mask."""
    if name in ("integral", "sqrt", "pi", "lbracket", "rbracket", "partial", "fact", "lambda", "mu", "theta"):
        strokes = _new_symbol_strokes(name, rng)
    elif name in ("lparen", "rparen"):
        span, mid = rng.uniform(100, 160), 180 if name == "lparen" else 0
        strokes = [_arc(64, 64, rng.uniform(15, 40), rng.uniform(35, 55), mid - span / 2, mid + span / 2)]
    elif name == "arrow":
        length, tilt = rng.uniform(60, 100), rng.uniform(-8, 8)
        tail, tip = np.array([64 - length / 2, 64 - tilt / 2]), np.array([64 + length / 2, 64 + tilt / 2])
        along, angle, head = (tip - tail) / np.linalg.norm(tip - tail), rng.uniform(25, 50), rng.uniform(0.15, 0.35) * length
        strokes = [np.array([tail, tip]), np.array([tip - head * _turn(along, angle), tip, tip - head * _turn(along, -angle)])]
    elif name == "slash":  # BHMSDS has slashes from only a few writers; MNIST 1s lean too, so add clear ones
        lean, length = np.radians(rng.uniform(25, 60)), rng.uniform(70, 110)
        half = length / 2 * np.array([np.sin(lean), -np.cos(lean)])
        bow = rng.uniform(-6, 6) * np.array([np.cos(lean), np.sin(lean)])
        strokes = [np.array([64 - half, 64 + bow, 64 + half])]
    elif name == "inf":
        a, squash, t = rng.uniform(40, 55), rng.uniform(0.6, 1.3), np.linspace(0, 2 * np.pi, 80)
        d = 1 + np.sin(t) ** 2
        strokes = [np.stack([64 + a * np.cos(t) / d, 64 + squash * a * np.sin(t) * np.cos(t) / d], axis=1)]
    else:  # curly x
        rx, ry = rng.uniform(25, 45), rng.uniform(25, 40)
        strokes = [_arc(64 - rx, 64, rx, ry, -70, 70), _arc(64 + rx, 64, rx, ry, 110, 250)]
    tilt = np.radians(rng.uniform(-8, 8))
    rotate = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
    canvas = np.zeros((128, 128), np.uint8)
    width = int(rng.integers(4, 12))
    for pts in strokes:
        pts = (pts - 64) @ rotate.T + 64 + rng.normal(0, 1.0, pts.shape)  # tilt and a shaky hand
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, 255, width, cv2.LINE_AA)
    return np.where(canvas > 127, 255, 0).astype(np.uint8)


def load_drawn():
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for name, count in [(n, PER_DRAWN) for n in DRAWN] + [("x", PER_DRAWN), ("slash", PER_DRAWN)]:
        xs += [np.rint(normalize(draw_symbol(name, rng)) * 255).astype(np.uint8) for _ in range(count)]
        ys += [CLASSES.index(name)] * count
    return np.array(xs), np.array(ys)


def load_test(data):
    """The test images, scaled 0-1. Real handwriting only: a score on this script's own drawings says nothing about handwriting."""
    with np.load(pathlib.Path(data) / "symbols.npz") as cache:
        x = np.concatenate([cache[f"{key}_test_x"] for key in REAL.values()])
        y = np.concatenate([cache[f"{key}_test_y"] for key in REAL.values()])
    return x.astype(np.float32) / 255, y


def load_split(data):
    """Each source's own training images, one tenth of them kept for validation, unless the source has a validation split."""
    trains, vals, sources = [], [], {}
    with np.load(pathlib.Path(data) / "symbols.npz") as cache:
        for name, key in REAL.items():
            x, y = cache[f"{key}_train_x"], cache[f"{key}_train_y"]
            sources[name] = (x, y)
            if f"{key}_val_x" in cache:
                trains.append((x, y))
                vals.append((cache[f"{key}_val_x"], cache[f"{key}_val_y"]))
            else:
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, stratify=y, random_state=SEED)
                trains.append((x_train, y_train))
                vals.append((x_val, y_val))
    sources["drawn"] = load_drawn()
    trains.append(sources["drawn"])
    x_train, y_train = (np.concatenate(parts) for parts in zip(*trains))
    x_val, y_val = (np.concatenate(parts) for parts in zip(*vals))
    return sources, (x_train, y_train), (x_val.astype(np.float32) / 255, y_val), load_test(data)


def build_model(n_classes):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        tf.keras.Input((28, 28, 1)),
        layers.RandomRotation(0.03, fill_mode="constant"),  # about ±11 degrees
        layers.RandomZoom(0.1, fill_mode="constant"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])


def expression_check(model, images, labels, n=300):
    """Glue random test symbols into sums like '47+3×12', run the full reader on them, count exact reads."""
    rng = np.random.default_rng(SEED)
    by_class = [images[labels == c] for c in range(len(CLASSES))]
    ops = [CLASSES.index(o) for o in ("plus", "minus", "x", "slash")]
    correct = 0
    for _ in range(n):
        seq = []
        for k in range(rng.integers(2, 4)):  # 2 or 3 numbers of 1 or 2 digits
            if k:
                seq.append(rng.choice(ops))
            seq += list(rng.integers(0, 10, rng.integers(1, 3)))
        canvas = np.zeros((112, 112 * len(seq)), np.uint8)
        for i, c in enumerate(seq):
            tile = by_class[c][rng.integers(len(by_class[c]))]
            canvas[:, 112 * i:112 * (i + 1)] = cv2.resize((tile * 255).astype(np.uint8), (112, 112))
        symbols = segment(np.where(canvas > 64, 255, 0).astype(np.uint8))  # low cut keeps thin strokes joined
        to_classify = [s for s in symbols if s["kind"] is None]
        if len(symbols) != len(seq) or not to_classify:
            continue
        probs = model(np.stack([s["image"] for s in to_classify])[..., None], training=False).numpy()
        for s, p in zip(to_classify, probs):
            s["probs"] = p
        want = "".join({"plus": "+", "minus": "-", "x": "×", "slash": "÷"}.get(CLASSES[c], CLASSES[c]) for c in seq)
        correct += to_text(symbols, CLASSES) == want
    return correct / n


def save_figures(history, model, x_test, y_test, pred, reports):
    names = [SHOWN.get(c, c) for c in CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set(title=key.capitalize(), xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "training_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(13, 12))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=False, include_values=False)  # too many classes to print a count in every cell
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=150)

    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(x_test), 30, replace=False)
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.2))
    for ax, i in zip(axes.flat, picks):
        ax.imshow(x_test[i], cmap="gray")
        ok = pred[i] == y_test[i]
        ax.set_title(f"{names[pred[i]]}" + ("" if ok else f" (true {names[y_test[i]]})"),
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample test predictions (red = wrong)")
    fig.tight_layout()
    fig.savefig(reports / "sample_predictions.png", dpi=150)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder made by prepare_handwriting.py")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the report prints − × ÷, which a Windows console cannot encode
    tf.keras.utils.set_random_seed(SEED)

    sources, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_split(args.data)
    print(", ".join(f"{len(ys)} {name}" for name, (_, ys) in sources.items()), f"images, {len(CLASSES)} classes", flush=True)

    # No class weights: digits outnumber the rest here as in real sums, and balanced weights pushed every doubtful '1' over to '/'.
    # Images stay as bytes until a batch needs them, as there are too many to hold as floats.
    batches = (tf.data.Dataset.from_tensor_slices((x_train[..., None], y_train)).shuffle(len(y_train), seed=SEED).batch(128)
               .map(lambda x, y: (tf.cast(x, tf.float32) / 255, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    model = build_model(len(CLASSES))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        batches, validation_data=(x_val[..., None], y_val), validation_batch_size=256,
        epochs=args.epochs, verbose=2,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
                   tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)])

    models, reports = HERE / "models", HERE / "reports"
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    model.save(models / "symbol_cnn.keras")
    (models / "labels.json").write_text(json.dumps(CLASSES))

    pred = model.predict(x_test[..., None], verbose=0).argmax(axis=1)
    names = [SHOWN.get(c, c) for c in CLASSES]
    report = classification_report(y_test, pred, target_names=names, digits=4)
    metrics = {
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "expression_accuracy": round(expression_check(model, x_test, y_test), 4),
        "images": {**{name: len(ys) for name, (_, ys) in sources.items()},
                   "train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "epochs_run": len(history.history["loss"]),
    }
    print(report)
    print(json.dumps(metrics, indent=2))

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(reports / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line, **kw: f.write(line + "\n"))
    save_figures(history, model, x_test, y_test, pred, reports)


if __name__ == "__main__":
    main()
