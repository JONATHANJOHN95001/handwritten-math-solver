"""Turn a canvas drawing into separate symbols and 28x28 images for the CNN (layout.py reads their arrangement)."""
import functools

import cv2
import numpy as np

SIZE, BOX = 28, 20  # MNIST layout: the symbol fits a 20 px box centred in a 28 px square
WORK, PEN = 80, 9   # strokes are redrawn at this pen width (px) on a symbol scaled to WORK px
MIN_AREA = 30       # px; smaller blobs are stray marks
MIN_PEN_AREA = 8    # px; a pen stroke was meant, so even a thin pen's tap counts: it is the dot of an i, a ! or a decimal
TALL_BESIDE = 2.5   # a stroke this many times taller than its neighbour is a different symbol: ∫ and its limit


def ink_mask(rgba):
    """Canvas RGBA -> uint8 mask that is 255 where there is ink."""
    if rgba.shape[2] == 4 and np.median(rgba[..., 3]) < 128:  # mostly transparent: alpha is the ink
        return np.where(rgba[..., 3] > 100, 255, 0).astype(np.uint8)
    # opaque background: ink is whatever differs enough from the background colour (light pencil included)
    gray = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.int16)
    return np.where(np.abs(gray - int(np.median(gray))) > 60, 255, 0).astype(np.uint8)


def _fit(mask, box):
    """Crop to the ink and scale it so its longer side is `box` px, keeping its shape."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = box / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def normalize(mask):
    """Redraw the symbol with one standard pen width, scale it into a 20x20 box and centre it in 28x28.

    Thinning to a 1 px skeleton and re-thickening removes stroke width as a clue, so a thick
    canvas pen, thin dataset ink and MNIST strokes all look alike to the CNN.
    """
    scaled = _fit(mask, WORK)
    big = np.where(scaled > (127 if (scaled > 127).any() else 0), 255, 0).astype(np.uint8)  # a long thin line can fade when it shrinks
    big = cv2.copyMakeBorder(big, PEN, PEN, PEN, PEN, cv2.BORDER_CONSTANT, value=0)
    pen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PEN, PEN))
    skeleton = cv2.ximgproc.thinning(big)
    if not skeleton.any():  # thinning can wear a round blob away to nothing
        skeleton = big
    crop = _fit(cv2.dilate(skeleton, pen), BOX)
    out = np.zeros((SIZE, SIZE), np.float32)
    top, left = (SIZE - crop.shape[0]) // 2, (SIZE - crop.shape[1]) // 2
    out[top:top + crop.shape[0], left:left + crop.shape[1]] = crop / 255.0
    return out


def _centre_line(obj):
    """A canvas object's centre line on the board: a Line, or a pen path (Move changes its left and top, not its points)."""
    if obj["type"] == "line":
        return [(obj["left"] + obj["x1"], obj["top"] + obj["y1"]), (obj["left"] + obj["x2"], obj["top"] + obj["y2"])]
    points = [point for command in obj["path"] for point in zip(command[1::2], command[2::2])]
    half = obj["strokeWidth"] / 2
    dx, dy = obj["left"] + half - min(x for x, _ in points), obj["top"] + half - min(y for _, y in points)
    return [(x + dx, y + dy) for x, y in points]


def stroke_masks(objects, board):
    """One mask per pen stroke, from the canvas's own record of what was drawn, with erased parts cut out.

    None when the record cannot be trusted (stretched or turned with Move, or not what the board shows):
    the reader then works from the board's pixels alone.
    """
    masks = []
    try:
        for obj in objects:
            if obj["scaleX"] != 1 or obj["scaleY"] != 1 or obj["angle"]:
                return None
            layer = np.zeros(board.shape, np.uint8)
            # a thickness of t draws t + 2 px wide here, against t + 1 on the canvas, and the CNN can tell a fat 1 from a thin one
            cv2.polylines(layer, [np.rint(_centre_line(obj)).astype(np.int32)], False, 255, max(1, round(obj["strokeWidth"]) - 1))
            if obj["stroke"].upper() == "#FFFFFF":  # the eraser paints the board colour
                for mask in masks:
                    mask[layer > 0] = 0
            else:
                masks.append(layer)
    except (KeyError, TypeError, ValueError):
        return None
    if not masks:
        return None
    drawn, seen = np.max(masks, axis=0) > 0, board > 0
    return masks if np.count_nonzero(drawn & seen) >= 0.6 * np.count_nonzero(drawn | seen) else None


def _pieces(layers):
    """Every connected piece of ink in the layers: their OpenCV stats rows, and ink(i), the pixels of piece i."""
    rows, where = [], []
    for layer in layers:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(layer, connectivity=8)
        rows.extend(stats[1:n])
        where.extend((labels, j) for j in range(1, n))

    @functools.lru_cache(maxsize=None)  # the pairs of strokes ask for the same pieces again and again, and a big board makes each one slow
    def ink(i):
        return (where[i][0] == where[i][1]).astype(np.uint8)

    return np.array(rows).reshape(-1, 5), ink


def segment(mask, pen_strokes=None):
    """Split the drawing into symbols, left to right.

    Strokes join into one symbol when they overlap left-to-right and touch or nearly touch (a '5'
    with its own top bar, a two-stroke 'x'), when they are the two bars of '=', when one is a
    dot just above or below another stroke (the dot of 'i', the dots of '÷') or when one is a small
    extra mark on another (a slip of the pen at the foot of a '3'). Strokes stacked with a
    clear gap stay apart, so a fraction's top and bottom and the 'x→0' under 'lim' are separate.
    A fraction bar, a digit under a root and a small limit beside a tall ∫ stay apart even when they touch.
    pen_strokes (one mask per stroke, from stroke_masks) keeps symbols that touch apart: from the
    pixels alone a 2 whose tail runs into the next x is one blob.
    Each symbol is a dict: 'box' (x, y, w, h), 'kind' ('=', '÷' or 'dot' when the shape alone
    decides, else None) and 'image' (the 28x28 CNN input when kind is None).
    """
    stats, ink = _pieces([mask] if pen_strokes is None else pen_strokes)
    parts = [i for i in range(len(stats)) if stats[i, cv2.CC_STAT_AREA] >= (MIN_AREA if pen_strokes is None else MIN_PEN_AREA)]
    tallest = max((stats[i, cv2.CC_STAT_HEIGHT] for i in parts), default=0)
    group = {i: i for i in parts}

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    def centre_is_ink(i):  # next to tall matrix brackets an ordinary 0 is small too, but a ring's middle is empty
        x, y, w, h = stats[i, :4]
        return ink(i)[y + h // 2, x + w // 2] > 0

    def touch(a, b):
        ax, ay, aw, ah = stats[a, :4]
        bx, by, bw, bh = stats[b, :4]
        if ax > bx + bw or bx > ax + aw or ay > by + bh or by > ay + ah:
            return False  # their boxes are too far apart to touch
        return np.any(cv2.dilate(ink(a), np.ones((3, 3), np.uint8)) & ink(b))

    def close(a, b):  # how far a is from the ink of b, against the narrower side of b
        return cv2.distanceTransform(1 - ink(b), cv2.DIST_L2, 3)[ink(a) > 0].min() < 0.15 * min(stats[b, 2], stats[b, 3])

    def in_root(a, b):  # a digit under a root or above its tick: small, inside the box of a stroke with a long top bar, and not crossed by it
        for inner, outer in ((a, b), (b, a)):
            ix, iy, iw, ih = stats[inner, :4]
            ox, oy, ow, oh = stats[outer, :4]
            if (not _flat(stats[inner]) and ow > 1.2 * iw and oh > 1.05 * ih and ox <= ix + iw / 2 <= ox + ow and oy <= iy + ih / 2 <= oy + oh
                    and ink(outer)[oy:oy + max(2, oh // 6), ox:ox + ow].any(axis=0).mean() > 0.5
                    and not ink(outer)[iy + ih // 4:iy + ih - ih // 4, ix + iw // 4:ix + iw - iw // 4].any()):
                return True
        return False

    def beside_tall(a, b):  # ∫ and a small limit, a tall bracket and a power: a far shorter stroke next to a tall thin one, not across it
        small, tall = (a, b) if stats[a, 3] < stats[b, 3] else (b, a)
        x, y, w, h = stats[small, :4]
        if (stats[tall, 3] < TALL_BESIDE * h or stats[tall, 3] < TALL_BESIDE * stats[tall, 2] or _flat(stats[small])
                or _gap(stats[small], stats[tall]) > 0):
            return False
        return not ink(tall)[y:y + h, x + w // 4:x + w - w // 4].any()

    def back_to_back(a, b):  # a curly x as two arcs, apart or overlapping: ')' then '(' bending towards each other, no taller than the writing around them
        left, right = (a, b) if stats[a, 0] < stats[b, 0] else (b, a)
        top = max(stats[left, 1], stats[right, 1])
        bottom = min(stats[left, 1] + stats[left, 3], stats[right, 1] + stats[right, 3])
        tall = max(stats[left, 3], stats[right, 3])
        if (bottom - top < 0.7 * tall or tall > 1.25 * typical or stats[right, 0] - stats[left, 0] - stats[left, 2] > 0.35 * tall
                or stats[right, 0] + stats[right, 2] - stats[left, 0] > 2.2 * tall):
            return False
        return _bends(ink(left)[top:bottom], ")") and _bends(ink(right)[top:bottom], "(")

    dots = [i for i in parts if _is_dot(stats[i], tallest) and centre_is_ink(i)]
    strokes = [i for i in parts if i not in dots]
    typical = np.median(stats[strokes, 3]) if strokes else 0
    upright = [i for i in strokes if not _flat(stats[i])]
    bars = {i for i in strokes if _flat(stats[i]) and _has_both_sides(stats[i], stats[[s for s in upright if s != i]])}
    for k, a in enumerate(strokes):
        for b in strokes[k + 1:]:
            if a in bars or b in bars or beside_tall(a, b) or in_root(a, b):
                continue  # what is above or below a fraction bar may touch it, but is never part of it
            if pen_strokes is not None and touch(a, b):
                same = _touching_join(stats[a], stats[b])
            else:
                same = _together(stats[a], stats[b], tallest)
            if same or back_to_back(a, b):
                group[root(a)] = root(b)
    for a in strokes:
        for b in strokes:
            if a != b and _is_small_mark(stats[a], stats[b]) and close(a, b):
                group[root(a)] = root(b)
    for d in dots:  # a dot joins the nearest stroke straight above or below it (an 'i', a '÷'), if close
        centre, near = stats[d, 0] + stats[d, 2] / 2, stats[d, 2]  # a little to one side still counts, but not beside it: that is a decimal point
        below_or_above = [(_gap(stats[d], stats[s]), s) for s in strokes
                          if stats[s, 0] <= centre <= stats[s, 0] + stats[s, 2]
                          or (_gap(stats[d], stats[s]) >= 0 and stats[s, 0] - near <= centre <= stats[s, 0] + stats[s, 2] + near)]
        if below_or_above:
            gap, s = min(below_or_above)
            if gap < 0.9 * max(stats[s, 2], stats[s, 3]):
                group[root(d)] = root(s)

    members = {}
    for i in parts:
        members.setdefault(root(i), []).append(i)
    symbols = []
    for ids in sorted(members.values(), key=lambda ids: min(stats[i, cv2.CC_STAT_LEFT] for i in ids)):
        boxes = [stats[i, :4] for i in ids]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        kind = _shape_kind([stats[i] for i in ids], len(ids) == 1 and ids[0] in dots, tallest)
        image = None
        if kind is None:
            image = normalize(np.max([ink(i)[y0:y1, x0:x1] for i in ids], axis=0) * 255)
        small_stroke = len(ids) == 1 and kind is None and _is_small_stroke(stats[ids[0]], tallest)
        symbols.append({"box": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), "kind": kind, "image": image,
                        "small_stroke": small_stroke, **_stem_and_dot([stats[i] for i in ids], [i in dots for i in ids])})
    return symbols


def _stem_and_dot(pieces, is_dot):
    """'exclaim': a stem with a dot below it, the one shape '!' can have (to the CNN a short 1 or an 'i' can look just like one).
    'dotted': a stem with a dot above it, an 'i'."""
    if sorted(is_dot) != [False, True]:
        return {"exclaim": False, "dotted": False}
    dot, stem = (pieces[0], pieces[1]) if is_dot[0] else (pieces[1], pieces[0])
    return {"exclaim": bool(dot[1] >= stem[1] + stem[3]), "dotted": bool(dot[1] + dot[3] <= stem[1])}


def _flat(box):
    return box[2] > 2 * box[3]  # more than twice as wide as tall: a bar


def _has_both_sides(bar, others):
    """Is there a stroke above this bar and one below it, as a fraction has? A 5's top bar or the crossbar of a t has one side at most,
    or is no wider than the stroke it belongs to."""
    centres_x = others[:, 0] + others[:, 2] / 2
    over = (bar[0] <= centres_x) & (centres_x <= bar[0] + bar[2]) & (others[:, 2] < bar[2] / 1.2)
    middle = bar[1] + bar[3] / 2
    return bool(((others[:, 1] + others[:, 3] <= middle) & over).any() and ((others[:, 1] >= middle) & over).any())


def _equals_bars(a, b, tallest):
    """Two bars, one above the other and about as wide: '='. Measured on 4,758 real ones, which lean and differ far more than neat ones."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a[:4], b[:4]
    flat = (_flat(a) and _flat(b)) or (aw > 1.4 * ah and bw > 1.4 * bh and max(ah, bh) < 0.4 * tallest)
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    gap = _gap(a, b)
    return flat and x_overlap > 0.5 * min(aw, bw) and max(aw, bw) < 2.2 * min(aw, bw) and -0.15 * max(aw, bw) <= gap < 1.2 * max(aw, bw)


def _is_dot(stat, tallest):
    """Small, roughly round and filled in: a dot, not a small '0', '1' or '−'."""
    x, y, w, h, area = stat[:5]
    return max(w, h) < 0.3 * tallest and w <= 2 * h and h <= 2 * w and area > 0.5 * w * h


def _is_small_stroke(stat, tallest):
    """A short thin stroke: where it stands decides whether it is a comma, a prime or a small 1."""
    w, h = stat[2], stat[3]
    return max(w, h) < 0.4 * tallest and h > 1.3 * w


def _gap(a, b):
    """Vertical gap between two OpenCV stats rows; negative when they overlap."""
    return max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])


def _inside(inner, outer):
    """Is one stroke well inside the box of a much larger one, like a digit under the bar of a square root?"""
    (ix, iy, iw, ih), (ox, oy, ow, oh) = inner[:4], outer[:4]
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh and ow > 1.5 * iw and oh > 1.2 * ih


def _is_small_mark(inner, outer):
    """A short thin stroke inside another stroke's box, small even beside its narrower side: a slip of the pen."""
    iw, ih = inner[2], inner[3]
    return _inside(inner, outer) and max(iw, ih) < 0.4 * min(outer[2], outer[3]) and (iw > 2 * ih or ih > 2 * iw)


def _bends(rows, shape):
    """Does a stroke (its rows of ink) bend like ')' or '(': both its edges further that way at its middle than at its two ends?"""
    rows = rows[rows.any(axis=1)]
    if len(rows) < 6:
        return False
    lefts, rights = rows.argmax(axis=1), rows.shape[1] - 1 - rows[:, ::-1].argmax(axis=1)
    third, margin = len(rows) // 3, 0.05 * len(rows)
    for edge in (lefts, rights):
        middle, ends = edge[third:-third].mean(), (edge[:third // 2 + 1].mean(), edge[-(third // 2 + 1):].mean())
        if (shape == ")" and middle < max(ends) + margin) or (shape == "(" and middle > min(ends) - margin):
            return False
    return True


def _touching_join(a, b):
    """Do two pen strokes whose ink touches make one symbol? Touching alone does not say so: a 2's tail can run into an x."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a[:4], b[:4]
    if min(ax + aw, bx + bw) - max(ax, bx) > 0.5 * min(aw, bw):  # one across or above the other: '+', 't', a '5' and its top bar
        return True
    wide, tall = max(ax + aw, bx + bw) - min(ax, bx), max(ay + ah, by + bh) - min(ay, by)
    if min(ah, bh) >= 0.7 * max(ah, bh) and wide <= 1.6 * tall:  # side by side at one height: a 'v', a curly 'x' drawn as two arcs
        return True
    for stem, arm in ((a, b), (b, a)):  # a straight stem with a shorter stroke against its side: k, h, b, d, 4
        if stem[3] >= 4 * stem[2] and arm[3] < 0.85 * stem[3] and arm[1] >= stem[1] and arm[1] + arm[3] <= stem[1] + stem[3] and wide <= 1.1 * tall:
            return True
    for bar, other in ((a, b), (b, a)):  # the top bar of a square root, starting from the top of its tick
        if _flat(bar) and bar[0] > other[0] and abs(bar[1] - other[1]) < 0.25 * other[3]:
            return True
    return False


def _together(a, b, tallest):
    """Do two strokes (OpenCV stats rows, not dots) belong to the same symbol?"""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a[:4], b[:4]
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    gap = _gap(a, b)
    close_to_a_bar = (_flat(a) or _flat(b)) and -0.1 * max(ah, bh) <= gap < 0.15 * max(ah, bh)
    if close_to_a_bar and x_overlap > 0.25 * min(aw, bw) and not (_inside(a, b) or _inside(b, a)):  # the top bar of a 5, a T or a π
        return True
    if x_overlap <= 0.5 * min(aw, bw) or _inside(a, b) or _inside(b, a):
        return False
    if _equals_bars(a, b, tallest):
        return True
    return gap <= 0


def _shape_kind(stats, lone_dot, tallest):
    """'=' for two stacked bars, '÷' for a bar with a dot above and below it, 'dot' for a lone dot, else None.

    A dot is judged by its size next to the tallest stroke, which the CNN cannot see: its input is
    always scaled to the same 20 px box.
    """
    if lone_dot:
        return "dot"
    boxes = sorted((s[:4] for s in stats), key=lambda b: b[1])
    if len(boxes) == 2 and (all(_flat(b) for b in boxes) or _equals_bars(*boxes, tallest)):
        return "="
    if len(boxes) == 3:
        top, bar, bottom = boxes
        small = lambda b: max(b[2], b[3]) < 0.6 * bar[2]
        if (_flat(bar) and small(top) and small(bottom)
                and top[1] + top[3] <= bar[1] and bottom[1] >= bar[1] + bar[3]):
            return "÷"
    return None

