"""Read the arrangement of recognised symbols into lines of maths text for the solver.

Writing is two-dimensional: a fraction stacks a top over a bottom, a power sits high, a log base or subscript sits low,
'lim' carries 'x→a' underneath, ∫ carries its limits, a matrix lines symbols up in rows between tall brackets and each new
line of writing is a new equation. This module finds those from the symbol boxes, spots the function words from the CNN's
letter scores, and writes 'lim_(x→0) sin(x)/x', 'log_2(8)', '(1)/(2)', 'x^2', '∫_(0)^(1) x dx', '[[1,2],[3,4]]' and so on.
Separate lines of writing are joined with '; '.
"""
import math
import re

WORDS = ("lim", "log", "sin", "cos", "tan", "ln", "cot", "sec", "exp", "rank", "det", "eigen", "grad", "div", "curl", "mean")
ALL_LETTERS = set("abcdefghijklmnopqrstuvwyz")
WORD_ONLY = ALL_LETTERS - {"e"}  # letters that normally appear only inside a function word (e and x may stand alone)
MAIN_LINE = (WORD_ONLY - set("yztabckmnpd")) | {"arrow", "inf"}  # → and ∞ only belong under 'lim'
# These may stand alone, but only when clearly better than the best other reading: z looks like 2, and [ like (.
CHECKED = set("yztabckmnpd") | {"integral", "sqrt", "pi", "lbracket", "rbracket", "partial", "fact", "lambda", "mu", "theta"}
ALONE_MARGIN = 4
BRACKETS = {"lparen", "rparen"}
ROOT_OVERLAP = 0.3         # a symbol is under a root when at least this much of its width is below the bar: the last digit often sticks out
STEM = {"1": 0.83, "l": 0.15, "i": 0.02}  # how the CNN splits its belief for a plain straight stroke
COMMA_LOOKALIKES = {"1", "l", "i", "j", "slash", "lparen", "rparen", "minus", "fact"}
TOKENS = {"plus": "+", "minus": "-", "slash": "÷", "lparen": "(", "rparen": ")", "arrow": "→", "inf": "∞", "integral": "∫", "sqrt": "√",
          "pi": "π", "lbracket": "[", "rbracket": "]", "partial": "∂", "fact": "!", "lambda": "λ", "mu": "μ", "theta": "θ"}
WORD_BIAS = math.log(0.01)  # a word may score up to 100x lower than the best non-letter reading: l, o and g all have a digit twin
MIN_LETTER = 0.01           # but each of its letters must have at least this much of the CNN's belief
ROW_GAP = 0.5              # a gap this many symbol heights across the whole board starts a new line of writing
BAR_GAP = 1.0               # more room next to a fraction bar before it counts as a new line


def to_text(symbols, labels):
    """symbols: segment() output, each CNN symbol with 'probs' (aligned with labels). Sets 'read_as' on each."""
    items = []
    for s in symbols:
        x, y, w, h = s["box"]
        item = {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "kind": s["kind"], "source": [s],
                "small_stroke": s.get("small_stroke", False)}
        if s["kind"] is None:
            _trust_the_shape(s, labels)
            item["p"] = dict(zip(labels, (float(v) for v in s["probs"])))
            if "forced" in s:  # the writer's own choice of this symbol: no rule about what may stand here applies
                item["forced"] = s["forced"]
        items.append(item)
    lines = [_parse(row) for row in _rows(items)]
    text = "; ".join(line for line in lines if line)
    if "lim" not in text and "∫" not in text and not re.search(r"=.", text):  # plain sums write times as x too
        text = re.sub(r"(?<=[\d)])x(?=[\d(])", "×", text)
    if "y" in text and "=" in text and "(" in text:  # (d^2+5d+6)y = ...: the operator D, as EMNIST has one d for d and D
        text = re.sub(r"(?<![a-zA-Z(^/])d(?=\^|[+\-)]|\d)|(?<=\()d(?=\^|[+\-)])", "D", text)
    return text


def _trust_the_shape(s, labels):
    """'!' is a stem with a dot below it. segment() can see that and the CNN often cannot, so its '!' vote goes to what the shape is."""
    if s.get("exclaim", True):
        return
    probs = s["probs"].copy()
    fact = labels.index("fact")
    for label, share in ({"i": 1.0} if s.get("dotted") else STEM).items():
        probs[labels.index(label)] += share * probs[fact]
    probs[fact] = 0
    s["probs"] = probs


def _stroke_like(it):
    """Could a small stroke be a comma or a prime? Not when the CNN is sure it is a fuller symbol, like the 2 of log₂."""
    return max(it["p"], key=it["p"].get) in COMMA_LOOKALIKES


def _w(it):
    return it["x1"] - it["x0"]


def _h(it):
    return it["y1"] - it["y0"]


def _cx(it):
    return (it["x0"] + it["x1"]) / 2


def _cy(it):
    return (it["y0"] + it["y1"]) / 2


def _merge(parts, **extra):
    return {"x0": min(p["x0"] for p in parts), "y0": min(p["y0"] for p in parts),
            "x1": max(p["x1"] for p in parts), "y1": max(p["y1"] for p in parts),
            "kind": "group", "source": [s for p in parts for s in p["source"]], **extra}


def _best(p, banned=MAIN_LINE):
    """Best reading of a lone symbol among the classes allowed where it stands."""
    allowed = [label for label in p if label not in banned]
    best = max(allowed, key=p.get)
    if best in CHECKED:
        others = [label for label in allowed if label not in CHECKED]
        if others and p[best] < ALONE_MARGIN * max(p[label] for label in others):
            return max(others, key=p.get)
    return best


def _symbol_text(it, banned=MAIN_LINE):
    if it["kind"] == "group":
        return it["text"]
    if it["kind"] in ("=", "÷"):
        return it["kind"]
    if "forced" in it:
        return it["forced"]
    label = _best(it["p"], banned)
    return TOKENS.get(label, label)


def _typical_height(items):
    heights = sorted(_h(it) for it in items)
    return heights[3 * len(heights) // 4] if heights else 0


def _looks_like_bar(it):
    return _w(it) > 2.5 * _h(it)


def _rows(items):
    """Split the drawing at clear gaps across the whole board: each is a line of writing, top to bottom."""
    if len(items) < 2:
        return [items] if items else []
    typical = _typical_height(items)
    ordered = sorted(items, key=lambda it: it["y0"])
    rows, bottom, previous = [[ordered[0]]], ordered[0]["y1"], ordered[0]
    for it in ordered[1:]:
        limit = BAR_GAP if _looks_like_bar(it) or _looks_like_bar(previous) else ROW_GAP
        if it["y0"] - bottom > limit * typical:
            rows.append([])
        rows[-1].append(it)
        bottom = max(bottom, it["y1"]) if len(rows[-1]) > 1 else it["y1"]
        previous = it
    return rows


def _parse(items):
    items = _matrices(sorted(items, key=lambda it: it["x0"]))
    items = _fractions(sorted(items, key=lambda it: it["x0"]))
    items = _roots(sorted(items, key=lambda it: it["x0"]))
    items = _words(sorted(items, key=lambda it: it["x0"]))
    items = _limits(items)
    items = _integrals(sorted(items, key=lambda it: it["x0"]))
    return _line(sorted(items, key=lambda it: it["x0"]))


def _matrices(items):
    """Two tall thin strokes with rows of symbols between them are a matrix, whatever the CNN calls the strokes."""
    if len(items) < 6:
        return items
    typical = sorted(_h(it) for it in items)[len(items) // 2]
    tall = sorted((it for it in items if it["kind"] is None and _h(it) > 1.6 * typical and _w(it) < 0.6 * _h(it)), key=lambda it: it["x0"])
    for left, right in zip(tall, tall[1:]):
        if min(left["y1"], right["y1"]) - max(left["y0"], right["y0"]) < 0.6 * _h(left):
            continue
        inside = [it for it in items if left["x1"] <= _cx(it) <= right["x0"] and left["y0"] <= _cy(it) <= left["y1"]]
        rows = _matrix_rows(inside)
        cells = [_cells(row, inside) for row in rows]
        if len(rows) < 2 or len({len(row) for row in cells}) > 1 or not _rows_fill_brackets(rows, left):
            continue
        for s in left["source"]:
            s["read_as"] = "["
        for s in right["source"]:
            s["read_as"] = "]"
        text = ",".join("[" + ",".join(_line(cell, flat=True) for cell in row) + "]" for row in cells)
        group = _merge([left, right, *inside], text=f"[{text}]")
        return _without(items, [left, right, *inside]) + [group]
    return items


def _rows_fill_brackets(rows, bracket):
    """In a matrix the first row is high between the brackets and the last low. A line with a power above it sits in the middle."""
    centre = lambda row: sum(_cy(it) for it in row) / len(row)
    return (centre(rows[0]) < bracket["y0"] + 0.4 * _h(bracket) and centre(rows[-1]) > bracket["y0"] + 0.6 * _h(bracket))


def _matrix_rows(inside):
    rows = []
    for it in sorted(inside, key=_cy):
        if rows and _cy(it) < max(other["y1"] for other in rows[-1]):
            rows[-1].append(it)
        else:
            rows.append([it])
    return rows


def _cells(row, inside):
    """Split one matrix row at gaps wider than the gap between two digits of a number."""
    widths = sorted(_w(it) for it in inside)
    typical = widths[len(widths) // 2]
    cells, last = [], None
    for it in sorted(row, key=lambda it: it["x0"]):
        if last is None or it["x0"] - last["x1"] > 0.9 * typical:
            cells.append([])
        cells[-1].append(it)
        last = it
    return cells


def _is_bar(it):
    return it["kind"] is None and _best(it["p"]) == "minus" and _w(it) > 2.5 * _h(it)


def _fractions(items):
    """A bar with symbols above and below it is a fraction: (top)/(bottom). Widest bars first."""
    for bar in sorted((it for it in items if _is_bar(it)), key=_w, reverse=True):
        if not any(it is bar for it in items):  # already inside a wider fraction
            continue
        inside = [it for it in items if it is not bar and bar["x0"] <= _cx(it) <= bar["x1"]]
        top = [it for it in inside if _cy(it) < _cy(bar)]
        bottom = [it for it in inside if _cy(it) > _cy(bar)]
        if top and bottom:
            _spot_derivative(top, bottom)
            group = _merge([bar, *top, *bottom], text=f"({_parse(top)})/({_parse(bottom)})")
            for s in bar["source"]:
                s["read_as"] = "fraction bar"
            items = _without(items, [bar, *top, *bottom]) + [group]
    return items


def _spot_derivative(top, bottom):
    """dy/dx: a bottom of two symbols ending in a variable, with something d-like first on both lines."""
    below = sorted(bottom, key=lambda it: it["x0"])
    above = sorted(top, key=lambda it: it["x0"])
    if len(below) not in (2, 3) or below[0]["kind"] is not None or above[0]["kind"] is not None:
        return
    d_like = lambda it: "d" in it["p"] and (it["p"]["d"] > 0.05 or it["p"].get("partial", 0) > 0.05)
    if d_like(below[0]) and d_like(above[0]) and _best(below[1]["p"], WORD_ONLY - set("yt")) in ("x", "y", "t"):
        below[0]["forced"] = above[0]["forced"] = "d"


def _roots(items):
    """A square root with a bar over symbols: √(what is under the bar)."""
    for root in (it for it in items if it["kind"] is None and _best(it["p"]) == "sqrt"):
        under = [it for it in items if it is not root and root["x0"] + 0.2 * _w(root) <= _cx(it) and it["x0"] <= root["x1"] - ROOT_OVERLAP * _w(it)
                 and root["y0"] <= _cy(it) <= root["y1"]]
        if under:
            for s in root["source"]:
                s["read_as"] = "√"
            group = _merge([root, *under], text=f"√({_parse(under)})")
            return _roots(_without(items, [root, *under]) + [group])
    return items


def _word_score(run, word):
    """How much better (log-odds) the run reads as the word than as its best non-letter symbols."""
    if any(it["kind"] is not None or letter not in it["p"] for it, letter in zip(run, word)):
        return -math.inf
    if any(it["p"][letter] < MIN_LETTER for it, letter in zip(run, word)):  # with the whole alphabet, other letters can win
        return -math.inf
    as_word = sum(math.log(max(it["p"][letter], 1e-9)) for it, letter in zip(run, word))
    as_symbols = sum(math.log(max(max(v for k, v in it["p"].items() if k not in WORD_ONLY), 1e-9)) for it in run)  # → and ∞ count here
    return as_word - as_symbols


def _side_by_side(run):
    height = sorted(_h(it) for it in run)[len(run) // 2]
    return all(b["x0"] - a["x1"] < 0.8 * height and min(a["y1"], b["y1"]) > max(a["y0"], b["y0"])
               for a, b in zip(run, run[1:]))


def _without(items, gone):
    gone = {id(it) for it in gone}
    return [it for it in items if id(it) not in gone]


def _same_row(items, i, count):
    """items[i] and the next count-1 items to its right on its row (skipping anything above or below it)."""
    first = items[i]
    run = [first] + [it for it in items[i + 1:] if min(it["y1"], first["y1"]) > max(it["y0"], first["y0"])]
    return run[:count] if len(run) >= count else None


def _words(items):
    """Replace runs of letters by the function word they spell best, if that beats reading them as symbols."""
    i = 0
    while i < len(items):
        best = (-math.inf, None, None)
        for word in WORDS:
            run = _same_row(items, i, len(word))
            if run and _side_by_side(run):
                best = max(best, (_word_score(run, word), word, run), key=lambda b: b[0])
        score, word, run = best
        if score > WORD_BIAS:
            for it, letter in zip(run, word):
                for s in it["source"]:
                    s["read_as"] = letter
                    s["in_word"] = True
            items = _without(items, run)
            items.insert(i, _merge(run, text=word, word=word))
        i += 1
    return items


def _limits(items):
    """Whatever sits under 'lim' (x→a) becomes its subscript."""
    for lim in [it for it in items if it.get("word") == "lim"]:
        under = [it for it in items if it is not lim and it["y0"] > lim["y1"] - 0.2 * _h(lim)
                 and lim["x0"] - 0.3 * _w(lim) <= _cx(it) <= lim["x1"] + 0.3 * _w(lim)]
        if under:
            target = _line(sorted(under, key=lambda it: it["x0"]), banned=WORD_ONLY, flat=True)
            if "→" not in target:  # under lim there is always an arrow: a '-' there is one with a faint head
                target = target.replace("-", "→", 1)
            lim["text"] = f"lim_({target}) "
            items = _without(items, under)
    return items


def _is_integral(it):
    return it["kind"] is None and _best(it["p"]) == "integral"


def _integrals(items):
    """∫ with small limits at its top right and bottom right, and the d of the 'dx' that ends the line."""
    for sign in [it for it in items if _is_integral(it)]:
        height = _h(sign)
        near = sorted((it for it in items if it is not sign and _cx(it) > sign["x0"] and _h(it) < 0.55 * height), key=lambda it: it["x0"])

        def run_from_sign(candidates):
            """The limit starts next to the sign and carries on while its symbols stay close together."""
            run, end = [], sign["x1"]
            for it in candidates:
                if it["x0"] - end > 0.2 * height:
                    break
                run.append(it)
                end = it["x1"]
            return run

        upper = run_from_sign([it for it in near if _cy(it) < sign["y0"] + 0.35 * height])
        lower = run_from_sign([it for it in near if _cy(it) > sign["y1"] - 0.35 * height])
        text = "∫"
        if lower or upper:
            low = _line(sorted(lower, key=lambda it: it["x0"]), flat=True) or "0"
            high = _line(sorted(upper, key=lambda it: it["x0"]), flat=True) or "1"
            text = f"∫_({low})^({high})"
        for s in sign["source"]:
            s["read_as"] = "∫"
        items = _without(items, [sign, *upper, *lower])
        items.append(_merge([sign, *upper, *lower], text=text + " ", tall=True))
        body = sorted((it for it in items if not it.get("tall")), key=lambda it: it["x0"])
        if len(body) >= 2 and body[-1]["kind"] is None and body[-2]["kind"] is None:
            body[-2]["forced"] = "d"
            if _best(body[-1]["p"], WORD_ONLY - set("yt")) in ("x", "y", "t"):
                body[-1]["forced"] = _best(body[-1]["p"], WORD_ONLY - set("yt"))
    return items


def _balance_brackets(pieces, banned):
    """A bracket without a partner was most likely a '1' (or similar): read it again without brackets."""
    open_ones, lonely = [], []
    for k, (_, text, _, _) in enumerate(pieces):
        if text == "(":
            open_ones.append(k)
        elif text == ")":
            if open_ones:
                open_ones.pop()
            else:
                lonely.append(k)
    for k in lonely + open_ones:
        if "p" in pieces[k][0]:
            pieces[k][1] = _symbol_text(pieces[k][0], banned | BRACKETS)


def _script(parts):
    text = _line(parts)
    return text if len(text) == 1 else f"({text})"


def _band(items):
    """Top and bottom of the line, from its full-height symbols (small letters, scripts, dots and ∫ signs ignored)."""
    usable = [it for it in items if not it.get("tall")] or items
    tallest = max(_h(it) for it in usable)
    tall = [it for it in usable if _h(it) >= 0.6 * tallest]
    tops, bottoms = sorted(it["y0"] for it in tall), sorted(it["y1"] for it in tall)
    return tops[(len(tops) - 1) // 2], bottoms[len(bottoms) // 2]


def _takes_subscript(piece):
    """log takes a base; a lone letter such as x or a takes a subscript."""
    it, text = piece[0], piece[1]
    return it.get("word") == "log" or (it["kind"] is None and len(text) == 1 and text in ALL_LETTERS | {"x"} and text != "e")


def _is_raised(piece, it, height):
    """A power measured against what it follows, as people's lines drift: well above the base, and ending above its middle."""
    base = piece[0]
    if piece[1] in (",", ".", "·") or _h(base) < 0.4 * height or _h(it) > 1.1 * _h(base) or it["kind"] == "=":
        return False
    if it["kind"] is None and _best(it["p"]) in ("plus", "minus"):  # + and − sit at mid-height, which is high beside a small x
        return False
    return _cy(it) < base["y0"] + 0.3 * _h(base) and it["y1"] < base["y0"] + 0.7 * _h(base)


def _is_low_subscript(piece, it, least_height):
    """A subscript measured against its letter: low on it and reaching below it. The x of dx sits on the same line as its tall d."""
    base = piece[0]
    if piece[1] == "e" or not _takes_subscript(piece) or base.get("word"):
        return False
    return (_cy(it) > base["y0"] + 0.65 * _h(base) and it["y1"] > base["y1"] + 0.1 * _h(base)
            and least_height * _h(base) <= _h(it) < 0.9 * _h(base))


def _line(items, banned=MAIN_LINE, flat=False):
    """Left to right along the line; small symbols high on the line are a power, low after 'log' or a letter its base or subscript.

    flat=True is for the x→a under 'lim', which never has powers: a small arrow there can sit high next to the x.
    """
    if not items:
        return ""
    top, bottom = _band(items)
    height = max(bottom - top, 1)
    pieces = []  # [base item, its text, power items, base items]
    for it in items:
        if it.get("small_stroke") and pieces and not flat and _stroke_like(it):
            if _is_low_subscript(pieces[-1], it, 0.45):  # a 1 written low after a letter is long; a comma is short
                pieces[-1][3].append(it)
                continue
            if _cy(it) > top + 0.6 * height and _h(it) < 0.6 * height:  # low: a comma
                pieces.append([it, ",", [], []])
                continue
            if pieces[-1][1].rstrip("'") in ("y", "f") and _cy(it) < top + 0.5 * height:  # high after y: a prime
                pieces[-1][1] += "'"
                for s in it["source"]:
                    s["read_as"] = "'"
                continue
        if pieces and it["kind"] != "dot" and not flat:
            if (_cy(it) < top + 0.3 * height and it["y1"] < top + 0.65 * height) or _is_raised(pieces[-1], it, height):
                pieces[-1][2].append(it)
                continue
            if _is_low_subscript(pieces[-1], it, 0) or (pieces[-1][0].get("word") == "log" and _cy(it) > bottom - 0.3 * height and _h(it) < 0.7 * height):
                pieces[-1][3].append(it)
                continue
        if it["kind"] == "dot":  # low on the line a decimal point, at mid-height times
            text = "." if _cy(it) > top + 0.65 * height else "·"
        else:
            text = _symbol_text(it, banned)
        pieces.append([it, text, [], []])
    _balance_brackets(pieces, banned)
    for it, text, _, _ in pieces:
        for s in it["source"]:
            s.setdefault("read_as", text)
    out = []
    for _, text, powers, bases in pieces:
        if bases:
            text += "_" + _script(bases)
        if powers:
            text += "^" + _script(powers)
        out.append(text)
    return "".join(out).strip()
