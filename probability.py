"""Statistics and Probability: descriptive statistics, regression, probability, distributions, sampling and tests."""
import itertools
import math
import re

import numpy as np
import sympy as sp
from scipy import stats as scipy_stats

from results import Result, decimal, parts_of, read_numbers, show
from solver import read_equation, read_expression
from working import text_of


def _lists(text):
    """'x=1,2,3; y=4,5,6' as {'x': '1,2,3', 'y': '4,5,6'}."""
    found = {}
    for part in parts_of(text):
        match = re.fullmatch(r"\s*([A-Za-z_0-9]+)\s*=\s*(.+)", part)
        if match:
            found[match[1].lower()] = match[2].strip()
    return found


def _options(text):
    """'mean=70 sd=5 n=4' as {'mean': '70', ...}; values stop at a space, comma or semicolon."""
    return {key.lower(): value for key, value in re.findall(r"\b([A-Za-z_0-9]+)\s*=\s*([^\s;,]+)", text)}


def _num(value):
    return read_expression(str(value))


def _f(value):
    return float(sp.N(value))


# ---------- descriptive statistics ----------

def _spread(values, weights):
    n = sum(weights)
    mean = sum(v * w for v, w in zip(values, weights)) / n
    squares = sum(w * (v - mean) ** 2 for v, w in zip(values, weights))
    return n, mean, squares


def _modes(values, weights):
    highest = max(weights)
    return [v for v, w in zip(values, weights) if w == highest], highest


def _median_line(n, median):
    if n % 2:
        return f"Median: in order, the middle of the {n} values is place {(int(n) + 1) // 2}, which holds {show(median)}"
    return f"Median: in order, the middle of the {n} values is between places {int(n) // 2} and {int(n) // 2 + 1}, whose average is {show(median)}"


def describe_data(text):
    """stats 4,1,5,... or stats values=...; freq=... or stats classes=10-12,12-14; freq=7,12."""
    given = _lists(text)
    if "classes" in given:
        return _grouped(given)
    if "values" in given or "x" in given:
        values = read_numbers(given.get("values", given.get("x")))
        weights = read_numbers(given.get("freq", given.get("f", "")))
    else:
        values, weights = read_numbers(text), None
    if weights is None:
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        values = sorted(counts)
        weights = [counts[v] for v in values]
    if len(values) != len(weights):
        raise ValueError("There must be one frequency for each value")
    pairs = sorted(zip(values, weights), key=lambda p: _f(p[0]))
    values, weights = [p[0] for p in pairs], [p[1] for p in pairs]
    n, mean, squares = _spread(values, weights)
    stretched = [v for v, w in zip(values, weights) for _ in range(int(w))]
    median = (stretched[(int(n) - 1) // 2] + stretched[int(n) // 2]) / sp.Integer(2)
    modes, highest = _modes(values, weights)
    mode_text = "no single mode (every value occurs equally often)" if len(modes) == len(values) else ", ".join(text_of(m) for m in modes)
    answer = [f"Mean = {show(mean)}", f"Median = {show(median)}", f"Mode = {mode_text}",
              f"Variance (dividing by n) = {decimal(squares / n)}, s.d. = {decimal(sp.sqrt(squares / n))}"]
    if n > 1:
        answer.append(f"Variance (dividing by n-1) = {decimal(squares / (n - 1))}, s.d. = {decimal(sp.sqrt(squares / (n - 1)))}")
    cv = sp.sqrt(squares / n) / mean * 100 if mean != 0 else None
    steps = ["Values with their frequencies: " + ", ".join(f"{text_of(v)} ({text_of(w)})" for v, w in zip(values, weights)),
             f"Number of values: n = Σf = {text_of(n)}",
             f"Mean = Σfx ÷ n = {text_of(mean * n)} ÷ {text_of(n)} = {show(mean)}",
             _median_line(n, median),
             f"Mode: the highest frequency is {text_of(highest)}, which gives {mode_text}",
             f"Squared spread: Σf(x - mean)² = {decimal(squares)}",
             f"Variance dividing by n = {decimal(squares)} ÷ {text_of(n)} = {decimal(squares / n)}, and s.d. is its square root, {decimal(sp.sqrt(squares / n))}"]
    if n > 1:
        steps.append(f"Variance dividing by n-1 = {decimal(squares)} ÷ {text_of(n - 1)} = {decimal(squares / (n - 1))}, and s.d. is its square root, {decimal(sp.sqrt(squares / (n - 1)))}")
    if cv is not None:
        steps.append(f"Coefficient of variation = {decimal(cv, 2)}%: "
                     + ("the values are fairly steady" if _f(cv) < 15 else "the values vary quite a lot"))
    return Result("Mean, median, mode and spread", answer, steps)


def _grouped(given):
    classes = [re.split(r"\s*(?:-|–|to)\s*", item.strip()) for item in given["classes"].split(",")]
    edges = [(_num(a), _num(b)) for a, b in classes]
    weights = read_numbers(given.get("freq", given.get("f", "")))
    if len(edges) != len(weights):
        raise ValueError("There must be one frequency for each class")
    middles = [(a + b) / 2 for a, b in edges]
    n, mean, squares = _spread(middles, weights)
    running, median, median_line = sp.Integer(0), None, ""
    for (low, high), w in zip(edges, weights):
        if running + w >= n / 2:
            median = low + (n / 2 - running) / w * (high - low)
            median_line = (f"Median: n/2 = {text_of(n / 2)} falls in the class {text_of(low)} to {text_of(high)}, so median = L + ((n/2 - F) ÷ f) × h "
                           f"= {text_of(low)} + (({text_of(n / 2)} - {text_of(running)}) ÷ {text_of(w)}) × {text_of(high - low)} = {decimal(median)}")
            break
        running += w
    top = max(range(len(weights)), key=lambda k: weights[k])
    before = weights[top - 1] if top > 0 else 0
    after = weights[top + 1] if top + 1 < len(weights) else 0
    low, high = edges[top]
    mode = low + (weights[top] - before) / (2 * weights[top] - before - after) * (high - low)
    answer = [f"Mean = {show(mean)}", f"Median = {decimal(median)}", f"Mode = {decimal(mode)}",
              f"Variance = {decimal(squares / n)}, s.d. = {decimal(sp.sqrt(squares / n))}"]
    steps = ["Mid-point of each class: " + ", ".join(text_of(m) for m in middles), f"n = Σf = {text_of(n)}, Σfm = {text_of(mean * n)}",
             f"Mean = Σfm ÷ n = {text_of(mean * n)} ÷ {text_of(n)} = {show(mean)}", median_line,
             f"Mode: the largest frequency is {text_of(weights[top])} in the class {text_of(low)} to {text_of(high)}, so mode = L + ((f1 - f0) ÷ (2f1 - f0 - f2)) × h "
             f"= {text_of(low)} + (({text_of(weights[top])} - {text_of(before)}) ÷ ({text_of(2 * weights[top])} - {text_of(before)} - {text_of(after)})) × {text_of(high - low)} = {decimal(mode)}",
             f"Variance = Σf(m - mean)² ÷ n = {decimal(squares)} ÷ {text_of(n)} = {decimal(squares / n)}, and s.d. is its square root, {decimal(sp.sqrt(squares / n))}"]
    return Result("Grouped data", answer, steps)


def compare(text):
    """Which set has the higher level (mean), and which is more consistent (coefficient of variation)?"""
    given = _lists(text)
    names = list(given)
    rows, answer, steps = [], [], []
    for name in names:
        values = read_numbers(given[name])
        n, mean, squares = _spread(values, [1] * len(values))
        sd = sp.sqrt(squares / n)
        rows.append((name, mean, sd, sd / mean * 100))
        answer.append(f"{name}: mean = {decimal(mean, 2)}, s.d. = {decimal(sd, 2)}, CV = {decimal(sd / mean * 100, 2)}%")
        steps.append(f"{name}: n = {text_of(n)}, mean = Σx ÷ n = {text_of(mean * n)} ÷ {text_of(n)} = {decimal(mean, 2)}, s.d. = √(Σ(x - mean)² ÷ n) = √({decimal(squares, 2)} ÷ {text_of(n)}) = {decimal(sd, 2)}, "
                     f"CV = s.d. ÷ mean × 100 = {decimal(sd / mean * 100, 2)}%")
    higher = max(rows, key=lambda r: _f(r[1]))
    steadier = min(rows, key=lambda r: _f(r[3]))
    answer += [f"Higher level: {higher[0]} (larger mean)", f"More consistent: {steadier[0]} (smaller CV)"]
    steps += ["The larger mean shows the higher level, and the smaller CV shows the steadier set"]
    return Result("Comparing two sets of data", answer, steps)


def empirical(text):
    """Mode = 3 Median - 2 Mean with two of the three given."""
    given = _options(text)
    mean, median, mode = (_num(given[k]) if k in given else None for k in ("mean", "median", "mode"))
    rule = "The empirical relation: Mode = 3 Median - 2 Mean"
    if median is None:
        return Result("Empirical relation", [f"Median = {show(sp.simplify((mode + 2 * mean) / 3))}"],
                      [rule, f"So Median = (Mode + 2 Mean) ÷ 3 = ({text_of(mode)} + 2 × {text_of(mean)}) ÷ 3 = {show(sp.simplify((mode + 2 * mean) / 3))}"])
    if mode is None:
        return Result("Empirical relation", [f"Mode = {show(3 * median - 2 * mean)}"], [rule, f"Mode = 3 × {text_of(median)} - 2 × {text_of(mean)} = {show(3 * median - 2 * mean)}"])
    return Result("Empirical relation", [f"Mean = {show(sp.simplify((3 * median - mode) / 2))}"],
                  [rule, f"So Mean = (3 Median - Mode) ÷ 2 = (3 × {text_of(median)} - {text_of(mode)}) ÷ 2 = {show(sp.simplify((3 * median - mode) / 2))}"])


# ---------- correlation and regression ----------

def _xy(text):
    given = _lists(text)
    if "x" not in given or "y" not in given:
        raise ValueError("Give the data as x=...; y=...")
    xs, ys = np.array([_f(v) for v in read_numbers(given["x"])]), np.array([_f(v) for v in read_numbers(given["y"])])
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same number of values")
    return xs, ys, given


def _line(slope, intercept, y="y", x="x"):
    return f"{y} = {slope:.4f}{x} {'+' if intercept >= 0 else '-'} {abs(intercept):.4f}"


def correlation(text):
    xs, ys, _ = _xy(text)
    r = float(np.corrcoef(xs, ys)[0, 1])
    b_yx, b_xy = float(np.polyfit(xs, ys, 1)[0]), float(np.polyfit(ys, xs, 1)[0])
    a_yx, a_xy = ys.mean() - b_yx * xs.mean(), xs.mean() - b_xy * ys.mean()
    strength = "very strong" if abs(r) > .9 else "strong" if abs(r) > .7 else "moderate" if abs(r) > .4 else "weak"
    answer = [f"r = {r:.4f}: {strength} {'positive' if r > 0 else 'negative'} relationship",
              f"Regression of y on x: {_line(b_yx, a_yx)}", f"Regression of x on y: {_line(b_xy, a_xy, 'x', 'y')}"]
    sxx, syy, sxy = float(((xs - xs.mean()) ** 2).sum()), float(((ys - ys.mean()) ** 2).sum()), float(((xs - xs.mean()) * (ys - ys.mean())).sum())
    steps = [f"n = {len(xs)}, mean of x = {xs.mean():.4f}, mean of y = {ys.mean():.4f}", f"s.d. of x = {xs.std():.4f}, s.d. of y = {ys.std():.4f}",
             f"Σ(x - mean)² = {sxx:.4f}, Σ(y - mean)² = {syy:.4f}, Σ(x - mean)(y - mean) = {sxy:.4f}",
             f"r = Σ(x - mean)(y - mean) ÷ √(Σ(x - mean)² × Σ(y - mean)²) = {sxy:.4f} ÷ √({sxx:.4f} × {syy:.4f}) = {r:.4f}",
             f"Slope of y on x = {sxy:.4f} ÷ {sxx:.4f} = {b_yx:.4f}, and it crosses at mean y - slope × mean x = {a_yx:.4f}",
             f"Slope of x on y = {sxy:.4f} ÷ {syy:.4f} = {b_xy:.4f}, and it crosses at mean x - slope × mean y = {a_xy:.4f}"]
    return Result("Correlation and regression", answer, steps)


def linear_fit(text, kind="linear"):
    xs, ys, given = _xy(text)
    predict = _f(_num(given["predict"])) if "predict" in given else None
    steps = [f"{len(xs)} points. Least squares picks the curve with the smallest total squared distance from them",
             f"Σx = {xs.sum():g}, Σy = {ys.sum():g}, Σx² = {(xs ** 2).sum():g}, Σxy = {(xs * ys).sum():g}"]
    if kind == "quadratic":
        c, b, a = np.polyfit(xs, ys, 2)
        answer = [f"y = {a:.4f} {'+' if b >= 0 else '-'} {abs(b):.4f}x {'+' if c >= 0 else '-'} {abs(c):.4f}x²"]
        estimate = (lambda v: a + b * v + c * v * v)
        steps += [f"Also Σx³ = {(xs ** 3).sum():g}, Σx⁴ = {(xs ** 4).sum():g}, Σx²y = {(xs ** 2 * ys).sum():g}",
                  "The normal equations: Σy = na + bΣx + cΣx²; Σxy = aΣx + bΣx² + cΣx³; Σx²y = aΣx² + bΣx³ + cΣx⁴",
                  f"Solving them: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}"]
    elif kind == "exponential":
        if (ys <= 0).any():
            raise ValueError("An exponential fit needs every y above 0")
        b, log_a = np.polyfit(xs, np.log(ys), 1)
        a = math.exp(log_a)
        answer = [f"y = {a:.4f}·e^({b:.4f}x)"]
        estimate = (lambda v: a * math.exp(b * v))
        steps += ["Take ln of y = a·e^(bx) to get a straight line: ln y = ln a + bx", f"Fitting a line to (x, ln y) gives b = {b:.4f} and ln a = {log_a:.4f}", f"So a = e^({log_a:.4f}) = {a:.4f}"]
    else:
        slope, intercept = np.polyfit(xs, ys, 1)
        answer = [_line(slope, intercept)]
        estimate = (lambda v: intercept + slope * v)
        steps += ["The normal equations: Σy = na + bΣx; Σxy = aΣx + bΣx²", f"Solving them: slope b = {slope:.4f}, intercept a = {intercept:.4f}"]
    if predict is not None:
        answer.append(f"At x = {predict:g}: y ≈ {estimate(predict):.4f}")
        steps.append(f"Put x = {predict:g} into the curve: y ≈ {estimate(predict):.4f}")
    return Result(f"{kind.capitalize()} regression", answer, steps)


def regression_lines(text):
    """Two regression lines give the means (their meeting point) and r; with s.d. of x also the s.d. of y."""
    x, y = sp.symbols("x y")
    pieces = parts_of(text)
    sigma = None
    equations = []
    for piece in pieces:
        match = re.fullmatch(r"(?:sx|σx|sigma_?x)\s*=\s*(.+)", piece)
        if match:
            sigma = _num(match[1])
        else:
            left, right = read_equation(piece)
            equations.append(left - right)
    if len(equations) != 2:
        raise ValueError("Give the two regression lines")
    solved = sp.solve(equations, [x, y], dict=True)[0]
    def slopes(first, second):
        b_yx = sp.solve(first, y)[0].coeff(x)
        b_xy = sp.solve(second, x)[0].coeff(y)
        return b_yx, b_xy
    b_yx, b_xy = slopes(equations[0], equations[1])
    if b_yx * b_xy > 1:  # the first line must be x on y instead
        b_yx, b_xy = slopes(equations[1], equations[0])
    r = sp.sqrt(b_yx * b_xy) * (1 if b_yx > 0 else -1)
    answer = [f"Mean of x = {show(solved[x])}, mean of y = {show(solved[y])}", f"r = {show(r)}"]
    steps = [f"Both lines pass through the point (mean of x, mean of y), so solve them together: x = {show(solved[x])}, y = {show(solved[y])}",
             f"Write each line with its slope: slope of y on x = {show(b_yx)}, slope of x on y = {show(b_xy)}. Their product must be at most 1, which tells which line is which",
             f"r² = the product of the two slopes = {show(b_yx * b_xy)}, and r takes the sign of the slopes: r = {show(r)}"]
    if sigma is not None:
        answer.append(f"s.d. of y = {show(b_yx * sigma / r)}")
        steps.append(f"Slope of y on x = r·σy/σx, so σy = slope × σx ÷ r = {show(b_yx)} × {text_of(sigma)} ÷ {show(r)} = {show(b_yx * sigma / r)}")
    return Result("Regression lines", answer, steps)


# ---------- probability ----------

def set_probabilities(text):
    """From P(A), P(B) and P(AnB): every related probability."""
    values = {}
    for part in parts_of(text.replace(",", ";")):
        match = re.fullmatch(r"P\(\s*([^)]+?)\s*\)\s*=\s*(.+)", part)
        if match:
            values[match[1].replace(" ", "").replace("∩", "n").upper()] = _num(match[2])
    a, b = values.get("A"), values.get("B")
    both = values.get("ANB", values.get("A&B"))
    if a is None or b is None or both is None:
        raise ValueError("Give P(A), P(B) and P(AnB)")
    union = a + b - both
    lines = [("P(A∪B)", union), ("P(A∩B̄)", a - both), ("P(Ā∩B)", b - both), ("P(A|B)", both / b), ("P(B|A)", both / a),
             ("P(Ā|B̄)", (1 - union) / (1 - b)), ("P(B̄|Ā)", (1 - union) / (1 - a)), ("P(A|B̄)", (a - both) / (1 - b)), ("P(Ā|B)", (b - both) / b)]
    a_, b_, ab_ = text_of(a), text_of(b), text_of(both)
    steps = [f"Given: P(A) = {a_}, P(B) = {b_}, P(A∩B) = {ab_}",
             f"P(A∪B) = P(A) + P(B) - P(A∩B) = {a_} + {b_} - {ab_} = {show(union)}",
             f"P(A∩B̄) = P(A) - P(A∩B) = {a_} - {ab_} = {show(a - both)}", f"P(Ā∩B) = P(B) - P(A∩B) = {b_} - {ab_} = {show(b - both)}",
             f"P(A|B) = P(A∩B) ÷ P(B) = {ab_} ÷ {b_} = {show(both / b)}", f"P(B|A) = P(A∩B) ÷ P(A) = {ab_} ÷ {a_} = {show(both / a)}",
             f"P(Ā∩B̄) = 1 - P(A∪B) = {show(1 - union)}", f"P(Ā|B̄) = P(Ā∩B̄) ÷ P(B̄) = {text_of(1 - union)} ÷ {text_of(1 - b)} = {show((1 - union) / (1 - b))}",
             f"P(B̄|Ā) = P(Ā∩B̄) ÷ P(Ā) = {text_of(1 - union)} ÷ {text_of(1 - a)} = {show((1 - union) / (1 - a))}",
             f"P(A|B̄) = P(A∩B̄) ÷ P(B̄) = {text_of(a - both)} ÷ {text_of(1 - b)} = {show((a - both) / (1 - b))}",
             f"P(Ā|B) = P(Ā∩B) ÷ P(B) = {text_of(b - both)} ÷ {b_} = {show((b - both) / b)}"]
    return Result("Probability of events", [f"{name} = {show(value)}" for name, value in lines], steps)


def bayes(text):
    """bayes priors=20%,15%,25%,40%; likelihoods=5%,4%,3%,2%; want=A,D"""
    given = _lists(text)
    priors = read_numbers(given["priors"])
    likelihoods = read_numbers(given["likelihoods"])
    if len(priors) != len(likelihoods):
        raise ValueError("There must be one likelihood for each prior")
    labels = given.get("labels", ",".join("ABCDEFGH"[:len(priors)])).replace(" ", "").split(",")
    total = sum(p * l for p, l in zip(priors, likelihoods))
    posterior = {label: p * l / total for label, p, l in zip(labels, priors, likelihoods)}
    products = [p * l for p, l in zip(priors, likelihoods)]
    steps = ["Write down the chance of each source and the chance of the event from it:"]
    steps += [f"P({label}) = {text_of(p)}, P(event | {label}) = {text_of(l)}" for label, p, l in zip(labels, priors, likelihoods)]
    steps.append("Multiply each pair: " + ", ".join(f"{label}: {text_of(v)}" for label, v in zip(labels, products)))
    steps.append(f"Total probability, adding them: P(event) = {' + '.join(text_of(v) for v in products)} = {show(total)}")
    steps.append("Bayes' theorem: P(source | event) = its product ÷ P(event)")
    steps += [f"P({label} | event) = {text_of(v)} ÷ {text_of(total)} = {show(value)}" for (label, value), v in zip(posterior.items(), products)]
    if "want" in given:
        wanted = given["want"].replace(" ", "").split(",")
        steps.append(f"Add the ones asked for: {' + '.join(text_of(posterior[w]) for w in wanted)} = {show(sum(posterior[w] for w in wanted))}")
        return Result("Bayes' theorem", [f"P({' or '.join(wanted)} | event) = {show(sum(posterior[w] for w in wanted))}"], steps)
    return Result("Bayes' theorem", [f"P(event) = {show(total)}"] + [f"P({label} | event) = {show(value)}" for label, value in posterior.items()], steps)


def transfer(text):
    """Balls moved from one bag to another, then one drawn from the second bag."""
    given = _options(text)
    w1, r1, w2, r2, moved = (int(_num(given[k])) for k in ("white1", "red1", "white2", "red2", "moved"))
    n1, n2 = w1 + r1, w2 + r2 + moved
    total = sp.Integer(0)
    steps = []
    for k in range(0, moved + 1):
        chance = sp.Rational(sp.binomial(w1, k) * sp.binomial(r1, moved - k), sp.binomial(n1, moved))
        total += chance * sp.Rational(w2 + k, n2)
        steps.append(f"{k} white moved: chance C({w1},{k})·C({r1},{moved - k}) ÷ C({n1},{moved}) = {text_of(chance)}, then white from the second bag with chance {w2 + k}/{n2}, "
                     f"together {text_of(chance * sp.Rational(w2 + k, n2))}")
    steps.append(f"Add the cases: {show(total)}")
    return Result("Total probability", [f"P(white) = {show(total)}"], steps)


CONDITION = re.compile(r"P\(\s*([^)]*?)\s*\)")
TERMS_SHOWN = 8  # terms of a sum of chances written out before it says '...'


def _condition(text):
    """'x<=1', '-1<x<=2', 'X>=6', 'x=2' as (low, low_closed, high, high_closed)."""
    text = text.replace(" ", "").replace("≤", "<=").replace("≥", ">=").lower().replace("xbar", "x").replace("x̄", "x")
    match = re.fullmatch(r"(.+?)(<=?)x(<=?)(.+)", text)
    if match:
        return _f(_num(match[1])), match[2] == "<=", _f(_num(match[4])), match[3] == "<="
    match = re.fullmatch(r"x(<=|>=|<|>|=)(.+)", text)
    if not match:
        raise ValueError(f"Cannot read the condition '{text}'")
    value = _f(_num(match[2]))
    return {"<=": (-math.inf, False, value, True), "<": (-math.inf, False, value, False), ">=": (value, True, math.inf, False),
            ">": (value, False, math.inf, False), "=": (value, True, value, True)}[match[1]]


def _inside(value, condition):
    low, low_closed, high, high_closed = condition
    return ((value > low) or (low_closed and value == low)) and ((value < high) or (high_closed and value == high))


def discrete_table(text):
    """distribution x=-3,-2,...; p=k,2k,3k,...; P(x<=1); P(x>1)"""
    given = _lists(text)
    xs = read_numbers(given["x"])
    k = sp.Symbol("k")
    ps = [read_expression(item) for item in re.split(r"[,\s]+", given["p"].strip()) if item]
    steps = []
    if any(p.has(k) for p in ps):
        roots = [r for r in sp.solve(sum(ps) - 1, k) if r >= 0]
        if not roots:
            raise ValueError("No value of k makes these probabilities add up to 1")
        steps.append(f"The probabilities must add up to 1: {show(sum(ps))} = 1, which gives k = {text_of(roots[0])}")
        answer = [f"k = {show(roots[0])}"]
        ps = [p.subs(k, roots[0]) for p in ps]
        steps.append("So the probabilities are " + ", ".join(text_of(p) for p in ps))
    else:
        answer = []
    mean = sum(x * p for x, p in zip(xs, ps))
    variance = sum(x ** 2 * p for x, p in zip(xs, ps)) - mean ** 2
    answer += [f"Mean = {show(mean)}", f"Variance = {show(variance)}", f"s.d. = {show(sp.sqrt(variance))}"]
    steps.append("Mean = Σxp = " + " + ".join(f"({text_of(x)})({text_of(p)})" for x, p in zip(xs, ps)) + f" = {show(mean)}")
    steps.append(f"Σx²p = {show(sum(x ** 2 * p for x, p in zip(xs, ps)))}, so variance = Σx²p - mean² = {show(variance)}, and s.d. is its square root, {show(sp.sqrt(variance))}")
    for query in CONDITION.findall(text):
        chosen = [(x, p) for x, p in zip(xs, ps) if _inside(_f(x), _condition(query))]
        chance = sum(p for _, p in chosen)
        answer.append(f"P({query}) = {show(chance)}")
        steps.append(f"P({query}) adds the probabilities of x = {', '.join(text_of(x) for x, _ in chosen)}: {' + '.join(text_of(p) for _, p in chosen) or '0'} = {show(chance)}")
    return Result("Probability distribution", answer, steps)


def _queries(text):
    return CONDITION.findall(text)


def _count_line(chance, total):
    return f", expected count {_f(chance) * total:.1f}" if total else ""


def binomial(text):
    """binomial n=10 p=1/2 P(X>=6); N=number of repeats for counts"""
    given = _options(text)
    n, p = int(_num(given["n"])), _num(given["p"])
    total = _f(_num(given["repeats"])) if "repeats" in given else None
    answer, steps = [], [f"n = {n} trials, each with chance p = {text_of(p)}, so q = 1 - p = {text_of(1 - p)}", f"P(X=r) = C({n},r) ({text_of(p)})^r ({text_of(1 - p)})^({n}-r)",
                         f"Mean = np = {show(n * p)}, variance = npq = {show(n * p * (1 - p))}"]
    for query in _queries(text) or ["x>=0"]:
        condition = _condition(query)
        chosen = [r for r in range(n + 1) if _inside(r, condition)]
        chance = sum(sp.binomial(n, r) * p ** r * (1 - p) ** (n - r) for r in chosen)
        answer.append(f"P({query}) = {show(chance)}{_count_line(chance, total)}")
        terms = [f"P(X={r}) = {decimal(sp.binomial(n, r) * p ** r * (1 - p) ** (n - r), 6)}" for r in chosen[:TERMS_SHOWN]]
        steps.append(f"P({query}) adds r = {', '.join(map(str, chosen[:TERMS_SHOWN]))}{', ...' if len(chosen) > TERMS_SHOWN else ''}: {'; '.join(terms)}. The total is {decimal(chance, 6)}")
        if total:
            steps.append(f"Expected count = chance × {total:g} = {_f(chance) * total:.1f}")
    return Result("Binomial distribution", answer, steps)


def poisson(text):
    """poisson m=2 P(X=0); poisson n=2000 p=0.001 P(X>2); repeats=10000 for counts"""
    given = _options(text)
    m = _num(given["m"]) if "m" in given else _num(given["n"]) * _num(given["p"])
    total = _f(_num(given["repeats"])) if "repeats" in given else None
    answer, steps = [], [f"The mean: m = {text_of(m)}" + ("" if "m" in given else f", from np = {given['n']} × {given['p']}"), f"P(X=r) = e^(-m) m^r / r!"]
    term = lambda r: f"P(X={r}) = {decimal(sp.exp(-m) * m ** r / sp.factorial(r), 5)}"
    for query in _queries(text):
        condition = _condition(query)
        low, _, high, _ = condition
        if high == math.inf:  # the complement of a finite sum
            below = [r for r in range(0, int(low) + 2) if not _inside(r, condition)]
            chance = 1 - sum(sp.exp(-m) * m ** r / sp.factorial(r) for r in below)
            steps.append(f"P({query}) = 1 minus the chance of r = {', '.join(map(str, below))}: {'; '.join(term(r) for r in below)}. So 1 - {decimal(1 - chance, 5)} = {decimal(chance, 5)}")
        else:
            chosen = [r for r in range(0, int(high) + 1) if _inside(r, condition)]
            chance = sum(sp.exp(-m) * m ** r / sp.factorial(r) for r in chosen)
            steps.append(f"P({query}) adds r = {', '.join(map(str, chosen))}: {'; '.join(term(r) for r in chosen)}. The total is {decimal(chance, 5)}")
        answer.append(f"P({query}) = {decimal(chance, 5)}{_count_line(chance, total)}")
    return Result("Poisson distribution", answer, steps)


def poisson_equation(a, r, b, s):
    """a P(X=r) = b P(X=s): the value of m."""
    m = sp.Rational(a) * sp.factorial(s) / (sp.Rational(b) * sp.factorial(r))
    value = sp.root(m, s - r)
    return Result("Poisson distribution", [f"m = {show(value)}"], [f"Put P(X=r) = e^(-m) m^r / r! on both sides: {a}·e^(-m) m^{r}/{r}! = {b}·e^(-m) m^{s}/{s}!", "e^(-m) cancels on both sides",
                                                                   f"Collect the powers of m: m^{s - r} = {a} × {s}! ÷ ({b} × {r}!) = {text_of(m)}", f"Take the root: m = {show(value)}"])


def normal(text):
    """normal mean=70 sd=5 P(X<65) P(65<X<75) repeats=1000"""
    given = _options(text)
    mean, sd = _f(_num(given["mean"])), _f(_num(given["sd"]))
    total = _f(_num(given["repeats"])) if "repeats" in given else None
    answer, steps = [], []
    for query in _queries(text):
        low, _, high, _ = _condition(query)
        chance = scipy_stats.norm.cdf(high, mean, sd) - scipy_stats.norm.cdf(low, mean, sd)
        answer.append(f"P({query}) = {chance:.4f}{_count_line(chance, total)}")
        z = [f"z = ({v:g} - {mean:g}) ÷ {sd:g} = {(v - mean) / sd:.2f}, with area {scipy_stats.norm.cdf(v, mean, sd):.4f} to its left" for v in (low, high) if math.isfinite(v)]
        steps.append(f"For {query}: " + "; ".join(z) + f". The area wanted is {chance:.4f}")
        if total:
            steps.append(f"Expected count = {chance:.4f} × {total:g} = {chance * total:.1f}")
    return Result("Normal distribution", answer, steps)


def uniform(text):
    given = _options(text)
    a, b = _num(given["a"]), _num(given["b"])
    answer = [f"Mean = {show((a + b) / 2)}", f"s.d. = {show(sp.sqrt((b - a) ** 2 / 12))}"]
    steps = [f"Density = 1 ÷ (b - a) = 1 ÷ ({text_of(b)} - {text_of(a)}) = {text_of(1 / (b - a))} between a and b", f"Mean = (a + b) ÷ 2 = {show((a + b) / 2)}",
             f"Variance = (b - a)² ÷ 12 = {show((b - a) ** 2 / 12)}, and s.d. is its square root, {show(sp.sqrt((b - a) ** 2 / 12))}"]
    for query in _queries(text):
        low, _, high, _ = _condition(query)
        low, high = max(low, _f(a)), min(high, _f(b))
        value = sp.nsimplify((sp.Float(high) - sp.Float(low)) / (b - a), rational=True)
        answer.append(f"P({query}) = {show(value)}")
        steps.append(f"P({query}) = width × density = ({high:g} - {low:g}) × {text_of(1 / (b - a))} = {show(value)}")
    return Result("Uniform distribution", answer, steps)


def density(text):
    """pdf kx^2 from 0 to 3; P(1<x<2); P(x<=1); P(x>1): the constant, chances, mean and variance."""
    match = re.match(r"\s*(.+?)\s+from\s+(\S+)\s+to\s+([^\s;]+)", text)
    if not match:
        raise ValueError("Write it as: pdf kx^2 from 0 to 3; P(1<x<2)")
    x, k = sp.symbols("x k")
    f, low, high = read_expression(match[1]), _num(match[2]), _num(match[3])
    answer, steps = [], []
    if f.has(k):
        root = sp.solve(sp.integrate(f, (x, low, high)) - 1, k)[0]
        answer.append(f"k = {show(root)}")
        steps.append(f"The total area must be 1: ∫ {show(f)} dx from {text_of(low)} to {text_of(high)} = {show(sp.integrate(f, (x, low, high)))} = 1, which gives k = {text_of(root)}")
        f = f.subs(k, root)
    steps.append(f"The density is f(x) = {show(f)} from {text_of(low)} to {text_of(high)}")
    mean = sp.integrate(x * f, (x, low, high))
    variance = sp.integrate(x ** 2 * f, (x, low, high)) - mean ** 2
    for query in _queries(text):
        a, _, b, _ = _condition(query)
        a, b = max(a, _f(low)), min(b, _f(high))
        value = sp.integrate(f, (x, sp.nsimplify(a), sp.nsimplify(b)))
        answer.append(f"P({query}) = {show(value)}")
        steps.append(f"P({query}) = ∫ f(x) dx from {a:g} to {b:g} = {show(value)}")
    answer += [f"Mean = {show(mean)}", f"Variance = {show(variance)}"]
    steps += [f"Mean = ∫ x·f(x) dx = {show(mean)}", f"∫ x²·f(x) dx = {show(variance + mean ** 2)}, so variance = that - mean² = {show(variance)}"]
    return Result("Continuous distribution", answer, steps)


# ---------- sampling ----------

def sampling(text, replacement=True):
    """sampling population=2,3,6,8,11; size=2 and 'with replacement' or 'without replacement'."""
    given = _lists(text)
    population = read_numbers(given["population"])
    size = int(re.search(r"size\s*=\s*(\d+)", text)[1])
    replacement = not re.search(r"without", text, re.IGNORECASE)
    n, mean, squares = _spread(population, [1] * len(population))
    sd = sp.sqrt(squares / n)
    samples = list(itertools.product(population, repeat=size)) if replacement else list(itertools.combinations(population, size))
    means = [sum(s) / sp.Integer(size) for s in samples]
    mean_of_means = sum(means) / len(means)
    se = sp.sqrt(sum((m - mean_of_means) ** 2 for m in means) / len(means))
    answer = [f"Population mean = {show(mean)}", f"Population s.d. = {show(sd)}",
              f"Number of samples = {len(samples)}", f"Mean of the sample means = {show(mean_of_means)}", f"Standard error = {show(se)}"]
    steps = [f"Population mean = {text_of(mean * n)} ÷ {text_of(n)} = {show(mean)}, and s.d. = √({text_of(squares)} ÷ {text_of(n)}) = {show(sd)}",
             f"With{'' if replacement else 'out'} replacement, the samples of size {size} number {len(samples)}",
             "Their means: " + ", ".join(text_of(m) for m in means[:12]) + (", ..." if len(means) > 12 else ""),
             f"Mean of the sample means = {text_of(sum(means))} ÷ {len(means)} = {show(mean_of_means)}",
             f"Standard error, the s.d. of the sample means: {show(se)}"]
    return Result("Sampling distribution of means", answer, steps)


def _standard_error(sd, n, population=None):
    se = sd / math.sqrt(n)
    if population:
        se *= math.sqrt((population - n) / (population - 1))
    return se


def sample_mean(text):
    """clt mu=51.4 sd=6.8 n=64 P(xbar>52.9)"""
    given = _options(text)
    mu, sd, n = _f(_num(given["mu"])), _f(_num(given["sd"])), _f(_num(given["n"]))
    population = _f(_num(given["population"])) if "population" in given else None
    se = _standard_error(sd, n, population)
    answer = [f"Standard error = {se:.4f}"]
    steps = ["By the central limit theorem, the sample mean is close to normal around μ", f"Standard error = σ ÷ √n = {sd:g} ÷ √{n:g}" + (f", times √((N - n) ÷ (N - 1)) for a population of {population:g}" if population else "") + f" = {se:.4f}"]
    for query in _queries(text):
        low, _, high, _ = _condition(query)
        chance = scipy_stats.norm.cdf(high, mu, se) - scipy_stats.norm.cdf(low, mu, se)
        answer.append(f"P({query}) = {chance:.4f}")
        z = [f"z = ({v:g} - {mu:g}) ÷ {se:.4f} = {(v - mu) / se:.2f}" for v in (low, high) if math.isfinite(v)]
        steps.append(f"For {query}: " + ", ".join(z) + f". The area wanted is {chance:.4f}")
    return Result("Central limit theorem", answer, steps)


def sample_proportion(text):
    """proportion p=0.14 n=128 P(phat>0.18)"""
    given = _options(text)
    p, n = _f(_num(given["p"])), _f(_num(given["n"]))
    se = math.sqrt(p * (1 - p) / n)
    answer = [f"Mean = {p:g}, standard error = {se:.4f}"]
    steps = ["The sample proportion is close to normal around p", f"Standard error = √(p(1 - p) ÷ n) = √({p:g} × {1 - p:g} ÷ {n:g}) = {se:.4f}"]
    for query in _queries(text):
        low, _, high, _ = _condition(query.lower().replace("phat", "x"))
        chance = scipy_stats.norm.cdf(high, p, se) - scipy_stats.norm.cdf(low, p, se)
        answer.append(f"P({query}) = {chance:.4f}")
        z = [f"z = ({v:g} - {p:g}) ÷ {se:.4f} = {(v - p) / se:.2f}" for v in (low, high) if math.isfinite(v)]
        steps.append(f"For {query}: " + ", ".join(z) + f". The area wanted is {chance:.4f}")
    return Result("Sampling distribution of a proportion", answer, steps)


def sum_of_normals(text):
    given = _lists(text)
    means, sds = read_numbers(given["means"]), read_numbers(given["sds"])
    return Result("Sum of independent values", [f"Mean of the sum = {decimal(sum(means))}", f"s.d. of the sum = {decimal(sp.sqrt(sum(s ** 2 for s in sds)))}"], [f"The mean of a sum is the sum of the means: {' + '.join(text_of(m) for m in means)} = {decimal(sum(means))}",
                   f"For independent values the variances add: {' + '.join(f'{text_of(v)}²' for v in sds)} = {decimal(sum(v ** 2 for v in sds))}", f"s.d. is its square root: {decimal(sp.sqrt(sum(v ** 2 for v in sds)))}"])


# ---------- estimation and tests ----------

def _t_or_z(n, known_sd):
    """t for a small sample with s.d. from the sample, else z."""
    return "t" if n < 30 and not known_sd else "z"


def confidence_interval(text):
    """ci mean=900 sd=50 n=35 conf=95; or ci data=60,80,...; 'known' when the population s.d. is known; z=1.96 to use a given value."""
    given = _options(text)
    lists = _lists(text)
    if "data" in lists:
        values = np.array([_f(v) for v in read_numbers(lists["data"])])
        mean, sd, n = values.mean(), values.std(ddof=1), len(values)
    else:
        mean, sd, n = _f(_num(given["mean"])), _f(_num(given["sd"])), _f(_num(given["n"]))
    conf = _f(_num(given.get("conf", 95))) / 100
    kind = _t_or_z(n, bool(re.search(r"\bknown\b", text)))
    if "z" in given:
        critical, kind = _f(_num(given["z"])), "z"
    elif kind == "t":
        critical = scipy_stats.t.ppf((1 + conf) / 2, n - 1)
    else:
        critical = scipy_stats.norm.ppf((1 + conf) / 2)
    margin = critical * sd / math.sqrt(n)
    answer = [f"{conf * 100:g}% confidence interval: ({mean - margin:.4f}, {mean + margin:.4f})"]
    steps = [f"Mean = {mean:.4f}, s.d. = {sd:.4f}, n = {n:g}", f"For {conf * 100:g}% confidence, the critical {kind} value is {critical:.4f}" + (f", with {n - 1:g} degrees of freedom" if kind == "t" else ""),
             f"Margin = critical × s.d. ÷ √n = {critical:.4f} × {sd:.4f} ÷ √{n:g} = {margin:.4f}", f"Interval = mean ± margin = {mean:.4f} ± {margin:.4f} = ({mean - margin:.4f}, {mean + margin:.4f})"]
    return Result("Confidence interval", answer, steps)


def _decision(statistic, critical, tail):
    if tail == "left":
        return statistic < -critical
    if tail == "right":
        return statistic > critical
    return abs(statistic) > critical


def _tail_of(text):
    given = _options(text)
    return given.get("tail", "two").lower()


def mean_test(text, kind):
    """ztest mean=1038 mu=1000 sd=146 n=64 alpha=0.05 tail=right critical=1.645"""
    given = _options(text)
    mean, mu, sd, n = _f(_num(given["mean"])), _f(_num(given["mu"])), _f(_num(given["sd"])), _f(_num(given["n"]))
    alpha, tail = _f(_num(given.get("alpha", "0.05"))), _tail_of(text)
    statistic = (mean - mu) / (sd / math.sqrt(n))
    area = alpha if tail != "two" else alpha / 2
    critical = _f(_num(given["critical"])) if "critical" in given else (
        scipy_stats.norm.ppf(1 - area) if kind == "z" else scipy_stats.t.ppf(1 - area, n - 1))
    reject = _decision(statistic, critical, tail)
    answer = [f"{kind} = {statistic:.4f}", f"Critical value = {critical:.4f}", ("Reject H0" if reject else "Do not reject H0") + f" at the {alpha * 100:g}% level"]
    hypothesis = {"two": f"H1: μ ≠ {mu:g}", "right": f"H1: μ > {mu:g}", "left": f"H1: μ < {mu:g}"}[tail]
    return Result(f"{kind}-test for a mean", answer, [f"H0: μ = {mu:g}, {hypothesis}", f"Standard error = s ÷ √n = {sd:g} ÷ √{n:g} = {sd / math.sqrt(n):.4f}",
                                                      f"{kind} = (mean - μ) ÷ standard error = ({mean:g} - {mu:g}) ÷ {sd / math.sqrt(n):.4f} = {statistic:.4f}",
                                                      f"At the {alpha * 100:g}% level, {'two tails' if tail == 'two' else 'one tail'}" + (f", {n - 1:g} degrees of freedom" if kind == "t" else "") + f": critical value {critical:.4f}",
                                                      f"|{kind}| = {abs(statistic):.4f} is {'beyond' if reject else 'inside'} the critical value, so {'reject' if reject else 'do not reject'} H0"])


def two_sample_test(text):
    """twosample mean1=22.73 sd1=0.164 n1=6 mean2=21.94 sd2=0.24 n2=5 diff=0 alpha=0.05 tail=two"""
    given = _options(text)
    m1, s1, n1 = _f(_num(given["mean1"])), _f(_num(given["sd1"])), _f(_num(given["n1"]))
    m2, s2, n2 = _f(_num(given["mean2"])), _f(_num(given["sd2"])), _f(_num(given["n2"]))
    diff, alpha, tail = _f(_num(given.get("diff", 0))), _f(_num(given.get("alpha", "0.05"))), _tail_of(text)
    area = alpha if tail != "two" else alpha / 2
    if n1 >= 30 and n2 >= 30:
        kind, se = "z", math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        critical = scipy_stats.norm.ppf(1 - area)
    else:
        pooled = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        kind, se = "t", math.sqrt(pooled * (1 / n1 + 1 / n2))
        critical = scipy_stats.t.ppf(1 - area, n1 + n2 - 2)
    if "critical" in given:
        critical = _f(_num(given["critical"]))
    statistic = (m1 - m2 - diff) / se
    reject = _decision(statistic, critical, tail)
    answer = [f"{kind} = {statistic:.4f}", f"Critical value = {critical:.4f}", ("Reject H0" if reject else "Do not reject H0") + f" at the {alpha * 100:g}% level"]
    pooling = f"Pooled variance = ((n1-1)s1² + (n2-1)s2²) ÷ (n1+n2-2) = {pooled:.4f}, so standard error = √(pooled × (1/n1 + 1/n2)) = {se:.4f}" if kind == "t" else f"Standard error = √(s1²/n1 + s2²/n2) = {se:.4f}"
    return Result(f"Two-sample {kind}-test", answer, [f"H0: μ1 - μ2 = {diff:g}", pooling, f"{kind} = (mean1 - mean2 - {diff:g}) ÷ standard error = ({m1:g} - {m2:g} - {diff:g}) ÷ {se:.4f} = {statistic:.4f}",
                                                      f"At the {alpha * 100:g}% level: critical value {critical:.4f}", f"|{kind}| = {abs(statistic):.4f} is {'beyond' if reject else 'inside'} the critical value, so {'reject' if reject else 'do not reject'} H0"])


def chi_square(text):
    """chisquare observed=20,22,...; expected=20,20,...; alpha=0.05 (equal shares when expected is left out)"""
    given = _lists(text)
    observed = [_f(v) for v in read_numbers(given["observed"])]
    expected = [_f(v) for v in read_numbers(given["expected"])] if "expected" in given else [sum(observed) / len(observed)] * len(observed)
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have the same length")
    statistic = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    alpha = _f(_num(given.get("alpha", "0.05")))
    critical = scipy_stats.chi2.ppf(1 - alpha, len(observed) - 1)
    reject = statistic > critical
    answer = [f"χ² = {statistic:.4f}", f"Critical value with {len(observed) - 1} degrees of freedom = {critical:.4f}",
              ("Reject H0: the data do not fit" if reject else "Do not reject H0: the data fit") + f" at the {alpha * 100:g}% level"]
    return Result("Chi-square goodness of fit", answer, ["H0: the data fit the expected counts", "Expected: " + ", ".join(f"{e:g}" for e in expected), "χ² = Σ(O - E)² ÷ E",
                                                         *[f"O = {o:g}, E = {e:g}: ({o:g} - {e:g})² ÷ {e:g} = {(o - e) ** 2 / e:.4f}" for o, e in zip(observed, expected)], f"Add them: χ² = {statistic:.4f}",
                                                         f"Degrees of freedom = {len(observed)} - 1 = {len(observed) - 1}, so at the {alpha * 100:g}% level the critical value is {critical:.4f}",
                                                         f"{statistic:.4f} is {'above' if reject else 'below'} {critical:.4f}, so {'reject' if reject else 'do not reject'} H0"])
