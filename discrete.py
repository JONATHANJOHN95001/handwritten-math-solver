"""Discrete Mathematics: counting, inclusion-exclusion, rook polynomials, generating functions and recurrences."""
import itertools
import math
import re

import sympy as sp

from results import Result, parts_of, read_numbers, show, show_plain
from solver import read_equation, read_expression

X = sp.Symbol("x")


def _ints(text):
    return [int(v) for v in read_numbers(text)]


def _pairs(text):
    """'a=1 b=2,3' as {'a': '1', 'b': '2,3'} with values running to the next key or ';'."""
    found = {}
    for part in parts_of(text):
        for key, value in re.findall(r"([A-Za-z_]+)\s*=\s*(.+?)(?=\s+[A-Za-z_]+\s*=|$)", part):
            found[key.lower()] = value.strip()
    return found


# ---------- counting ----------

def product_rule(text):
    numbers = _ints(text)
    total = math.prod(numbers)
    return Result("Product rule", [f"{' × '.join(map(str, numbers))} = {total}"], ["Choices made one after another multiply"])


def pigeonhole(items, boxes):
    least = math.ceil(sp.Rational(items, boxes))
    return Result("Pigeonhole principle", [f"At least one box holds {least} or more"], [f"{items} ÷ {boxes} = {show(sp.Rational(items, boxes))}, rounded up"])


def derangements(n):
    d = int(sp.subfactorial(n))
    return Result("Derangements", [f"D({n}) = {d}", f"At least one in the right place: {math.factorial(n)} - {d} = {math.factorial(n) - d}"],
              [f"D(n) = n! (1 - 1/1! + 1/2! - ... ± 1/n!)", f"{n}! = {math.factorial(n)}"])


def integer_solutions(text):
    """solutions total=30 lower=2,3,4,2,0 or solutions total=24 n=4 lower=3 upper=8"""
    given = _pairs(text)
    total = int(read_expression(given["total"]))
    lower_list = re.split(r"\s*,\s*", given.get("lower", "0"))
    n = int(given["n"]) if "n" in given else max(len(lower_list), len(re.split(r"\s*,\s*", given.get("upper", "0"))))
    lower = [int(read_expression(v)) if v else 0 for v in lower_list]
    lower = lower * n if len(lower) == 1 else lower
    if "upper" in given:
        items = re.split(r"\s*,\s*", given["upper"])
        items = items * n if len(items) == 1 else items
        upper = [int(read_expression(v)) if v else None for v in items]
    else:
        upper = [None] * n
    if all(u is None for u in upper):
        left = total - sum(lower)
        count = sp.binomial(left + n - 1, n - 1) if left >= 0 else 0
        return Result("Number of integer solutions", [f"{count} solutions"], [f"Take away the lower limits: {left} left to share among {n}", f"C({left}+{n}-1, {n}-1) = C({left + n - 1}, {n - 1}) = {count}"])
    factors = []
    for low, high in zip(lower, upper):
        if high is None:
            factors.append(X ** low / (1 - X))
        else:
            factors.append(sum(X ** k for k in range(low, high + 1)))
    gf = sp.prod(factors)
    shift = -min(min(lower), 0)
    series = sp.series(sp.expand(gf * X ** (shift * n)) if not gf.has(1 / (1 - X)) else gf * X ** (shift * n), X, 0, total + shift * n + 1).removeO()
    count = sp.Poly(series, X).coeff_monomial(X ** (total + shift * n))
    return Result("Number of integer solutions", [f"{count} solutions"], [f"Generating function: {show(gf)}", f"Coefficient of x^{total} in it"])


def inclusion_exclusion(text):
    """inclusion singles=12,20,20,8; pairs=5,7,4,16,4,3; triples=3,2,2,3; all=2; none=71"""
    given = _pairs(text)
    sums, sign, steps = 0, 1, []
    for key in ("singles", "pairs", "triples", "all"):
        if key in given:
            part = sum(_ints(given[key]))
            sums += sign * part
            steps.append(f"Sum of {key}: {part}")
            sign = -sign
    none = int(read_expression(given.get("none", "0")))
    return Result("Inclusion and exclusion", [f"In at least one: {sums}", f"Total: {sums + none}"], steps)


def divisible(low, high, divisors):
    total = 0
    steps = []
    for size in range(1, len(divisors) + 1):
        part = 0
        for group in itertools.combinations(divisors, size):
            lcm = math.lcm(*group)
            part += high // lcm - (low - 1) // lcm
        steps.append(f"Taking {size} at a time: {part}")
        total += part if size % 2 else -part
    count = high - low + 1
    return Result("Inclusion and exclusion", [f"Divisible by at least one of {divisors}: {total}", f"Divisible by none: {count - total}"], steps)


def patterns(n, words):
    """Permutations of n different letters containing none of the given patterns, which share no letters."""
    if len(set("".join(words))) != len("".join(words)):
        raise ValueError("These patterns share letters, so they cannot all be treated as blocks")
    total, steps = 0, []
    for size in range(0, len(words) + 1):
        part = 0
        for group in itertools.combinations(words, size):
            part += math.factorial(n - sum(len(w) - 1 for w in group))
        steps.append(f"With {size} patterns fixed: {part}")
        total += part if size % 2 == 0 else -part
    return Result("Inclusion and exclusion", [f"{total} permutations contain none of the patterns"], steps)


def rook_polynomial(cells):
    """r(C, x) for a board given as a list of cells, by expanding on one cell."""
    def expand(board):
        if not board:
            return sp.Integer(1)
        first, rest = board[0], board[1:]
        removed = [c for c in rest if c[0] != first[0] and c[1] != first[1]]
        return expand(removed) * X + expand(rest)
    return sp.expand(expand(list(cells)))


def rook(text):
    """rook 3x3, or rook cells=(1,1),(1,2),(2,3)"""
    size = re.search(r"(\d+)\s*[x×]\s*(\d+)", text)
    if size:
        cells = list(itertools.product(range(int(size[1])), range(int(size[2]))))
    else:
        cells = [tuple(map(int, c)) for c in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)]
    if not cells:
        raise ValueError("Give the board as 3x3 or as a list of cells like (1,1),(1,2)")
    return Result("Rook polynomial", [f"r(C, x) = {show(rook_polynomial(cells))}"], [f"Board of {len(cells)} cells, expanded one cell at a time"])


def forbidden(text):
    """forbidden B1:apple; B2:apple; B3:banana,mango; B4:orange with the items taken from the list."""
    rows = {}
    for part in parts_of(text):
        name, _, items = part.partition(":")
        rows[name.strip()] = [item.strip() for item in items.split(",") if item.strip()]
    people = list(rows)
    items = sorted({item for values in rows.values() for item in values} | set(re.findall(r"\bitems?\s*=\s*(.+)", text)))
    if len(items) < len(people):
        items += [f"other{i}" for i in range(len(people) - len(items))]
    count = sum(all(item not in rows[p] for p, item in zip(people, order)) for order in itertools.permutations(items, len(people)))
    cells = [(people.index(p), items.index(i)) for p in people for i in rows[p]]
    poly = rook_polynomial(cells)
    return Result("Arrangements with forbidden positions", [f"{count} ways"], [f"Rook polynomial of the forbidden board: {show(poly)}"])


# ---------- generating functions ----------

def coefficient(power, text):
    """The coefficient of x^power in a formula, which may be a rational function."""
    text = text.replace("⋯", "...").replace("…", "...")
    text = re.sub(r"\(\s*x\s*\+\s*2x\^?2\s*\+\s*3x\^?3\s*(?:\+\s*4x\^?4\s*)?\+\s*\.\.\.\s*\)", "(x/(1-x)^2)", text)
    text = re.sub(r"\(\s*1\s*\+\s*x\s*\+\s*x\^?2\s*\+\s*x\^?3\s*\+\s*\.\.\.\s*\)", "(1/(1-x))", text)
    expression = read_expression(text)
    series = sp.series(expression, X, 0, power + 1).removeO()
    value = sp.Poly(series, X).coeff_monomial(X ** power) if series.has(X) else (series if power == 0 else 0)
    return Result("Coefficient in a generating function", [f"Coefficient of x^{power} = {show(value)}"], [f"Expanded {show(expression)} up to x^{power}"])


def _polynomial_rule(numbers):
    """The polynomial through the sequence, if its differences run out before the terms do, or the whole thing has only a few terms."""
    n = sp.Symbol("n")
    return sp.interpolate(list(zip(range(len(numbers)), numbers)), n), n


def generating_function(text):
    """gf 1,2,3,4,... or gf 0,1,-2,3,-4,... or gf 1^3,2^3,3^3,4^3,..."""
    infinite = "..." in text or "…" in text or "⋯" in text
    items = [item for item in re.split(r"[,\s]+", re.sub(r"\.\.\.|…|⋯", "", text)) if item]
    numbers = [read_expression(item) for item in items]
    if not infinite:
        return Result("Generating function", [f"G(x) = {show(sum(a * X ** k for k, a in enumerate(numbers)))}"], [])
    alternating = len(numbers) > 2 and all(numbers[k] * numbers[k + 1] <= 0 for k in range(len(numbers) - 1)) and any(a < 0 for a in numbers)
    size = [abs(a) for a in numbers] if alternating else numbers
    ratio = [sp.Rational(size[k + 1], size[k]) for k in range(len(size) - 1) if size[k] != 0]
    if ratio and len(set(ratio)) == 1 and ratio[0] != 1:
        sign = -1 if alternating else 1
        return Result("Generating function", [f"G(x) = {show(sp.simplify(size[0] / (1 - sign * ratio[0] * X)))}"], ["A geometric sequence"])
    rule, n = _polynomial_rule(size)
    more = [rule.subs(n, k) * ((-1) ** k * (1 if numbers[1] >= 0 else -1) * -1 if alternating else 1) for k in range(14)]
    sequence = sp.sequence(more, (n, 0, 13))
    _, gf = sequence.find_linear_recurrence(14, gfvar=X)
    if gf is None:
        raise ValueError("I could not find a pattern in this sequence")
    return Result("Generating function", [f"G(x) = {show(sp.factor(gf))}"], [f"Terms follow {show(rule)}" + (" with alternating signs" if alternating else "")])


def sequence_of(text, exponential=False, count=8):
    """gf sequence of (3+x)^3, or the sequence an exponential generating function produces."""
    expression = read_expression(text)
    series = sp.series(expression, X, 0, count).removeO()
    terms = [sp.Poly(series, X).coeff_monomial(X ** k) * (sp.factorial(k) if exponential else 1) for k in range(count)]
    name = "exponential generating function" if exponential else "generating function"
    return Result(f"Sequence from a {name}", ["Sequence: " + ", ".join(show_plain(t) for t in terms) + ", ..."], [f"Expanded up to x^{count - 1}"])


def exponential_gf(text):
    """egf 1,2,3,0,0,0"""
    numbers = read_numbers(text)
    formula = sum(a * X ** k / sp.factorial(k) for k, a in enumerate(numbers))
    return Result("Exponential generating function", [f"E(x) = {show(formula)}"], [])


def partitions(n):
    count = int(sp.functions.combinatorial.numbers.partition(n))
    return Result("Partitions", [f"p({n}) = {count}"], [f"Coefficient of x^{n} in the product of 1/(1-x^k)"])


def arrangements(counts, r):
    """Arrangements of r items from a multiset, by exponential generating function."""
    gf = sp.prod(sum(X ** k / sp.factorial(k) for k in range(c + 1)) for c in counts)
    value = sp.Poly(sp.expand(gf), X).coeff_monomial(X ** r) * sp.factorial(r)
    return Result("Arrangements from an exponential generating function", [f"{value} arrangements"], [f"Coefficient of x^{r}/{r}! in {show(gf)}"])


def word_arrangements(word, r):
    counts = [word.upper().count(letter) for letter in sorted(set(word.upper()))]
    result = arrangements(counts, r)
    result.steps.insert(0, "Letter counts: " + ", ".join(f"{letter}: {word.upper().count(letter)}" for letter in sorted(set(word.upper()))))
    return result


# ---------- recurrences ----------

def _normalise_recurrence(text):
    """a(n+1), a_(n+1), a_n and an+1 become one form, a(n+1), so SymPy can read them."""
    text = re.sub(r"(?<![A-Za-z])([a-zA-Z])_?\(\s*n\s*([+-]\s*\d+)?\s*\)", lambda m: f"{m[1]}[n{(m[2] or '').replace(' ', '')}]", text)
    text = re.sub(r"(?<![A-Za-z])([a-zA-Z])_n\s*([+-]\s*\d+)?(?![\w(])", lambda m: f"{m[1]}[n{(m[2] or '').replace(' ', '')}]", text)
    text = re.sub(r"(?<![A-Za-z])([aF])n\s*([+-]\s*\d+)?(?![\w(])", lambda m: f"{m[1]}[n{(m[2] or '').replace(' ', '')}]", text)
    return text


def recurrence(text):
    """recurrence a(n+1)=4a(n); a(0)=3"""
    parts = parts_of(text)
    main = _normalise_recurrence(parts[0])
    letters = re.findall(r"(?<![A-Za-z])([a-zA-Z])\[n", main)
    if not letters:
        raise ValueError("Write the recurrence with a(n), a(n+1), a(n-1) and so on")
    letter = letters[0]
    n = sp.Symbol("n", integer=True)
    function = sp.Function(letter)
    placeholders = {}

    def stash(match):
        key = f"Z{len(placeholders)}"
        shift = int((match[1] or "0").replace(" ", ""))
        placeholders[sp.Symbol(key.lower())] = function(n + shift)
        return f"({key.lower()})"

    body = re.sub(rf"(?<![A-Za-z]){letter}\[n([+-]\d+)?\]", stash, main)
    left, right = read_equation(body)
    expression = (left - right).subs({sp.Symbol("n"): n}).subs(placeholders)
    conditions = {}
    for part in parts[1:]:
        for key, value in re.findall(rf"{letter}_?\(?(\d+)\)?\s*=\s*([^;,\s]+)", part):
            conditions[function(int(key))] = read_expression(value)
    squared = [a for a in expression.atoms(sp.Pow) if a.base.func == function and a.exp == 2]
    steps = []
    if squared:  # a(n)² appears: solve for b(n) = a(n)² first
        b = sp.Function("b")
        expression = expression.replace(lambda e: e.is_Pow and e.base.func == function and e.exp == 2, lambda e: b(e.base.args[0]))
        solution = sp.rsolve(expression, b(n), {b(k.args[0]): v ** 2 for k, v in conditions.items()} or None)
        steps.append("Put b(n) = a(n)², which makes the equation linear")
        return Result("Recurrence relation", [f"{letter}(n)² = {show(sp.simplify(solution))}"], steps)
    solution = sp.rsolve(expression, function(n), conditions or None)
    if solution is None:
        raise ValueError("I could not solve this recurrence")
    if not conditions:
        steps.append("C0, C1 are constants fixed by the starting values")
    return Result("Recurrence relation", [f"{letter}(n) = {show(sp.simplify(solution))}"], steps)


def growth(start, percent, periods):
    """A quantity that increases by a percentage each period."""
    factor = 1 + sp.Rational(percent) / 100
    value = start * factor ** periods
    return Result("Recurrence relation", [f"After {periods} periods: {show(value)}"], [f"a(n) = {show_plain(factor)}·a(n-1), a(0) = {start}", f"a(n) = {start}·({show_plain(factor)})^n"])


def hanoi(n=None):
    answer = ["Moves for n disks: a(n) = 2^n - 1"]
    if n is not None:
        answer.append(f"For {n} disks: {2 ** n - 1}")
    return Result("Tower of Hanoi", answer, ["a(n) = 2·a(n-1) + 1, a(1) = 1"])


def handshakes(n=None):
    answer = ["Handshakes among n people: a(n) = n(n-1)/2"]
    if n is not None:
        answer.append(f"For {n} people: {n * (n - 1) // 2}")
    return Result("Handshakes", answer, ["a(n) = a(n-1) + (n-1), a(2) = 1"])
