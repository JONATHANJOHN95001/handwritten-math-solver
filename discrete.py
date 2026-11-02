"""Discrete Mathematics: counting, inclusion-exclusion, rook polynomials, generating functions and recurrences."""
import itertools
import math
import re

import sympy as sp

from results import Result, parts_of, read_numbers, show, show_plain
from solver import read_equation, read_expression
from working import text_of

X = sp.Symbol("x")
COUNTS_OF = {"singles": "counts of one set", "pairs": "counts of two sets", "triples": "counts of three sets", "all": "count of all the sets"}


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
    if not numbers:
        raise ValueError("Give the number of choices at each stage, like: product rule 4,12,3,2")
    total = math.prod(numbers)
    running, lines = numbers[0], [f"Choices made one after another multiply. Start with {numbers[0]}:"]
    for number in numbers[1:]:
        lines.append(f"{running} × {number} = {running * number}")
        running *= number
    return Result("Product rule", [f"{' × '.join(map(str, numbers))} = {total}"], lines)


def pigeonhole(items, boxes):
    least = math.ceil(sp.Rational(items, boxes))
    return Result("Pigeonhole principle", [f"At least one box holds {least} or more"], [f"{items} items go into {boxes} boxes", f"If every box held fewer than {least}, the boxes would hold at most {boxes} × {least - 1} = {boxes * (least - 1)} items, which is fewer than {items}",
                                                                                         f"{items} ÷ {boxes} = {show(sp.Rational(items, boxes))}, rounded up to {least}"])


def derangements(n):
    d = int(sp.subfactorial(n))
    return Result("Derangements", [f"D({n}) = {d}", f"At least one in the right place: {math.factorial(n)} - {d} = {math.factorial(n) - d}"],
              [f"D(n) = n! (1 - 1/1! + 1/2! - ... ± 1/n!)", f"{n}! = {math.factorial(n)}",
               "Multiply out, term by term: " + " ".join(f"{'+' if k % 2 == 0 else '-'} {math.factorial(n) // math.factorial(k)}" for k in range(n + 1)) + f" = {d}",
               f"All arrangements minus those with none in the right place: {math.factorial(n)} - {d} = {math.factorial(n) - d}"])


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
        return Result("Number of integer solutions", [f"{count} solutions"], [f"Take away the lower limits, {' + '.join(map(str, lower))} = {sum(lower)}: {total} - {sum(lower)} = {left} left to share among {n}",
                                                                                 f"Sharing {left} among {n} is choosing where {n - 1} dividers go among {left} + {n - 1} places", f"C({left}+{n}-1, {n}-1) = C({left + n - 1}, {n - 1}) = {count}"])
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
    return Result("Number of integer solutions", [f"{count} solutions"], ["Each unknown gives one bracket, with one power of x for each value it may take", f"Generating function: {show(gf)}",
                                                                          f"A solution picks one power from each bracket so that the powers add up to {total}, so the count is the coefficient of x^{total}", f"Coefficient of x^{total} = {count}"])


def inclusion_exclusion(text):
    """inclusion singles=12,20,20,8; pairs=5,7,4,16,4,3; triples=3,2,2,3; all=2; none=71"""
    given = _pairs(text)
    sums, sign, steps = 0, 1, []
    for key in ("singles", "pairs", "triples", "all"):
        if key in given:
            numbers = _ints(given[key])
            part = sum(numbers)
            sums += sign * part
            written = " + ".join(map(str, numbers)) + (f" = {part}" if len(numbers) > 1 else "")
            steps.append(f"{'Add' if sign > 0 else 'Take away'} the {COUNTS_OF[key]}: {written}, which leaves {sums}")
            sign = -sign
    none = int(read_expression(given.get("none", "0")))
    steps.append(f"In at least one: {sums}. Adding those in none, {none}, gives the total {sums + none}")
    return Result("Inclusion and exclusion", [f"In at least one: {sums}", f"Total: {sums + none}"], steps)


def divisible(low, high, divisors):
    total = 0
    steps = []
    for size in range(1, len(divisors) + 1):
        part = 0
        pieces = []
        for group in itertools.combinations(divisors, size):
            lcm = math.lcm(*group)
            found = high // lcm - (low - 1) // lcm
            part += found
            pieces.append(f"by {lcm}: ⌊{high}/{lcm}⌋ - ⌊{low - 1}/{lcm}⌋ = {found}")
        total += part if size % 2 else -part
        steps.append(f"Taking {size} at a time, {'; '.join(pieces)}. Their sum, {part}, is {'added' if size % 2 else 'taken away'}, which leaves {total}")
    count = high - low + 1
    steps.append(f"There are {count} numbers from {low} to {high}, so divisible by none: {count} - {total} = {count - total}")
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
        total += part if size % 2 == 0 else -part
        steps.append(f"With {size} patterns kept together as blocks: {part}. It is {'added' if size % 2 == 0 else 'taken away'}, which leaves {total}")
    return Result("Inclusion and exclusion", [f"{total} permutations contain none of the patterns"], ["A pattern kept together counts as one letter, so it lowers the number of things to arrange", *steps])


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
    poly = rook_polynomial(cells)
    lines = [f"Board of {len(cells)} cells", "The coefficient of x^k counts the ways to place k rooks with no two in a row or column",
             "Expand on one cell: r(C) = x·r(C with its row and column removed) + r(C with only that cell removed)",
             *[f"{k} rook{'' if k == 1 else 's'}: {sp.Poly(poly, X).coeff_monomial(X ** k)} way{'' if sp.Poly(poly, X).coeff_monomial(X ** k) == 1 else 's'}" for k in range(sp.degree(poly, X) + 1)]]
    return Result("Rook polynomial", [f"r(C, x) = {show(poly)}"], lines)


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
    n = len(items)
    coefficients = [sp.Poly(poly, X).coeff_monomial(X ** k) for k in range(sp.degree(poly, X) + 1)]
    terms = " ".join(f"{'+' if k % 2 == 0 else '-'} {c}·{n - k}!" for k, c in enumerate(coefficients))
    lines = [f"Mark the forbidden places on a board of {len(people)} rows and {n} columns", f"Rook polynomial of the forbidden board: {show(poly)}"]
    if n == len(people):
        lines.append(f"Inclusion and exclusion: ways = {n}! - r1·{n - 1}! + r2·{n - 2}! - ... = {terms.lstrip('+ ')} = {count}")
    lines.append(f"Counting the allowed arrangements gives {count} ways")
    return Result("Arrangements with forbidden positions", [f"{count} ways"], lines)


# ---------- generating functions ----------

def coefficient(power, text):
    """The coefficient of x^power in a formula, which may be a rational function."""
    text = text.replace("⋯", "...").replace("…", "...")
    text = re.sub(r"\(\s*x\s*\+\s*2x\^?2\s*\+\s*3x\^?3\s*(?:\+\s*4x\^?4\s*)?\+\s*\.\.\.\s*\)", "(x/(1-x)^2)", text)
    text = re.sub(r"\(\s*1\s*\+\s*x\s*\+\s*x\^?2\s*\+\s*x\^?3\s*\+\s*\.\.\.\s*\)", "(1/(1-x))", text)
    expression = read_expression(text)
    series = sp.series(expression, X, 0, power + 1).removeO()
    value = sp.Poly(series, X).coeff_monomial(X ** power) if series.has(X) else (series if power == 0 else 0)
    return Result("Coefficient in a generating function", [f"Coefficient of x^{power} = {show(value)}"], [f"The expression: {show(expression)}", f"Expand it as a power series up to x^{power}: {show(series)}", f"Read off the number in front of x^{power}: {show(value)}"])


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
        return Result("Generating function", [f"G(x) = {show(sum(a * X ** k for k, a in enumerate(numbers)))}"], ["Each term of the sequence goes in front of its own power of x: G(x) = a0 + a1·x + a2·x² + ...",
                                                                                                                  *[f"a{k} = {show_plain(a)} gives {show(a * X ** k)}" for k, a in enumerate(numbers)]])
    alternating = len(numbers) > 2 and all(numbers[k] * numbers[k + 1] <= 0 for k in range(len(numbers) - 1)) and any(a < 0 for a in numbers)
    size = [abs(a) for a in numbers] if alternating else numbers
    ratio = [sp.Rational(size[k + 1], size[k]) for k in range(len(size) - 1) if size[k] != 0]
    if ratio and len(set(ratio)) == 1 and ratio[0] != 1:
        sign = -1 if alternating else 1
        return Result("Generating function", [f"G(x) = {show(sp.simplify(size[0] / (1 - sign * ratio[0] * X)))}"], [f"Each term is {show_plain(sign * ratio[0])} times the one before, so this is a geometric sequence", f"G(x) = a0 ÷ (1 - rx), with a0 = {show_plain(size[0])} and r = {show_plain(sign * ratio[0])}"])
    rule, n = _polynomial_rule(size)
    more = [rule.subs(n, k) * ((-1) ** k * (1 if numbers[1] >= 0 else -1) * -1 if alternating else 1) for k in range(14)]
    sequence = sp.sequence(more, (n, 0, 13))
    _, gf = sequence.find_linear_recurrence(14, gfvar=X)
    if gf is None:
        raise ValueError("I could not find a pattern in this sequence")
    return Result("Generating function", [f"G(x) = {show(sp.factor(gf))}"], [f"The differences of the terms run out, so the terms follow a polynomial: {show(rule)}" + (", with alternating signs" if alternating else ""),
                                                                           "Terms that follow a polynomial obey a fixed recurrence, and its generating function is a fraction", f"Worked out from more terms of the same rule: G(x) = {show(sp.factor(gf))}"])


def sequence_of(text, exponential=False, count=8):
    """gf sequence of (3+x)^3, or the sequence an exponential generating function produces."""
    expression = read_expression(text)
    series = sp.series(expression, X, 0, count).removeO()
    terms = [sp.Poly(series, X).coeff_monomial(X ** k) * (sp.factorial(k) if exponential else 1) for k in range(count)]
    name = "exponential generating function" if exponential else "generating function"
    return Result(f"Sequence from a {name}", ["Sequence: " + ", ".join(show_plain(t) for t in terms) + ", ..."], [f"Expand {show(expression)} as a power series up to x^{count - 1}: {show(series)}", "Read off the number in front of each power of x" + (", multiplied by k! for x^k" if exponential else "")])


def exponential_gf(text):
    """egf 1,2,3,0,0,0"""
    numbers = read_numbers(text)
    formula = sum(a * X ** k / sp.factorial(k) for k, a in enumerate(numbers))
    return Result("Exponential generating function", [f"E(x) = {show(formula)}"], ["E(x) = a0 + a1·x + a2·x²/2! + a3·x³/3! + ...", *[f"a{k} = {show_plain(a)} gives {show(a * X ** k / sp.factorial(k))}" for k, a in enumerate(numbers)]])


def partitions(n):
    count = int(sp.functions.combinatorial.numbers.partition(n))
    lines = [f"A partition of {n} writes it as a sum, whatever the order of the parts", f"The count is the coefficient of x^{n} in the product of 1/(1-x^k) for k = 1, 2, 3, ..."]
    if n <= 10:
        lines.append("They are: " + "; ".join(" + ".join(str(part) for part, times in sorted(group.items(), reverse=True) for _ in range(times)) for group in sp.utilities.iterables.partitions(n)))
    return Result("Partitions", [f"p({n}) = {count}"], lines)


def arrangements(counts, r):
    """Arrangements of r items from a multiset, by exponential generating function."""
    gf = sp.prod(sum(X ** k / sp.factorial(k) for k in range(c + 1)) for c in counts)
    value = sp.Poly(sp.expand(gf), X).coeff_monomial(X ** r) * sp.factorial(r)
    return Result("Arrangements from an exponential generating function", [f"{value} arrangements"], ["Each kind of item gives a bracket 1 + x + x²/2! + ... up to the number of that item", f"Multiply them: {show(sp.expand(gf))}", f"The coefficient of x^{r} is {show_plain(value / sp.factorial(r))}", f"Multiply by {r}! to count arrangements: {value}"])


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
    steps += _recurrence_lines(expression, function, n)
    if conditions:
        steps.append("Put in the starting values to fix the constants: " + ", ".join(f"{letter}({k.args[0]}) = {show_plain(v)}" for k, v in conditions.items()))
    else:
        steps.append("C0, C1 are constants fixed by the starting values")
    steps.append(f"So {letter}(n) = {show(sp.simplify(solution))}")
    return Result("Recurrence relation", [f"{letter}(n) = {show(sp.simplify(solution))}"], steps)


def _recurrence_lines(expression, function, n):
    """The characteristic equation of a linear recurrence with constant coefficients, and its roots."""
    terms = [t for t in expression.atoms(sp.Function) if t.func == function]
    shifts = {t: sp.simplify(t.args[0] - n) for t in terms}
    expanded = sp.expand(expression)
    coefficients = {shift: expanded.coeff(t) for t, shift in shifts.items()}
    rest = sp.simplify(expanded - sum(expanded.coeff(t) * t for t in terms))
    if not all(c.is_number for c in coefficients.values()) or not all(s.is_Integer for s in shifts.values()):
        return [f"The recurrence, with everything on one side: {show(expression)} = 0"]
    r = sp.Symbol("r")
    lowest = min(shifts.values())
    equation = sum(c * r ** (s - lowest) for s, c in coefficients.items())
    roots = sp.roots(sp.Poly(equation, r))
    lines = [f"The recurrence, with everything on one side: {show(expression)} = 0", f"Try {function.__name__}(n) = r^n, which gives the characteristic equation: {show(equation)} = 0",
             "Its roots: " + ", ".join(show_plain(root) + (f" (twice)" if times == 2 else f" ({times} times)" if times > 2 else "") for root, times in roots.items()),
             "Each root r gives a term C·r^n, and a repeated root also gives C·n·r^n"]
    if rest != 0:
        lines.append(f"The part without {function.__name__}, {show(rest)}, adds a particular solution of the same form as it")
    return lines


def growth(start, percent, periods):
    """A quantity that increases by a percentage each period."""
    factor = 1 + sp.Rational(percent) / 100
    value = start * factor ** periods
    return Result("Recurrence relation", [f"After {periods} periods: {show(value)}"], [f"A rise of {percent}% each period multiplies by 1 + {percent}/100 = {text_of(factor)}", f"a(n) = {text_of(factor)}·a(n-1), a(0) = {start}", f"Multiplying n times: a(n) = {start}·({text_of(factor)})^n", f"For n = {periods}: {start}·({text_of(factor)})^{periods} = {show(value)}"])


def hanoi(n=None):
    answer = ["Moves for n disks: a(n) = 2^n - 1"]
    if n is not None:
        answer.append(f"For {n} disks: {2 ** n - 1}")
    lines = ["To move n disks: move n-1 disks aside, move the largest, move the n-1 disks back", "So a(n) = 2·a(n-1) + 1, with a(1) = 1", "Values: " + ", ".join(f"a({k}) = {2 ** k - 1}" for k in range(1, 6)), "These are one less than powers of 2: a(n) = 2^n - 1"]
    if n is not None:
        lines.append(f"For n = {n}: 2^{n} - 1 = {2 ** n - 1}")
    return Result("Tower of Hanoi", answer, lines)


def handshakes(n=None):
    answer = ["Handshakes among n people: a(n) = n(n-1)/2"]
    if n is not None:
        answer.append(f"For {n} people: {n * (n - 1) // 2}")
    lines = ["A new person shakes hands with the n-1 already there: a(n) = a(n-1) + (n-1), with a(2) = 1", "Adding these up: a(n) = 1 + 2 + ... + (n-1) = n(n-1)/2"]
    if n is not None:
        lines.append(f"For n = {n}: {n} × {n - 1} ÷ 2 = {n * (n - 1) // 2}")
    return Result("Handshakes", answer, lines)
