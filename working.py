"""The working of plain maths, one step per line: sums, equations, limits, derivatives and integrals.

Every value is computed at the step that shows it, never typed in, and tests/test_working.py checks that the working of each
kind of question ends on the answer solver.py found on its own.
"""
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.integrals import manualintegrate as rules

from results import time_limited
from solver import FUNCTIONS, describe, tokenize

SIGNS = {"add": " + ", "sub": " − ", "mul": " × ", "div": " ÷ "}
PRIORITY = {"add": 1, "sub": 1, "mul": 2, "div": 2}  # the rest (powers, roots, factorials, functions) come first
ACTIONS = {"add": "Add", "sub": "Subtract", "mul": "Multiply", "div": "Divide"}
SHORT_LISTS = 8  # write out n × (n-1) × ... or a × a × ... only up to this many factors
STEPS_SECONDS = 10  # the answer is already found by then, so a slow explanation must not hold it up


def text_of(value):
    """A number or formula as people write it: 2x + 5, x^2, 7/3, 0.75, √(x), ln(x), e^(2x), π, ∞."""
    value = sp.sympify(value)
    if value.is_Rational:
        return _rational(value)
    text = sp.sstr(value)
    text = re.sub(r"\blog\(([^,()]+), (\w+)\)", r"log_\2(\1)", text)
    text = text.replace("**", "^").replace("log(", "ln(").replace("exp(", "e^(").replace("sqrt(", "√(").replace("pi", "π").replace("oo", "∞")
    text = re.sub(r"\bE\b", "e", text)
    text = re.sub(r"(?<=[\d)])\*(?=[A-Za-zπ√(])", "", text)  # 2*x, 2*(x+1), (x+1)*(x+2)
    text = re.sub(r"(?<=\b[a-z])\*(?=[a-z]\b)", "", text)    # x*y
    return text.replace("*", "·").replace(" - ", " − ")


def _rational(value):
    """A fraction whose decimal ends is written as that decimal: 3/4 is 0.75, while 7/3 stays 7/3."""
    if value.q == 1:
        return str(value.p)
    q = value.q
    for p in (2, 5):
        while q % p == 0:
            q //= p
    return str(float(value)) if q == 1 else f"{value.p}/{value.q}"


def _about(value):
    """' ≈ 2.33333' after a value that is not already written as a plain decimal."""
    value = sp.sympify(value)
    if not value.is_number or value in (sp.oo, -sp.oo) or (value.is_Rational and "/" not in _rational(value)):
        return ""
    return f" ≈ {float(sp.N(value)):.6g}"


class _NotPlain(Exception):
    """The line holds letters or something else that is not plain arithmetic."""


@dataclass
class _Node:
    kind: str                                  # num, neg, add, sub, mul, div, pow, root, fact, comb, perm or func
    args: list = field(default_factory=list)
    value: object = None                       # the number of a num, the name of a func
    base: object = None                        # the base of a log
    depth: int = 0                             # how many brackets it is written inside
    bracket: bool = False                      # written in brackets of its own
    fraction: bool = False                     # a num that came from a division, so it stays a fraction: 1/2, not 0.5


class _SumReader:
    """Keeps a sum in the order it was written, which SymPy's own expressions do not: 7 × 4, not 4 × 7."""

    def __init__(self, tokens):
        self.tokens, self.i, self.depth = tokens, 0, 0

    def peek(self, ahead=0):
        return self.tokens[self.i + ahead] if self.i + ahead < len(self.tokens) else (None, None)

    def take(self, kind=None):
        token = self.peek()
        if token[0] is None or (kind and token[0] != kind):
            raise _NotPlain
        self.i += 1
        return token

    def node(self, kind, *args, **extra):
        return _Node(kind, list(args), depth=self.depth, **extra)

    def sum(self):
        node = self.product()
        while self.peek() in (("op", "+"), ("op", "-")):
            node = self.node("add" if self.take()[1] == "+" else "sub", node, self.product())
        return node

    def product(self):
        node = self.unary()
        while True:
            token = self.peek()
            if token in (("op", "*"), ("op", "/")):
                self.take()
                node = self.node("mul" if token[1] == "*" else "div", node, self.unary())
            elif token[0] in ("num", "func", "const", "(") or token == ("name", "e"):  # 3(4+1), 2π
                node = self.node("mul", node, self.power())
            else:
                return node

    def unary(self):
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        if self.peek() == ("op", "-"):
            self.take()
            child = self.unary()
            if child.kind == "num":
                child.value = -child.value
                return child
            return self.node("neg", child)
        return self.power()

    def power(self):
        node = self.atom()
        while self.peek()[0] == "!":
            self.take()
            node = self.node("fact", node)
        if self.peek() == ("op", "^"):
            self.take()
            node = self.node("pow", node, self.unary())
        return node

    def atom(self):
        kind, text = self.take()
        if kind == "num":
            number = self.node("num", value=sp.Rational(text))
            if self.peek()[0] == "name" and self.peek()[1] in ("C", "c", "P", "p") and self.peek(1)[0] == "num":
                letter = self.take()[1]
                return self.node("comb" if letter in "Cc" else "perm", number, self.node("num", value=sp.Rational(self.take()[1])))
            return number
        if (kind, text) == ("const", "pi"):
            return self.node("num", value=sp.pi)
        if (kind, text) == ("name", "e"):
            return self.node("num", value=sp.E)
        if kind == "(":
            self.depth += 1
            inner = self.sum()
            self.take(")")
            self.depth -= 1
            inner.bracket = True
            return inner
        if kind == "func" and text != "beta":
            return self.function(text)
        raise _NotPlain

    def function(self, name):
        base = None
        if name == "log":
            base = sp.Integer(10)
            if self.peek()[0] == "_":
                self.take()
                base = self.atom().value
        if self.peek() == ("op", "^"):  # sin^2 x: left to the general way of working
            raise _NotPlain
        self.depth += 1
        if self.peek()[0] == "(":
            self.take()
            inner = self.sum()
            self.take(")")
        else:
            inner = self.power()
        self.depth -= 1
        inner.bracket = False
        return self.node("root", inner) if name == "sqrt" else self.node("func", inner, value=name, base=base)


def _write(node):
    """A sum as it stands now, with the brackets it was written in."""
    if node.kind == "num":
        return f"{node.value.p}/{node.value.q}" if node.fraction and node.value.is_Rational and node.value.q != 1 else text_of(node.value)
    if node.kind in SIGNS:
        right = node.args[1]
        written = _write(right)
        if right.kind == "num" and right.value.is_number and right.value < 0:
            written = f"({written})"
        text = _write(node.args[0]) + SIGNS[node.kind] + written
    elif node.kind == "pow":
        base, exponent = node.args
        written = _write(base)
        if base.kind == "num" and (base.value < 0 or not base.value.is_Integer) and not base.value.is_NumberSymbol:
            written = f"({written})"
        text = f"{written}^{_write(exponent)}"
    elif node.kind == "neg":
        text = "-" + _write(node.args[0])
    elif node.kind == "fact":
        text = _write(node.args[0]) + "!"
    elif node.kind in ("comb", "perm"):
        text = f"{_write(node.args[0])}{'C' if node.kind == 'comb' else 'P'}{_write(node.args[1])}"
    elif node.kind == "root":
        text = f"√({_write(node.args[0])})"
    else:
        name = "log" if node.value == "log" else node.value
        text = f"{name}_{text_of(node.base)}({_write(node.args[0])})" if node.value == "log" and node.base != 10 else f"{name}({_write(node.args[0])})"
    return f"({text})" if node.bracket and node.kind in (*SIGNS, "pow", "neg") else text


def _ready(node):
    """The parts of the sum that can be worked out now, in writing order: those whose arguments are already numbers."""
    if node.kind == "num":
        return []
    found = [part for arg in node.args for part in _ready(arg)]
    return found or ([node] if all(arg.kind == "num" for arg in node.args) else [])


def _next(root):
    """Brackets first, then powers, roots and functions, then × and ÷, then + and −, each from the left."""
    ready = _ready(root)
    return max(ready, key=lambda node: (node.depth, PRIORITY.get(node.kind, 3)))


def _fractions_step(kind, a, b):
    """The middle of a sum of written fractions: 1/2 + 1/3 = 3/6 + 2/6, or 2/3 × 3/4 = (2 × 3)/(3 × 4)."""
    if kind in ("add", "sub"):
        common = sp.ilcm(a.q, b.q)
        return f"{a * common}/{common}{SIGNS[kind]}{b * common}/{common}"
    if kind == "mul":
        return f"({a.p} × {b.p})/({a.q} × {b.q})"
    return f"{a} × {b.q}/{b.p}"


def _line_of_sum(node):
    """Work out a part whose arguments are numbers: (its value, the line that shows how)."""
    values = [arg.value for arg in node.args]
    written = _write(_Node(node.kind, node.args, value=node.value, base=node.base))
    if node.kind in SIGNS:
        a, b = values
        if node.kind == "div" and b == 0:
            raise _NotPlain
        value = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b if b != 0 else None}[node.kind]
        fractions = any(arg.fraction for arg in node.args) and a.is_Rational and b.is_Rational
        middle = f" = {_fractions_step(node.kind, a, b)}" if fractions else ""
        node.fraction = (fractions or node.kind == "div") and value.is_Rational and value.q != 1
        result = f"{value.p}/{value.q}" if node.fraction else text_of(value)
        ending = f" = {text_of(value)}" if node.fraction and "/" not in text_of(value) else _about(value)
        return value, f"{ACTIONS[node.kind]}: {written}{middle} = {result}{ending}"
    if node.kind == "neg":
        return -values[0], f"Change the sign: {written} = {text_of(-values[0])}"
    if node.kind == "pow":
        return _line_of_power(values[0], values[1], written)
    if node.kind == "root":
        return _line_of_root(values[0], written)
    if node.kind == "fact":
        return _line_of_factorial(values[0], written)
    if node.kind in ("comb", "perm"):
        return _line_of_counting(node.kind, values[0], values[1], written)
    return _line_of_function(node, values[0], written)


def _line_of_power(a, b, written):
    value = sp.nsimplify(a ** b) if a.is_Rational and b.is_Rational else a ** b
    if b.is_Integer and 1 < b <= SHORT_LISTS and a.is_Rational:
        repeated = " × ".join([text_of(a) if a >= 0 else f"({text_of(a)})"] * int(b))
        return value, f"Power: {written} = {repeated} = {text_of(value)}"
    if b == 0:
        return value, f"Power: {written} = 1, as any number to the power 0 is 1"
    if b.is_Integer and b < 0:
        return value, f"Power: {written} = 1 ÷ {text_of(a)}^{-b} = {text_of(value)}{_about(value)}"
    return value, f"Power: {written} = {text_of(value)}{_about(value)}"


def _line_of_root(a, written):
    value = sp.sqrt(a)
    if value.is_Rational:
        return value, f"Square root: {written} = {text_of(value)}, because {text_of(value)} × {text_of(value)} = {text_of(a)}"
    return value, f"Square root: {written} = {text_of(value)}{_about(value)}"


def _line_of_factorial(n, written):
    if not n.is_Integer or n < 0:
        raise _NotPlain
    value = sp.factorial(n)
    if 1 < n <= SHORT_LISTS + 4:
        return value, f"Factorial: {written} = {' × '.join(str(k) for k in range(n, 0, -1))} = {value}"
    return value, f"Factorial: {written} = {value}"


def _line_of_counting(kind, n, r, written):
    if not (n.is_Integer and r.is_Integer and 0 <= r <= n):
        raise _NotPlain
    falling = " × ".join(str(n - k) for k in range(r)) or "1"
    if kind == "perm":
        value = sp.ff(n, r)
        return value, f"Permutations: {written} = {n}! ÷ {n - r}! = {falling} = {value}" if r <= SHORT_LISTS else f"Permutations: {written} = {n}! ÷ {n - r}! = {value}"
    value = sp.binomial(n, r)
    if r <= SHORT_LISTS:
        below = " × ".join(str(r - k) for k in range(r)) or "1"
        return value, f"Combinations: {written} = {n}! ÷ ({r}! × {n - r}!) = ({falling}) ÷ ({below}) = {value}"
    return value, f"Combinations: {written} = {n}! ÷ ({r}! × {n - r}!) = {value}"


def _line_of_function(node, x, written):
    if node.value == "log":
        value = sp.log(x, node.base)
        if value.is_Rational:
            return value, f"Logarithm: {written} = {text_of(value)}, because {text_of(node.base)}^{text_of(value)} = {text_of(x)}"
        return value, f"Logarithm: {written} = ln({text_of(x)}) ÷ ln({text_of(node.base)}){_about(value)}"
    value = FUNCTIONS[node.value](x)
    if node.value == "ln" and value.is_Rational:
        return value, f"Natural log: {written} = {text_of(value)}, because e^{text_of(value)} = {text_of(x)}"
    angle = ", with the angle in radians" if node.value in ("sin", "cos", "tan", "cot", "sec", "cosec", "csc") else ""
    return value, f"{written} = {text_of(value)}{_about(value)}{angle}"


def sum_lines(text):
    """(the working of a line of plain arithmetic, its value). Raises _NotPlain if it holds anything else."""
    reader = _SumReader(tokenize(text))
    root = reader.sum()
    if reader.i < len(reader.tokens):
        raise _NotPlain
    if root.kind == "num":
        return [f"The value is {text_of(root.value)}{_about(root.value)}"], root.value
    root.bracket = False
    lines = [f"Work out: {_write(root)}"]
    while root.kind != "num":
        node = _next(root)
        value, line = _line_of_sum(node)
        node.kind, node.args, node.value = "num", [], value
        lines.append(line if root.kind == "num" else f"{line}, which leaves {_write(root)}")
    return lines, root.value


def _expression_lines(text, parsed, value):
    """Letters but no '=': any calculus inside it first, then multiply out, collect like terms, simplify."""
    lines = [f"Expression: {text}"]
    for part in sorted(parsed.atoms(sp.Derivative, sp.Limit, sp.Integral), key=str):
        lines += _calculus_lines(part, part.doit())
    worked = parsed.doit()
    expanded = sp.expand(worked)
    shown = [_bare(text)]
    if _bare(text_of(expanded)) != _bare(text_of(worked)):
        lines.append(f"Multiply out the brackets: {text_of(expanded)}")
        shown.append(_bare(text_of(expanded)))
    elif _bare(text_of(worked)) not in shown:
        lines.append(f"Collect like terms: {text_of(worked)}")
        shown.append(_bare(text_of(worked)))
    if _bare(text_of(value)) not in shown and _bare(text_of(value)) != _bare(text_of(worked)):
        lines.append(f"Simplify: {text_of(value)}")
    return lines


def _bare(text):
    """Writing without its spacing or brackets, to tell whether a step really changed anything."""
    return re.sub(r"[\s()]", "", text).replace("−", "-").replace("×", "*").replace("·", "*")


def _statement_lines(text, holds):
    """A statement with no unknown, such as 2+2=4: work out each side and compare."""
    lines = []
    values = []
    for name, side in zip(("Left side", "Right side"), text.split("=", 1)):
        working, value = sum_lines(side)
        lines += [f"{name}. {line}" for line in working]
        values.append(value)
    lines.append(f"The two sides are {text_of(values[0])} and {text_of(values[1])}, so the statement is {'true' if holds else 'false'}")
    return lines


def _moved(term):
    """'Subtract 5 from' or 'Add 5 to': what takes a term off the left side."""
    negative = term.could_extract_minus_sign()
    return f"Add {text_of(-term)} to" if negative else f"Subtract {text_of(term)} from"


def _linear_lines(left, right, var):
    """a·x + b = c·x + d, one move at a time."""
    lines = []
    expanded_left, expanded_right = sp.expand(left), sp.expand(right)
    if expanded_left != left or expanded_right != right:
        lines.append(f"Multiply out the brackets: {text_of(expanded_left)} = {text_of(expanded_right)}")
    a, b = expanded_left.coeff(var, 1), expanded_left.coeff(var, 0)
    c, d = expanded_right.coeff(var, 1), expanded_right.coeff(var, 0)
    if c != 0:
        a -= c
        lines.append(f"{_moved(c * var)} both sides: {text_of(a * var + b)} = {text_of(d)}")
    if b != 0:
        d -= b
        lines.append(f"{_moved(b)} both sides: {text_of(a * var)} = {text_of(d)}")
    if a != 1:
        word = "Multiply both sides by -1" if a == -1 else f"Divide both sides by {text_of(a)}"
        lines.append(f"{word}: {var} = {text_of(sp.simplify(d / a))}{_about(sp.simplify(d / a))}")
    return lines


def _quadratic_lines(difference, var, right):
    """a·x² + b·x + c = 0: factors if it has them, the discriminant, the formula."""
    a, b, c = sp.Poly(difference, var).all_coeffs()
    if not all(k.is_number for k in (a, b, c)):
        raise _NotPlain
    lines = [f"Move everything to one side: {text_of(difference)} = 0"] if right != 0 and b != 0 else []
    if b == 0:
        square = -c / a
        if a != 1 or right == 0:
            lines.append(f"Get the square on its own: {var}^2 = {text_of(square)}")
        if square < 0:
            lines.append("A square cannot be negative, so there is no real solution")
        elif square == 0:
            lines.append(f"So {var} = 0")
        else:
            lines.append(f"Take the square root of both sides: {var} = ±√({text_of(square)}) = ±{text_of(sp.sqrt(square))}")
        return lines
    lines.append(f"This is a quadratic with a = {text_of(a)}, b = {text_of(b)} and c = {text_of(c)}")
    factored = sp.factor(difference)
    if factored.is_Mul or factored.is_Pow:
        pieces = [piece for piece, _ in sp.factor_list(factored)[1]]
        lines.append(f"Factor it: {text_of(factored)} = 0")
        lines.append("A product is 0 only when one factor is 0: " + " or ".join(f"{text_of(piece)} = 0" for piece in pieces))
    discriminant = b ** 2 - 4 * a * c
    lines.append(f"Discriminant: b^2 − 4ac = ({text_of(b)})^2 − 4({text_of(a)})({text_of(c)}) = {text_of(discriminant)}")
    if discriminant < 0:
        lines.append("The discriminant is negative, so there is no real solution")
    elif discriminant == 0:
        lines.append(f"The discriminant is 0, so there is one repeated root: {var} = -b ÷ 2a = {text_of(-b / (2 * a))}")
    else:
        root = sp.sqrt(discriminant)
        lines.append(f"Quadratic formula: {var} = (-b ± √D) ÷ 2a = ({text_of(-b)} ± {text_of(root)}) ÷ {text_of(2 * a)}")
        for sign, word in ((1, "+"), (-1, "−")):
            value = sp.simplify((-b + sign * root) / (2 * a))
            lines.append(f"With {word}: {var} = {text_of(value)}{_about(value)}")
    return lines


def _higher_lines(difference, var, right):
    lines = [f"Move everything to one side: {text_of(difference)} = 0"] if right != 0 else []
    factored = sp.factor(difference)
    pieces = [piece for piece, _ in sp.factor_list(factored)[1]] if factored.is_Mul or factored.is_Pow else []
    if pieces:
        lines.append(f"Factor it: {text_of(factored)} = 0")
        for piece in pieces:
            roots = sp.solve(piece, var)
            lines.append(f"{text_of(piece)} = 0 gives " + (" or ".join(f"{var} = {text_of(r)}" for r in roots) if roots else "no value"))
    else:
        lines.append("It does not break into simpler factors, so its roots are found directly")
    return lines


def _undo_lines(left, right, var):
    """Undo what was done to the unknown, outermost first, doing the same to both sides. None if that cannot finish."""
    lines = []
    if right.has(var):
        if left.has(var):
            left, right = left - right, sp.Integer(0)
            lines.append(f"Move everything to one side: {text_of(left)} = 0")
        else:
            left, right = right, left
    for _ in range(12):
        if left == var:
            return lines
        if left.is_Add:
            rest, part = left.as_independent(var, as_Add=True)
            if rest == 0:
                return None
            word, left, right = _moved(rest), part, sp.simplify(right - rest)
            lines.append(f"{word} both sides: {text_of(left)} = {text_of(right)}")
        elif left.is_Mul:
            factor, part = left.as_independent(var, as_Add=False)
            if factor == 1:
                return None
            left, right = part, sp.simplify(right / factor)
            lines.append(f"Divide both sides by {text_of(factor)}: {text_of(left)} = {text_of(right)}")
        elif isinstance(left, sp.exp):
            left, right = left.exp, sp.log(right)
            lines.append(f"Take ln of both sides: {text_of(left)} = {text_of(right)}")
        elif left.is_Pow and not left.base.has(var):
            base = left.base
            left, right = left.exp, sp.simplify(sp.log(right, base))
            lines.append(f"Ask what power of {text_of(base)} gives the right side: {text_of(left)} = {text_of(right)}")
        elif left.is_Pow and left.exp.is_Rational and not left.exp.p % 2 == 0:
            power = left.exp
            word = "Square both sides" if power == sp.Rational(1, 2) else "Turn both sides upside down" if power == -1 else f"Raise both sides to the power {text_of(1 / power)}"
            left, right = left.base, sp.simplify(right ** (1 / power))
            lines.append(f"{word}: {text_of(left)} = {text_of(right)}")
        elif isinstance(left, sp.log):
            left, right = left.args[0], sp.exp(right)
            lines.append(f"Raise e to both sides, which undoes ln: {text_of(left)} = {text_of(right)}")
        else:
            return None
    return None


def _general_lines(left, right, var):
    """When no simple move is left: one side, over one denominator, in factors."""
    difference = sp.together(left - right)
    lines = [f"Move everything to one side: {text_of(difference)} = 0"]
    top, bottom = sp.fraction(difference)
    if bottom.has(var):
        lines.append(f"A fraction is 0 only when its top is 0: {text_of(top)} = 0")
    factored = sp.factor(top)
    if factored.is_Mul and text_of(factored) != text_of(top):
        lines.append(f"Factor it: {text_of(factored)} = 0")
    return lines


def _check_lines(left, right, var, solutions):
    """Put each solution back into the equation."""
    if not isinstance(solutions, sp.FiniteSet) or len(solutions) > 4:
        return []
    lines = []
    for value in solutions:
        at_left, at_right = sp.simplify(left.subs(var, value)), sp.simplify(right.subs(var, value))
        if sp.simplify(at_left - at_right) == 0:
            lines.append(f"Check {var} = {text_of(value)}: the left side is {text_of(at_left)} and the right side is {text_of(at_right)} ✓")
    return lines


def _equation_lines(text, answer):
    var = answer.variable
    written_left, written_right = answer.parsed.lhs, answer.parsed.rhs
    left, right = written_left.doit(), written_right.doit()
    lines = [f"Equation: {text}"]
    worked = f"{text_of(left)} = {text_of(right)}"
    if _bare(worked) != _bare(text) and not written_left.has(sp.log):
        lines.append(f"Work out each side: {worked}")
    difference = sp.expand(left - right)
    if not difference.has(var):
        lines.append(f"The {var} terms cancel" + (", and both sides are the same, so it is true for every value" if difference == 0
                                                  else f", leaving {text_of(difference)} = 0, which is never true"))
        return lines
    if difference.is_polynomial(var):
        degree = sp.degree(difference, var)
        if degree == 1:
            lines += _linear_lines(left, right, var)
        elif degree == 2:
            lines += _quadratic_lines(difference, var, right)
        else:
            lines += _higher_lines(difference, var, right)
    else:
        if isinstance(written_left, sp.log) and len(written_left.args) == 2 and not written_right.has(var):  # log_b(u) = c means u = b^c
            base = written_left.args[1]
            left_inner, value = written_left.args[0].doit(), base ** right
            lines.append(f"A log of base {text_of(base)} undoes a power of {text_of(base)}: {text_of(left_inner)} = {text_of(base)}^{text_of(right)} = {text_of(value)}")
            undone = _undo_lines(left_inner, value, var)
        else:
            undone = _undo_lines(left, right, var)
        lines += undone if undone is not None else _general_lines(left, right, var)
    if isinstance(answer.solutions, sp.FiniteSet) and len(answer.solutions) > 0:
        lines += _check_lines(left, right, var, answer.solutions)
    return lines


def _at(expression, var, target):
    """The value at a point, or None when putting it in gives no number."""
    value = expression.subs(var, target)
    return None if value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) else sp.simplify(value)


def _limit_at_point(function, var, target):
    """(lines, value); value None when the lines do not reach an answer."""
    top, bottom = sp.fraction(sp.together(function))
    at_top, at_bottom = _at(top, var, target), _at(bottom, var, target)
    put = f"Put {var} = {text_of(target)}"
    if at_top is None or at_bottom is None:
        return [f"{put}: this gives no number directly"], None
    if bottom == 1:
        return [f"{put} into the expression: {text_of(at_top)}"], at_top
    lines = [f"{put}: the top is {text_of(at_top)} and the bottom is {text_of(at_bottom)}"]
    if at_bottom != 0:
        value = sp.simplify(at_top / at_bottom)
        lines.append(f"Nothing goes wrong, so the limit is {text_of(at_top)} ÷ {text_of(at_bottom)} = {text_of(value)}")
        return lines, value
    if at_top != 0:
        lines.append("The bottom tends to 0 while the top does not, so the expression grows without limit")
        return lines, None
    lines.append("That is 0/0, which has no value yet, so the expression has to be changed first")
    reduced = sp.cancel(function)
    new_top, new_bottom = sp.fraction(reduced)
    if text_of(reduced) != text_of(function) and _at(new_bottom, var, target) not in (None, 0):
        common = sp.gcd(top, bottom)
        lines.append(f"Factor the top and the bottom: ({text_of(sp.factor(top))}) ÷ ({text_of(sp.factor(bottom))})")
        lines.append(f"Cancel the common factor {text_of(common)}: {text_of(reduced)}")
        value = _at(reduced, var, target)
        lines.append(f"{put} now: {text_of(value)}")
        return lines, value
    for _ in range(4):
        top, bottom = sp.diff(top, var), sp.diff(bottom, var)
        at_top, at_bottom = _at(top, var, target), _at(bottom, var, target)
        lines.append(f"L'Hôpital's rule: differentiate the top and the bottom separately: ({text_of(top)}) ÷ ({text_of(bottom)})")
        if at_top is None or at_bottom is None:
            return lines, None
        lines.append(f"{put}: the top is {text_of(at_top)} and the bottom is {text_of(at_bottom)}")
        if at_bottom != 0:
            return lines, sp.simplify(at_top / at_bottom)
        if at_top != 0:
            return lines, None
    return lines, None


def _limit_at_infinity(function, var, target):
    top, bottom = sp.fraction(sp.together(function))
    if top.is_polynomial(var) and bottom.is_polynomial(var) and bottom.has(var):
        high_top, high_bottom = sp.degree(top, var), sp.degree(bottom, var)
        lines = [f"The highest power is {var}^{high_top} on top and {var}^{high_bottom} below",
                 f"Divide the top and the bottom by {var}^{high_bottom}: every term with {var} below a number tends to 0"]
        if high_top < high_bottom:
            lines.append("The bottom grows faster than the top, so the expression tends to 0")
            return lines, sp.Integer(0)
        if high_top == high_bottom:
            value = sp.LC(top, var) / sp.LC(bottom, var)
            lines.append(f"Only the leading numbers stay: {text_of(sp.LC(top, var))} ÷ {text_of(sp.LC(bottom, var))} = {text_of(value)}")
            return lines, value
        lines.append("The top grows faster than the bottom, so the expression grows without limit")
        return lines, None
    limit_top, limit_bottom = sp.limit(top, var, target), sp.limit(bottom, var, target)
    lines = [f"As {var} → {text_of(target)}, the top tends to {text_of(limit_top)} and the bottom tends to {text_of(limit_bottom)}"]
    for _ in range(4):
        if not ((limit_top.is_infinite and limit_bottom.is_infinite) or (limit_top == 0 and limit_bottom == 0)):
            break
        top, bottom = sp.diff(top, var), sp.diff(bottom, var)
        limit_top, limit_bottom = sp.limit(top, var, target), sp.limit(bottom, var, target)
        lines.append(f"L'Hôpital's rule: differentiate both: ({text_of(top)}) ÷ ({text_of(bottom)}), which tend to {text_of(limit_top)} and {text_of(limit_bottom)}")
    if limit_bottom.is_finite and limit_bottom != 0 and limit_top.is_finite:
        return lines, sp.simplify(limit_top / limit_bottom)
    if limit_bottom.is_infinite and limit_top.is_finite:
        return lines, sp.Integer(0)
    return lines, None


def _limit_lines(limit, value):
    function, var, target = limit.args[0].doit(), limit.args[1], limit.args[2]
    lines = [f"Find the limit of {text_of(function)} as {var} → {text_of(target)}"]
    more, reached = _limit_at_infinity(function, var, target) if target.is_infinite else _limit_at_point(function, var, target)
    lines += more
    if reached is None or sp.simplify(reached - value) != 0:
        lines.append(f"Working the limit out exactly gives {text_of(value)}")
    lines.append(f"So the limit is {text_of(value)}{_about(value)}")
    return lines


def _rule_of_term(term, var):
    """A line for the derivative of one term, naming the rule used."""
    result = sp.diff(term, var)
    shown = f"d/d{var}({text_of(term)}) = {text_of(result)}"
    if not term.has(var):
        return f"{shown}: a constant does not change"
    constant, rest = term.as_independent(var, as_Add=False)
    prefix = f"keep the constant {text_of(constant)} and apply " if constant != 1 else ""
    if rest == var:
        return f"{shown}: the derivative of {var} is 1"
    if rest.is_Pow and rest.base == var and not rest.exp.has(var):
        return f"{shown}: {prefix}the power rule, bring the power {text_of(rest.exp)} down and lower it by 1"
    if rest.is_Pow and not rest.base.has(var):
        return f"{shown}: {prefix}the rule for a number to a power, a^u gives a^u · ln(a) · u'"
    if rest.is_Mul:
        factors = [factor for factor in rest.args if factor.has(var)]
        below = [factor for factor in factors if factor.is_Pow and factor.exp.is_negative]
        if below and len(factors) >= 2:
            u, v = sp.Mul(*[f for f in factors if f not in below]), 1 / sp.Mul(*below)
            return (f"{shown}: {prefix}the quotient rule, (u'v − uv') ÷ v^2, with u = {text_of(u)}, u' = {text_of(sp.diff(u, var))}, "
                    f"v = {text_of(v)} and v' = {text_of(sp.diff(v, var))}")
        u, v = factors[0], sp.Mul(*factors[1:])
        return (f"{shown}: {prefix}the product rule, u'v + uv', with u = {text_of(u)}, u' = {text_of(sp.diff(u, var))}, "
                f"v = {text_of(v)} and v' = {text_of(sp.diff(v, var))}")
    inner = rest.base if rest.is_Pow else rest.args[0] if rest.args else var
    if inner != var:
        return f"{shown}: {prefix}the chain rule, derivative of the outside times derivative of the inside, {text_of(inner)}, which is {text_of(sp.diff(inner, var))}"
    return f"{shown}: {prefix}the standard derivative of this function"


def _derivative_lines(derivative, value):
    function = derivative.expr.doit()
    var, order = derivative.variable_count[0]
    lines = [f"Find the derivative of {text_of(function)} with respect to {var}" + (f", {order} times" if order > 1 else "")]
    for number in range(1, order + 1):
        if order > 1:
            lines.append(f"Derivative number {number}, of {text_of(function)}")
        if function.is_Add:
            lines.append("Differentiate each term on its own")
            lines += [_rule_of_term(term, var) for term in sp.Add.make_args(function)]
            lines.append(f"Add the results: {text_of(sp.diff(function, var))}")
        else:
            lines.append(_rule_of_term(function, var))
        function = sp.diff(function, var)
    simplified = sp.simplify(function)
    if text_of(simplified) != text_of(function):
        lines.append(f"Simplify: {text_of(simplified)}")
    return lines


def _rule_lines(rule, var_name):
    """The steps of one of SymPy's integration rules, in words."""
    name = type(rule).__name__
    if isinstance(rule, rules.AlternativeRule):
        return _rule_lines(rule.alternatives[0], var_name)
    shown = f"∫ {text_of(rule.integrand)} d{rule.variable} = {text_of(rule.eval())}"
    if isinstance(rule, rules.DontKnowRule):
        raise _NotPlain
    if isinstance(rule, rules.AddRule):
        lines = ["Integrate each term on its own"]
        for step in rule.substeps:
            lines += _rule_lines(step, var_name)
        return lines + [f"Add the results: {text_of(rule.eval())}"]
    if isinstance(rule, rules.ConstantTimesRule):
        return [f"Keep the constant {text_of(rule.constant)} outside: {text_of(rule.constant)} × ∫ {text_of(rule.other)} d{rule.variable}",
                *_rule_lines(rule.substep, var_name), f"Multiply back by {text_of(rule.constant)}: {text_of(rule.eval())}"]
    if isinstance(rule, rules.PowerRule):
        return [f"{shown}, by the power rule: add 1 to the power and divide by the new power"]
    if isinstance(rule, rules.ConstantRule):
        return [f"{shown}: the integral of a constant"]
    if isinstance(rule, rules.ReciprocalRule):
        return [f"{shown}: the integral of 1/{text_of(rule.base)} is its natural log"]
    if isinstance(rule, rules.URule):
        return [f"Substitute {rule.u_var} = {text_of(rule.u_func)}, so d{rule.u_var} = {text_of(sp.diff(rule.u_func, rule.variable))} d{rule.variable}",
                *_rule_lines(rule.substep, var_name), f"Put {text_of(rule.u_func)} back in place of {rule.u_var}: {text_of(rule.eval())}"]
    if isinstance(rule, rules.PartsRule):
        return [f"Integrate by parts, ∫ u dv = uv − ∫ v du, with u = {text_of(rule.u)} and dv = {text_of(rule.dv)} d{rule.variable}",
                *_rule_lines(rule.v_step, var_name),
                *(_rule_lines(rule.second_step, var_name) if rule.second_step else []),
                f"Put it together: {text_of(rule.eval())}"]
    if isinstance(rule, rules.RewriteRule):
        return [f"Rewrite the integrand: {text_of(rule.rewritten)}", *_rule_lines(rule.substep, var_name)]
    if isinstance(rule, rules.AtomicRule):
        return [f"{shown}: a standard integral"]
    return [f"{shown}, using {name.removesuffix('Rule').lower()}"]


def integral_lines(integral, value):
    """The working of one integral: the rules used, then the limits put in."""
    function = integral.function.doit()
    limits = integral.limits[0]
    var = limits[0]
    if len(limits) == 3:
        lines = [f"Find ∫ {text_of(function)} d{var} from {text_of(limits[1])} to {text_of(limits[2])}"]
    else:
        lines = [f"Find ∫ {text_of(function)} d{var}"]
    rule = rules.integral_steps(function, var)
    letter = next(name for name in ("t", "u", "w", "z") if sp.Symbol(name) not in function.free_symbols and sp.Symbol(name) != var)
    lines += [line.replace("_u", letter) for line in _rule_lines(rule, str(var))]
    anti = rule.eval()
    if len(limits) == 3:
        low, high = limits[1], limits[2]
        lines.append(f"So F({var}) = {text_of(anti)}. Now put in the limits: F({text_of(high)}) − F({text_of(low)})")
        if not (low.is_infinite or high.is_infinite):
            at_high, at_low = sp.simplify(anti.subs(var, high)), sp.simplify(anti.subs(var, low))
            lines.append(f"F({text_of(high)}) = {text_of(at_high)} and F({text_of(low)}) = {text_of(at_low)}")
            lines.append(f"{text_of(at_high)} − {text_of(at_low)} = {text_of(value)}{_about(value)}")
        else:
            lines.append(f"Taking the limit at the infinite end gives {text_of(value)}{_about(value)}")
    else:
        lines.append("Add the constant of integration, C, to any indefinite integral")
    return lines


def _calculus_lines(part, value):
    if isinstance(part, sp.Limit):
        return _limit_lines(part, value)
    if isinstance(part, sp.Derivative):
        return _derivative_lines(part, value)
    return integral_lines(part, value)


def _value_lines(text, answer):
    parsed, value = answer.parsed, answer.value
    if isinstance(parsed, (sp.Limit, sp.Derivative, sp.Integral)):
        return _calculus_lines(parsed, value)
    if parsed.free_symbols or parsed.has(sp.Derivative, sp.Limit, sp.Integral):
        return _expression_lines(text, parsed, value)
    lines, reached = sum_lines(text)
    if sp.simplify(reached - value) != 0:
        raise _NotPlain
    return lines


def _lines_for(text, answer):
    if answer.value is not None:
        return _value_lines(text, answer)
    if answer.holds is not None:
        return _statement_lines(text, answer.holds)
    return _equation_lines(text, answer)


def _plain_fallback(answer):
    """When there is no way to spell out the steps: at least what was read and what it came to."""
    return [f"Read as: {text_of(answer.parsed)}", f"Worked out exactly: {describe(answer)}"]


def _beta_lines(function, var, low, high):
    """sin^m cos^n from 0 to π/2, and x^(a-1)/(1+x^b) from 0 to ∞, which the course solves with Beta and Gamma functions."""
    if (low, high) == (0, sp.pi / 2):
        m, n = sp.Wild("m", exclude=[var]), sp.Wild("n", exclude=[var])
        found = function.match(sp.sin(var) ** m * sp.cos(var) ** n)
        if found and found[m].is_number and found[n].is_number:
            p, q = (found[m] + 1) / 2, (found[n] + 1) / 2
            return [f"With m = {text_of(found[m])} and n = {text_of(found[n])}, this is a Beta function: ∫ sin^m cos^n from 0 to π/2 = ½·B((m+1)/2, (n+1)/2)",
                    f"½·B({text_of(p)}, {text_of(q)}) = Γ({text_of(p)})Γ({text_of(q)}) ÷ (2Γ({text_of(p + q)}))",
                    f"= {text_of(sp.gamma(p))} × {text_of(sp.gamma(q))} ÷ (2 × {text_of(sp.gamma(p + q))})"]
    if (low, high) == (0, sp.oo):
        top, bottom = sp.fraction(function)
        b = sp.Wild("b", exclude=[var])
        found = bottom.match(1 + var ** b)
        power = sp.degree(top, var) if top.is_polynomial(var) else None
        if found and found[b].is_number and power is not None:
            a = power + 1
            return [f"Put t = {var}^{text_of(found[b])}, which turns it into a Beta function: ∫ x^(a-1) ÷ (1 + x^b) dx from 0 to ∞ = (1/b)·B(a/b, 1 - a/b), with a = {a} and b = {text_of(found[b])}",
                    f"B(p, 1 - p) = Γ(p)Γ(1 - p) = π ÷ sin(pπ), with p = {text_of(sp.Rational(a) / found[b])}",
                    f"So the integral is π ÷ ({text_of(found[b])} × sin({text_of(sp.Rational(a) / found[b])}π))"]
    return []


def _integral_without_rules(expression, value):
    """No chain of rules: the Beta form or the antiderivative, which are as much of the working as can be spelled out."""
    if not isinstance(expression, sp.Integral) or len(expression.limits[0]) != 3:
        return [f"The expression: {text_of(expression)}", f"Worked out exactly: {text_of(value)}{_about(value)}"]
    function, (var, low, high) = expression.function.doit(), expression.limits[0]
    lines = [f"Find ∫ {text_of(function)} d{var} from {text_of(low)} to {text_of(high)}"]
    more = _beta_lines(function, var, low, high)
    if not more:
        try:
            anti = time_limited(lambda: sp.integrate(function, var), STEPS_SECONDS)
            if not anti.has(sp.Integral):
                more = [f"Its antiderivative: F({var}) = {text_of(anti)}", f"Put in the limits: F({text_of(high)}) − F({text_of(low)})"]
        except Exception:  # the answer is already found, and this only explains it
            pass
    return lines + more + [f"That comes to {text_of(value)}{_about(value)}"]


def explain_integral(expression, value):
    """The working of an integral, or of each integral in an expression."""
    def work():
        if isinstance(expression, sp.Integral):
            return integral_lines(expression, value)
        lines = [line for part in sorted(expression.atoms(sp.Integral), key=str) for line in integral_lines(part, part.doit())]
        return lines + [f"Put it together: {text_of(value)}"]

    try:
        return time_limited(work, STEPS_SECONDS)
    except Exception:  # TimeoutError, _NotPlain or anything SymPy raises on an unusual integral
        return _integral_without_rules(expression, value)


def explain(text, answer):
    """The working of one line that solver.solve answered. The answer stands whatever happens here, so nothing may break it."""
    text = text.strip().rstrip("=")
    try:
        return time_limited(lambda: _lines_for(text, answer), STEPS_SECONDS)
    except Exception:  # TimeoutError, _NotPlain or anything SymPy raises on an unusual expression
        return _plain_fallback(answer)
