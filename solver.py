"""Read the AI's linear text (12+7×3, log_2(8), lim_(x→0) sin(x)/x, 2x+5=7) and solve it with SymPy.

Parsing is our own small recursive-descent parser (no eval). It builds the SymPy expression unevaluated,
so the app can show exactly what was read, and then works the answer out exactly.
"""
import re
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

X = sp.Symbol("x")
ALIASES = {"−": "-", "–": "-", "×": "*", "·": "*", "÷": "/", "X": "x", "∞": "oo", "π": "pi", "→": "->", "√": "sqrt ",
           "∫": "int ", "∂": "partial ", "’": "'", "′": "'", "″": "''"}
WORDS = ("lim", "cosec", "cosh", "sinh", "tanh", "cos", "sin", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
         "log", "ln", "exp", "sqrt", "abs", "beta", "gamma", "pi", "oo", "inf", "int", "partial")
TOKEN_RE = re.compile(r"\s*(?:([\d.]+)|(" + "|".join(sorted(WORDS, key=len, reverse=True))
                      + r")|(->)|([A-Za-zθλμφσα])(\d*)|(\S))")
NUMBER_RE = re.compile(r"\d+(\.\d*)?|\.\d+")
FUNCTIONS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "cosec": sp.csc, "csc": sp.csc,
             "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
             "ln": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs, "gamma": sp.gamma}
INVERSES = {"sin": sp.asin, "cos": sp.acos, "tan": sp.atan}
FUNC_WORDS = {"log", "beta", *FUNCTIONS}
STARTS_FACTOR = {"num", "name", "func", "const", "(", "int", "partial"}  # tokens that start an implicit multiplication
FRACTION_D = re.compile(r"\((d|partial )(\^\d+)?\s?([a-zθ]?)\)/\((?:d|partial )\s?([a-zθ])(\^\d+)?\)")


def tokenize(text, plain=True):
    """'2x+5=7' -> [('num', '2'), ('name', 'x'), ('op', '+'), ...]. plain=False keeps every x as a letter."""
    for old, new in ALIASES.items():
        text = text.replace(old, new)
    text = FRACTION_D.sub(lambda m: f"{m[1]}{m[2] or ''}{m[3]}/{m[1]}{m[4]}{m[5] or ''}", text)  # the reader's (dy)/(dx)
    has_equals = "=" in text
    tokens = []
    for number, word, arrow, letter, index, other in TOKEN_RE.findall(text):
        if number:
            if not NUMBER_RE.fullmatch(number):
                raise ValueError(f"'{number}' is not a number")
            tokens.append(("num", number))
        elif word:
            kind = ("lim" if word == "lim" else "int" if word == "int" else "partial" if word == "partial"
                    else "func" if word in FUNC_WORDS else "const")
            tokens.append((kind, word))
        elif arrow:
            tokens.append(("arrow", "->"))
        elif letter:
            after_factor = bool(tokens) and tokens[-1][0] in ("num", ")")
            if index and letter != "e" and (has_equals or not after_factor):  # x1, x2: one symbol with a number in its name
                tokens.append(("name", letter + index))
            else:
                tokens.append(("name", letter))
                if index:
                    tokens.append(("num", index))
        elif other in "+-*/^=()_!',":
            tokens.append(("op" if other in "+-*/^=" else other, other))
        else:
            raise ValueError(f"Unknown symbol '{other}'")
    if plain and not any(t[0] == "lim" for t in tokens) and ("op", "=") not in tokens:
        # Plain sums write times as x too: an x between two numbers means ×, not the variable.
        for i in range(1, len(tokens) - 1):
            if (tokens[i] == ("name", "x") and tokens[i - 1][0] in ("num", ")")
                    and tokens[i + 1][0] in ("num", "(")):
                tokens[i] = ("op", "*")
    return tokens


class _Parser:
    def __init__(self, tokens, natural_log=False):
        self.tokens, self.i = tokens, 0
        self.natural_log = natural_log  # in calculus 'log' means the natural log
        self.in_integral = 0

    def peek(self, ahead=0):
        j = self.i + ahead
        return self.tokens[j] if j < len(self.tokens) else (None, None)

    def take(self, kind=None):
        tok = self.peek()
        if tok[0] is None or (kind and tok[0] != kind):
            raise ValueError(f"Expected {kind or 'more'} {'here' if tok[0] else 'at the end'}")
        self.i += 1
        return tok

    def at_differential(self):
        """The 'dx' that ends an integral."""
        return self.in_integral and self.peek() == ("name", "d") and self.peek(1)[0] == "name"

    def sum(self):
        value = self.product()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            right = self.product()
            value = sp.Add(value, right if op == "+" else -right)
        return value

    def product(self):
        value = self.unary()
        while True:
            kind, op = self.peek()
            if self.at_differential():
                return value
            if (kind, op) in (("op", "*"), ("op", "/")):
                self.take()
                right = self.unary()
                if op == "/" and right.is_Number and right == 0:
                    raise ZeroDivisionError("Cannot divide by zero")
                value = sp.Mul(value, right if op == "*" else sp.Pow(right, -1))
            elif kind in STARTS_FACTOR:  # 2x, 3(4+1), x sin(x)
                value = sp.Mul(value, self.power())
            else:
                return value

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return -self.unary()
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        while self.peek()[0] == "!":
            self.take()
            base = sp.factorial(base)
        if self.peek() == ("op", "^"):
            self.take()
            exponent = self.unary()
            if base == sp.E and exponent.is_number and self.peek()[0] == "name" and not self.at_differential():  # e^2x
                exponent = sp.Mul(exponent, self.atom())
            return sp.Pow(base, exponent)
        return base

    def atom(self):
        kind, text = self.take()
        if kind == "num":
            number = sp.Rational(text)
            if self.peek()[0] == "name" and self.peek()[1] in ("C", "c", "P", "p") and self.peek(1)[0] == "num":  # 23C4, 5P2
                letter = self.take()[1]
                r = sp.Rational(self.take("num")[1])
                return sp.binomial(number, r) if letter in "Cc" else sp.ff(number, r)
            return number
        if kind == "name":
            return self.name(text)
        if kind == "const":
            return sp.pi if text == "pi" else sp.oo
        if kind == "(":
            inner = self.sum()
            self.take(")")
            return inner
        if kind == "func":
            return self.function(text)
        if kind == "lim":
            return self.limit()
        if kind == "int":
            return self.integral()
        if kind == "partial":
            return self.derivative(text)
        raise ValueError(f"'{text}' needs a number before it")

    def name(self, text):
        if text == "e":
            return sp.E
        if text == "d" and (self.peek() == ("op", "/") or self.is_derivative_ahead()):
            return self.derivative(text)
        if text in ("C", "P") and self.peek()[0] == "(":  # C(23,4)
            self.take()
            n = self.sum()
            self.take(",")
            r = self.sum()
            self.take(")")
            return sp.binomial(n, r) if text == "C" else sp.ff(n, r)
        if self.peek()[0] == "_" and self.peek(1)[0] in ("num", "name"):  # x_1
            self.take()
            return sp.Symbol(text + self.take()[1])
        primes = 0
        while self.peek()[0] == "'":
            self.take()
            primes += 1
        if primes:
            return sp.Derivative(sp.Function(text)(X), (X, primes))
        return sp.Symbol(text)

    def is_derivative_ahead(self):
        """d^2/dx^2, dy/dx, d^2y/dx^2 (not d times y)."""
        j = 0
        if self.peek(j) == ("op", "^") and self.peek(j + 1)[0] == "num":
            j += 2
        if self.peek(j)[0] == "name":
            j += 1
        return self.peek(j) == ("op", "/") and self.peek(j + 1) in (("name", "d"), ("partial", "partial"))

    def derivative(self, word):
        """After a 'd' or '∂': d/dx f, d^2/dx^2 f, dy/dx, ∂^2z/∂x^2. Returns the unevaluated derivative."""
        order = 1
        if self.peek() == ("op", "^"):
            self.take()
            order = int(self.take("num")[1])
        target = self.take()[1] if self.peek()[0] == "name" else None
        if self.take("op")[1] != "/":
            raise ValueError("A derivative is written d/dx")
        self.take()  # the d or ∂ below the line
        var = sp.Symbol(self.take("name")[1])
        if self.peek() == ("op", "^"):
            self.take()
            self.take("num")
        if target is not None:
            if word == "d":  # y as an unknown function: dy/dx in a differential equation
                return sp.Derivative(sp.Function(target)(var), (var, order))
            return sp.Derivative(sp.Symbol(target), (var, order))
        return sp.Derivative(self.unary_product(), (var, order))

    def unary_product(self):
        """What d/dx applies to: the next group, up to a + or -."""
        return self.product()

    def function(self, text):
        base = None
        if text == "beta":
            self.take("(")
            a = self.sum()
            self.take(",")
            b = self.sum()
            self.take(")")
            return sp.beta(a, b)
        if text == "log" and self.peek()[0] == "_":
            self.take()
            base = self.atom()
        power = None
        if self.peek() == ("op", "^"):  # sin^4 x, sin^-1 x
            self.take()
            power = self.unary()
        arg = self.argument()
        if power == -1 and text in INVERSES:
            return INVERSES[text](arg)
        if text == "log":
            value = sp.log(arg, base if base is not None else (sp.E if self.natural_log else 10))
        else:
            value = FUNCTIONS[text](arg)
        return value if power is None else sp.Pow(value, power)

    def argument(self):
        """sin(x), sin x, sin 2x: a bracket, else the next number or letter with its power and any letters after it."""
        if self.peek()[0] == "(":
            self.take()
            inner = self.sum()
            self.take(")")
            return inner
        value = self.power()
        while self.peek()[0] == "name" and not self.at_differential():
            value = sp.Mul(value, self.power())
        return value

    def limit(self):
        """lim_(x→a) body or lim x→a body; x→a+ / x→a- inside the bracket for one side."""
        if self.peek()[0] == "_":
            self.take()
        bracketed = self.peek()[0] == "("
        if bracketed:
            self.take()
        var = sp.Symbol(self.take("name")[1])
        self.take("arrow")
        target, side = self.unary(), "+-"
        if bracketed:
            if self.peek() in (("op", "+"), ("op", "-")) and self.peek(1)[0] == ")":
                side = self.take()[1]
            self.take(")")
        if target in (sp.oo, -sp.oo):
            side = "+"  # SymPy only accepts one side at infinity, and it makes no difference there
        return sp.Limit(self.sum(), var, target, side)

    def limit_value(self):
        """One end of an integral: 0, -1, (π/2), π/2, ∞. No power here: the ^ after the lower end starts the upper end."""
        negative = self.peek() == ("op", "-")
        if negative:
            self.take()
        value = -self.atom() if negative else self.atom()
        if self.peek() == ("op", "/") and self.peek(1)[0] == "num":  # ∫_0^π/2
            self.take()
            value = sp.Mul(value, sp.Pow(sp.Rational(self.take("num")[1]), -1))
        return value

    def integral(self):
        """∫_a^b f dx, ∫(a,b) f dx, ∫ f dx, ∫_a^b dx/(1+x^2)."""
        lower = upper = None
        if self.peek()[0] == "_":
            self.take()
            lower = self.limit_value()
            if self.peek() == ("op", "^"):
                self.take()
                upper = self.limit_value()
        elif self.peek()[0] == "(" and self.bracket_has_comma():
            self.take()
            lower = self.sum()
            self.take(",")
            upper = self.sum()
            self.take(")")
        self.in_integral += 1
        if self.at_differential():
            self.take()
            var = sp.Symbol(self.take("name")[1])
            body = sp.Integer(1)
            if self.peek() == ("op", "/"):
                self.take()
                body = sp.Pow(self.power(), -1)
                while self.peek()[0] in STARTS_FACTOR:
                    body = sp.Mul(body, self.power())
        else:
            body = self.sum()
            if self.at_differential():
                self.take()
                var = sp.Symbol(self.take("name")[1])
            else:
                var = X
        self.in_integral -= 1
        return sp.Integral(body, var) if lower is None else sp.Integral(body, (var, lower, upper))

    def bracket_has_comma(self):
        depth, j = 0, 0
        while self.peek(j)[0] is not None:
            kind = self.peek(j)[0]
            depth += (kind == "(") - (kind == ")")
            if kind == "," and depth == 1:
                return True
            if depth == 0:
                return False
            j += 1
        return False


@dataclass
class Answer:
    parsed: object            # what was read, unevaluated, for display
    value: object = None      # the answer to a sum, log or limit
    solutions: object = None  # the set of x for an equation
    holds: bool = None        # for an equation without x: is it true?
    variable: object = X      # the letter an equation was solved for


def _parse_side(tokens, natural_log=False, unevaluated=True):
    if not tokens:
        raise ValueError("Something is missing next to '='")
    parser = _Parser(tokens, natural_log)
    if unevaluated:
        with sp.evaluate(False):
            expr = parser.sum()
    else:
        expr = parser.sum()
    if parser.i < len(tokens):
        raise ValueError(f"Cannot read '{tokens[parser.i][1]}' here")
    return expr


def read_as_written(text):
    """Each side of a line, read but not worked out. Raises ValueError if it cannot be read."""
    sides = [[]]
    for tok in tokenize(text.strip().rstrip("=")):
        if tok == ("op", "="):
            sides.append([])
        else:
            sides[-1].append(tok)
    return [_parse_side(side) for side in sides]


def read_expression(text, natural_log=True):
    """A SymPy expression from text, for the course methods. Here x is always a letter and log is the natural log."""
    return _parse_side(tokenize(text, plain=False), natural_log, unevaluated=False)


def read_equation(text, natural_log=True):
    """(left, right) from 'a = b', or (a, 0) with no '='."""
    if text.count("=") > 1:
        raise ValueError("Use '=' only once")
    left, _, right = text.partition("=")
    return read_expression(left, natural_log), read_expression(right, natural_log) if right.strip() else sp.Integer(0)


def _work_out(expr):
    try:
        value = expr.doit()
    except ValueError as err:  # SymPy says a two-sided limit does not exist
        raise ValueError(str(err)) from None
    if value.has(sp.zoo):
        if expr.has(sp.Limit):  # SymPy's answer when the two sides head to -∞ and +∞
            raise ValueError("This limit does not exist: the left and right sides give different answers")
        raise ZeroDivisionError("Cannot divide by zero")
    if value.has(sp.nan):
        raise ValueError("The answer is undefined")
    if value.has(sp.Integral):
        if value.free_symbols:
            raise ValueError("This integral has no simple exact form")
        return sp.N(value, 10)
    return sp.nsimplify(value) if value.is_Float else sp.simplify(value)


def solve(text):
    """Read and solve one line of maths. A trailing '=' is ignored; 'a = b' with a letter is solved for x, else its first letter."""
    text = text.strip()
    if text.endswith("="):
        text = text[:-1]
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Nothing to solve")
    sides = [[]]
    for tok in tokens:
        if tok == ("op", "="):
            sides.append([])
        else:
            sides[-1].append(tok)
    if len(sides) > 2:
        raise ValueError("Use '=' only once")
    if len(sides) == 1:
        parsed = _parse_side(sides[0])
        return Answer(parsed, value=_work_out(parsed))
    left, right = (_parse_side(side) for side in sides)
    parsed = sp.Eq(left, right, evaluate=False)
    difference = _work_out(left) - _work_out(right)
    letters = sorted(parsed.free_symbols, key=lambda s: (s != X, s.name))
    if not letters:  # no x at all: just check the statement
        return Answer(parsed, holds=sp.simplify(difference) == 0)
    domain = sp.S.Reals if len(letters) == 1 else sp.S.Complexes  # with other letters in it, 'real' cannot be decided
    return Answer(parsed, solutions=sp.solveset(difference, letters[0], domain=domain), variable=letters[0])


def evaluate(text):
    """Value of a sum, log or limit (not an equation)."""
    answer = solve(text)
    if answer.value is None:
        raise ValueError("That is an equation; use solve()")
    return answer.value


def format_value(value):
    """33 -> '33', 7/2 -> '3.5', 7/3 -> '7/3 ≈ 2.33333', log(2) -> 'log(2) ≈ 0.693147'."""
    value = sp.sympify(value)
    if value in (sp.oo, -sp.oo):
        return "∞" if value == sp.oo else "-∞"
    if value.is_Rational:
        value = Fraction(int(value.p), int(value.q))
        if value.denominator == 1:
            return str(value.numerator)
        d = value.denominator
        for p in (2, 5):
            while d % p == 0:
                d //= p
        if d == 1:  # the decimal ends, so show it as a decimal
            return str(float(value))
        return f"{value.numerator}/{value.denominator} ≈ {float(value):.6g}"
    if value.is_Float:
        return f"{float(value):.6g}"
    text = sp.sstr(value).replace("**", "^").replace("*", "·").replace("E", "e").replace("pi", "π")
    if value.is_number:
        return f"{text} ≈ {float(sp.N(value)):.6g}"
    return text


def describe(answer):
    """One line for the answer: '33', 'x = 1', 'x = -2 or x = 2', 'true', ..."""
    if answer.value is not None:
        return format_value(answer.value)
    if answer.holds is not None:
        return "true ✓" if answer.holds else "false ✗"
    sols = answer.solutions
    name = answer.variable
    if sols == sp.S.EmptySet:
        return "no real solution"
    if sols == sp.S.Reals:
        return f"true for every {name}"
    if isinstance(sols, sp.FiniteSet):
        ordered = sorted(sols, key=lambda s: float(sp.N(s))) if all(s.is_number for s in sols) else list(sols)
        return " or ".join(f"{name} = {format_value(s)}" for s in ordered)
    return f"{name} in {sols}"
