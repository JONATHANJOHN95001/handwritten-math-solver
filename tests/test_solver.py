from fractions import Fraction

import pytest
import sympy as sp

from solver import describe, evaluate, format_value, solve


@pytest.mark.parametrize("text, expected", [
    ("12+7×3", 33),
    ("12+7x3", 33),
    ("12+7*3", 33),
    ("10-4÷2", 8),
    ("10-4/2", 8),
    ("2×3+4×5", 26),
    ("100÷10÷5", 2),
    ("8-3-2", 3),
    ("-5+2", -3),
    ("5×-2", -10),
    ("12+7×3=", 33),
    ("42", 42),
    ("2.5×4", 10),
    ("0.1+0.2", Fraction(3, 10)),
    ("(1+2)×3", 9),
    ("2(3+4)", 14),
    ("2^10", 1024),
    ("2^-1", Fraction(1, 2)),
])
def test_evaluate(text, expected):
    assert evaluate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("log(100)", 2),
    ("log 1000", 3),
    ("log_2(8)", 3),
    ("log_(2)8", 3),
    ("ln(e)", 1),
    ("ln e^2", 2),
    ("e^0", 1),
    ("sin(0)+cos(0)", 1),
])
def test_logs_and_functions(text, expected):
    assert evaluate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("lim_(x→0) sin(x)/x", 1),
    ("lim_(x→2) (x^2-4)/(x-2)", 4),
    ("lim_(x→∞) (2x+1)/x", 2),
    ("lim x→3 x^2", 9),
    ("lim_(x->0) (1-cos(x))/x^2", Fraction(1, 2)),
    ("lim_(x→∞) (1+1/x)^x", sp.E),
    ("lim_(x→0+) 1/x", sp.oo),
])
def test_limits(text, expected):
    assert evaluate(text) == expected


def test_two_sided_limit_that_does_not_exist():
    with pytest.raises(ValueError):
        evaluate("lim_(x→0) 1/x")


@pytest.mark.parametrize("text, expected", [
    ("2x+5=7", "x = 1"),
    ("x^2=4", "x = -2 or x = 2"),
    ("3x=x+4", "x = 2"),
    ("log_2(x)=3", "x = 8"),
    ("x/2+1=4", "x = 6"),
    ("2+3=5", "true ✓"),
    ("2+3=6", "false ✗"),
    ("x^2=-1", "no real solution"),
    ("x+1=x+1", "true for every x"),
])
def test_equations(text, expected):
    assert describe(solve(text)) == expected


def test_division_stays_exact():
    assert evaluate("7÷2") == Fraction(7, 2)
    assert format_value(evaluate("7÷2")) == "3.5"
    assert format_value(evaluate("7÷3")) == "7/3 ≈ 2.33333"
    assert format_value(evaluate("6÷2")) == "3"
    assert format_value(evaluate("0.1+0.2")) == "0.3"
    assert format_value(evaluate("log(2)")).endswith("≈ 0.30103")


def test_expression_with_x_and_no_equation_is_simplified():
    assert describe(solve("2x+3x")) == "5·x"


@pytest.mark.parametrize("text", ["", "5+", "×5", "5+=3", "5%3", "1.2.3", "2=3=4"])
def test_bad_input_raises_value_error(text):
    with pytest.raises(ValueError):
        solve(text)


@pytest.mark.parametrize("text, expected", [
    ("7x4", 28),
    ("8!", 40320),
    ("23C4", 8855),
    ("23c4", 8855),
    ("C(23,4)", 8855),
    ("5P2", 20),
    ("sqrt(16)", 4),
    ("√16", 4),
    ("sin^2(0)+cos^2(0)", 1),
    ("beta(2,3)", Fraction(1, 12)),
    ("gamma(5)", 24),
    ("d/dx x^3", 3 * sp.Symbol("x") ** 2),
    ("d^2/dx^2 x^3", 6 * sp.Symbol("x")),
    ("(d)/(dx) x^3", 3 * sp.Symbol("x") ** 2),
    ("∫_0^(π/2) sin^4 x dx", 3 * sp.pi / 16),
    ("∫_0^π/2 sin^4 x dx", 3 * sp.pi / 16),
    ("int(0,1) x^2 dx", Fraction(1, 3)),
    ("∫_0^∞ dx/(1+x^2)", sp.pi / 2),
    ("∫ 2x dx", sp.Symbol("x") ** 2),
    ("∫_0^π sin θ dθ", 2),
    ("e^2x", sp.exp(2 * sp.Symbol("x"))),
    ("sin 2x", sp.sin(2 * sp.Symbol("x"))),
    ("q+1", sp.Symbol("q") + 1),
    ("lim_(y→0) y", 0),
])
def test_new_notation(text, expected):
    assert sp.simplify(evaluate(text) - expected) == 0


@pytest.mark.parametrize("text, expected", [
    ("x1+x2=3", "x1 = 3 - x2"),
    ("2y+1=7", "y = 3"),
    ("y=x+1", "x = y - 1"),
])
def test_equations_in_other_letters(text, expected):
    assert describe(solve(text)) == expected


def test_subscripts_are_never_read_as_times():
    assert "x = 2" not in describe(solve("x1+x2+x3+x4+x5=30"))


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        evaluate("8÷0")
