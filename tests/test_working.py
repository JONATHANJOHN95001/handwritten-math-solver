import pytest
import sympy as sp

from questions import answer_question
from solver import describe, solve
from working import _integral_without_rules, explain, sum_lines, text_of


@pytest.mark.parametrize("text", ["12+7×4", "3-4-2", "(2+3)(4-1)^2", "2^(10)", "2^(-3)", "(3)/(4)", "1/2+1/3", "2/3×3/4", "1/2÷3/4", "1.5·4", "-3+2", "7÷3",
                                  "log(100)", "log_2(8)", "ln(e)", "sin(0)", "cos(0)", "√(25)", "√(50)", "5!", "23c4", "5p2", "2π", "-(2+3)", "3(4+1)", "1+2×3^2", "10÷(2+3)"])
def test_each_sum_is_worked_out_to_the_same_value_as_the_solver(text):
    lines, value = sum_lines(text)
    assert lines and sp.simplify(value - solve(text).value) == 0


def test_a_sum_is_worked_out_in_the_order_of_operations_as_written():
    lines, _ = sum_lines("12+7×4")
    assert lines == ["Work out: 12 + 7 × 4", "Multiply: 7 × 4 = 28, which leaves 12 + 28", "Add: 12 + 28 = 40"]
    assert sum_lines("(1+2)×3^2")[0][1:] == ["Add: 1 + 2 = 3, which leaves 3 × 3^2", "Power: 3^2 = 3 × 3 = 9, which leaves 3 × 9", "Multiply: 3 × 9 = 27"]


def test_fractions_stay_fractions_with_a_common_denominator():
    lines, value = sum_lines("1/2+1/3")
    assert value == sp.Rational(5, 6)
    assert "Add: 1/2 + 1/3 = 3/6 + 2/6 = 5/6 ≈ 0.833333" in lines


def test_numbers_and_formulas_are_written_as_people_write_them():
    x = sp.Symbol("x")
    assert text_of(sp.Rational(3, 4)) == "0.75" and text_of(sp.Rational(7, 3)) == "7/3"
    assert text_of(2 * x + 5) == "2x + 5" and text_of(x ** 2 - 4) == "x^2 − 4"
    assert text_of(sp.sqrt(x)) == "√(x)" and text_of(sp.log(x)) == "ln(x)" and text_of(sp.pi) == "π" and text_of(sp.oo) == "∞"


def _steps(text):
    return answer_question(text).steps


def test_a_linear_equation_is_solved_one_move_at_a_time():
    assert _steps("2x+5=3") == ["Equation: 2x+5=3", "Subtract 5 from both sides: 2x = -2", "Divide both sides by 2: x = -1",
                                "Check x = -1: the left side is 3 and the right side is 3 ✓"]


def test_a_linear_equation_with_x_on_both_sides_and_brackets():
    steps = _steps("3(x-1)=2x+4")
    assert "Work out each side: 3x − 3 = 2x + 4" in steps
    assert "Subtract 2x from both sides: x − 3 = 4" in steps
    assert "Add 3 to both sides: x = 7" in steps


def test_a_quadratic_shows_factors_discriminant_formula_and_checks():
    steps = _steps("x^2+3x+2=0")
    assert "Factor it: (x + 1)(x + 2) = 0" in steps
    assert "Discriminant: b^2 − 4ac = (3)^2 − 4(1)(2) = 1" in steps
    assert "With +: x = -1" in steps and "With −: x = -2" in steps
    assert sum(step.startswith("Check") for step in steps) == 2
    assert "The discriminant is negative, so there is no real solution" in _steps("x^2+x+1=0")


def test_a_pure_square_is_solved_with_the_square_root():
    assert "Take the square root of both sides: x = ±√(4) = ±2" in _steps("x^2=4")


@pytest.mark.parametrize("text,line", [("log(x)=2", "A log of base 10 undoes a power of 10: x = 10^2 = 100"), ("2^x=8", "Ask what power of 2 gives the right side: x = 3"),
                                       ("√(x+1)=3", "Square both sides: x + 1 = 9"), ("e^(2x)=5", "Take ln of both sides: 2x = ln(5)"), ("1/x=2", "Turn both sides upside down: x = 0.5")])
def test_other_equations_undo_what_was_done_to_the_unknown(text, line):
    assert line in _steps(text)


def test_equations_that_are_always_or_never_true_say_so():
    assert "never true" in _steps("x+1=x")[-1]
    assert "true for every value" in _steps("x+1=x+1")[-1]


def test_a_statement_works_out_each_side():
    assert _steps("2+2=5")[-1] == "The two sides are 4 and 5, so the statement is false"


def test_a_limit_tries_the_value_then_cancels():
    steps = _steps("lim_(x→2) ((x^2-4))/((x-2))")
    assert "That is 0/0, which has no value yet, so the expression has to be changed first" in steps
    assert "Cancel the common factor x − 2: x + 2" in steps
    assert steps[-1] == "So the limit is 4"


def test_a_limit_uses_lhopital_when_nothing_cancels():
    steps = _steps("lim_(x→0) (sin(x))/(x)")
    assert any("L'Hôpital" in step for step in steps) and steps[-1] == "So the limit is 1"


def test_a_limit_at_infinity_compares_the_highest_powers():
    steps = _steps("lim_(x→∞) (3x^2+1)/(x^2-2)")
    assert "Only the leading numbers stay: 3 ÷ 1 = 3" in steps and steps[-1] == "So the limit is 3"


@pytest.mark.parametrize("text,rule", [("d/dx x^3", "power rule"), ("d/dx sin(2x)", "chain rule"), ("d/dx x sin(x)", "product rule"), ("d/dx (x)/(x+1)", "quotient rule")])
def test_a_derivative_names_its_rule(text, rule):
    assert any(rule in step for step in _steps(text))


def test_a_second_derivative_goes_one_at_a_time():
    steps = _steps("d^2/dx^2 x^4")
    assert any("4x^3" in step for step in steps) and any("12x^2" in step for step in steps)


def test_a_derivative_inside_a_sum_gets_its_own_working():
    steps = _steps("d/dx x^2+3x+1")
    assert any("power rule" in step for step in steps) and not any("Derivative(" in step for step in steps)


@pytest.mark.parametrize("text,word", [("∫_(0)^(1) x^2dx", "power rule"), ("∫ sin(x)dx", "standard integral"), ("∫ x e^x dx", "by parts"), ("∫ 3x^2+2x dx", "each term")])
def test_an_integral_names_its_rule(text, word):
    assert any(word in step for step in _steps(text))


def test_a_definite_integral_puts_in_its_limits():
    steps = _steps("∫_(0)^(1) x^2dx")
    assert any("F(1) = 1/3 and F(0) = 0" in step for step in steps)


def test_working_never_stops_an_answer_it_cannot_explain():
    answer = solve("2+3")
    answer.parsed = sp.Symbol("q") + 1  # working out the text cannot match what was parsed
    answer.value = sp.Symbol("q") + 1
    assert explain("not a sum (((", answer)
    answer = solve("sin^2(0)")
    assert explain("sin^2(0)", answer)


@pytest.mark.parametrize("text", ["rank[[1,2,3],[2,4,6],[1,0,1]]", "echelon[[1,2],[3,4]]", "det[[1,2],[3,4]]", "det[[1,2,3],[4,5,6],[7,8,10]]", "det[[1,2,3,4],[0,1,2,3],[0,0,1,2],[1,0,0,1]]",
                                  "inverse[[1,2],[3,4]]", "charpoly[[2,1],[1,2]]", "eigen[[2,1],[1,2]]", "eigenvalues[[2,1],[1,2]]", "diagonalize[[2,1],[1,2]]", "[[1,1],[0,1]]^3", "[[1,1],[0,1]]^20",
                                  "matrix[[2,0],[0,3]]", "x+y=3; x-y=1", "x+y=1; x+y=2", "x^2+y=3; x-y=1"])
def test_each_matrix_method_shows_more_than_the_matrix(text):
    assert len(_steps(text)) >= 2


def test_rank_shows_each_row_operation():
    steps = _steps("rank[[1,2,3],[2,4,6],[1,0,1]]")
    assert any(step.startswith("R2 → R2 − 2·R1") for step in steps)
    assert any(step.startswith("R3 → R3 − R1") or step.startswith("R3 → R3 − 1·R1") for step in steps)


def test_a_determinant_shows_its_expansion():
    assert "For a 2 × 2 matrix, det = ad − bc = (1)(4) − (2)(3) = -2" in _steps("det[[1,2],[3,4]]")
    steps = _steps("det[[1,2,3],[4,5,6],[7,8,10]]")
    assert any(step.startswith("M11 = ") for step in steps) and any(step.startswith("det = ") and step.endswith("= -3") for step in steps)


def test_a_linear_system_ends_with_back_substitution():
    steps = _steps("x+y=3; x-y=1")
    assert any("so y = 1" in step for step in steps) and any("so x = 2" in step for step in steps)


@pytest.mark.parametrize("text", ["2x+5=3", "12+7×4", "lim_(x→0) (sin(x))/(x)", "d/dx x^3", "∫_(0)^(1) x^2dx", "rank[[1,2],[3,4]]", "2+2=4", "x+x", "5!"])
def test_every_answer_comes_with_working(text):
    result = answer_question(text)
    assert result.steps and all(isinstance(step, str) and step for step in result.steps)
    if not text.startswith(("rank", "∫")):
        assert describe(solve(text)) == result.answer[0]


@pytest.mark.parametrize("text, part", [
    ("Solve (D^2+5D+6)y=e^(3x).", "y_p = e^(3·x)/30"),
    ("Solve d^3y/dx^3+4d^2y/dx^2+dy/dx-6y=0.", "Auxiliary equation"),
    ("Solve (2x+1)^2 d^2y/dx^2-6(2x+1) dy/dx+16y=8(2x+1)^2.", "Cauchy-Legendre"),
    ("partial z=x^3+y^3-3axy", "treating y as a constant"),
    ("total derivative u=xy+yz+zx; x=t cos(t); y=t sin(t); z=t", "∂u/∂x = y + z and dx/dt"),
    ("extrema xy+a^3/x+a^3/y", "rt - s² = 3"),
    ("minimize positive xy+2yz+2zx subject to xyz=32", "where f = 48"),
    ("grad 2x+yz-3y^2", "∂f/∂y = -6·y + z"),
    ("div 3xz i+2xy j-yz^2 k", "∂v3/∂z = ∂(-y·z^2)/∂z = -2·y·z"),
    ("curl yz i+3xz j+z k", "i part: ∂v3/∂y - ∂v2/∂z = (0) - (3·x) = -3·x"),
    ("potential (2xy^2+yz) i+(2x^2y+xz+2yz^2) j+(2y^2z+xy) k", "Integrate F1 by x"),
    ("directional 2x^2+3y^2+z^2 at (2,1,3) along i-2k", "(-4) ÷ √(5) = "),
    ("gauss seidel 10x+y+z=12; x+10y+z=12; x+y+10z=12; iterations=3", "x = (-y - z + 12) ÷ 10"),
    ("empirical mode=32.1 mean=35.4", "(32.1 + 2 × 35.4) ÷ 3 = 34.3"),
    ("stats 4,1,5,4,3,7,2,3,4,1", "between places 5 and 6"),
    ("stats 1,2,9", "is place 2"),
    ("stats classes=10-12,12-14,14-16,16-18; freq=7,12,18,13", "14 + ((25 - 19) ÷ 18) × 2 = 14.6667"),
    ("correlation x=60,62,64,65,66,67,69,70,72,74; y=115,120,130,135,140,145,155,160,170,180", "840.0000 ÷ 174.9000 = 4.8027"),
    ("regression lines 8x-10y+66=0; 40x-18y=214; sx=3", "0.8 × 3 ÷ 0.6 = 4"),
    ("bayes labels=right,wrong; priors=60%,40%; likelihoods=40%,70%; want=right", "P(event) = 0.24 + 0.28 = 0.52"),
    ("binomial n=10 p=1/2 P(X>=6)", "P(X=6) = 0.205078"),
    ("poisson n=2000 p=0.001 P(X>2)", "So 1 - 0.67668 = 0.32332"),
    ("normal mean=70 sd=5 repeats=1000 P(X<65)", "z = (65 - 70) ÷ 5 = -1.00"),
    ("chisquare observed=20,22,17,18,19,24", "(22 - 20)² ÷ 20 = 0.2000"),
    ("product rule 4,12,3,2", "48 × 3 = 144"),
    ("derangements 8", "+ 40320 - 40320 + 20160 - 6720 + 1680 - 336 + 56 - 8 + 1 = 14833"),
    ("solutions total=30 lower=2,3,4,2,0", "C(23, 4) = 8855"),
    ("divisible 1 to 300 by 5,6,8", "⌊300/8⌋ - ⌊0/8⌋ = 37"),
    ("rook 3x3", "2 rooks: 18 ways"),
    ("recurrence a(n+2)=a(n+1)+a(n); a(0)=0; a(1)=1", "characteristic equation: r^2 - r - 1 = 0"),
    ("chromatic K2,3", "P(2) = 2"),
    ("spanning trees K4", "take the determinant: 16"),
    ("regions degrees=2,2,3,3,3,4,5,6,6", "r = e - v + 2 = 17 - 9 + 2 = 10"),
    ("huffman a:20, o:28, q:4, u:17, y:12, z:7", "Merge weights 4 and 7 into 11"),
])
def test_each_method_shows_how_its_answer_is_reached(text, part):
    steps = answer_question(text).steps
    assert any(part in step for step in steps), steps


def test_a_substitution_uses_a_plain_letter():
    steps = answer_question("∫_0^(π/2) sin^4 x dx").steps
    assert "Substitute t = 4x, so dt = 4 dx" in steps and not any("_u" in step for step in steps)


def test_an_integral_without_rules_still_shows_its_method():
    x = sp.Symbol("x")
    beta = _integral_without_rules(sp.Integral(sp.sin(x) ** 7 * sp.cos(x) ** 5, (x, 0, sp.pi / 2)), sp.Rational(1, 120))
    assert "½·B(4, 3) = Γ(4)Γ(3) ÷ (2Γ(7))" in beta and "= 6 × 2 ÷ (2 × 720)" in beta
    assert any("(1/b)·B(a/b, 1 - a/b)" in line for line in _integral_without_rules(sp.Integral(1 / (1 + x ** 4), (x, 0, sp.oo)), sp.sqrt(2) * sp.pi / 4))
    assert any("antiderivative" in line for line in _integral_without_rules(sp.Integral(x, (x, 0, 2)), sp.Integer(2)))
