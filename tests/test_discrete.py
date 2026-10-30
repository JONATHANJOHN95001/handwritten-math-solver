"""Real questions from Discrete Mathematics and Graph Theory (23BSCSMA31), with answers worked out by hand or by counting."""
import itertools
import math

import pytest
import sympy as sp

from questions import answer_question


def text_of(question):
    return "\n".join(answer_question(question).answer)


def count_solutions(total, ranges):
    return sum(1 for values in itertools.product(*ranges) if sum(values) == total)


def test_product_rule_end_sem_1a():
    assert "288" in text_of("product rule 4,12,3,2")


def test_pigeonhole_end_sem_2a():
    assert "2045" in text_of("pigeonhole 61327 / 30")


def test_derangements_end_sem_2b():
    assert "At least one in the right place: 40320 - 14833 = 25487" in text_of("derangements 8")


def test_stars_and_bars_assignment1_q5():
    assert "8855 solutions" in text_of("solutions total=30 lower=2,3,4,2,0")


def test_end_sem_1c_with_slack():
    # x1+x2+x3 ≤ 11 is x1+x2+x3+x4 = 11
    assert f"{math.comb(14, 3)} solutions" in text_of("solutions total=11 n=4")


def test_pencils_assignment2_q7():
    assert f"{count_solutions(24, [range(3, 9)] * 4)} solutions" in text_of("solutions total=24 n=4 lower=3 upper=8")


def test_bounds_with_negative_lower_limits_assignment2_q6():
    # c3 from -5 to 5; c1, c2 from -3 up; c4 from 0 up
    expected = sum(1 for c3 in range(-5, 6) for c1 in range(-3, 40) for c2 in range(-3, 40) if 20 - c3 - c1 - c2 >= 0)
    assert f"{expected} solutions" in text_of("solutions total=20 lower=-3,-3,-5,0 upper=,,5,")


def test_inclusion_exclusion_assignment1_q6():
    assert "Total: 100" in text_of("inclusion singles=12,20,20,8; pairs=5,7,4,16,4,3; triples=3,2,2,3; all=2; none=71")


def test_divisibility_assignment1_q7():
    at_least_one = sum(1 for n in range(1, 301) if n % 5 == 0 or n % 6 == 0 or n % 8 == 0)
    answer = text_of("divisible 1 to 300 by 5,6,8")
    assert f"at least one of [5, 6, 8]: {at_least_one}" in answer and f"by none: {300 - at_least_one}" in answer


def test_patterns_assignment1_q8():
    answer = text_of("patterns n=26; words=CAR,DOG,PUN,BYTE")
    f = math.factorial
    expected = (f(26) - (3 * f(24) + f(23)) + (3 * f(22) + 3 * f(21)) - (f(20) + 3 * f(19)) + f(17))
    assert f"{expected} permutations" in answer


def test_rook_3_by_3_assignment1_q10():
    answer = text_of("rook 3x3")
    assert "6·x^3" in answer and "18·x^2" in answer and "9·x" in answer


def test_forbidden_positions_assignment1_q11():
    # B1, B2 no apple; B3 no banana or mango; B4 no orange: count directly
    boys = {"B1": {"apple"}, "B2": {"apple"}, "B3": {"banana", "mango"}, "B4": {"orange"}}
    fruit = ["apple", "banana", "mango", "orange"]
    expected = sum(all(f not in boys[b] for b, f in zip(boys, order)) for order in itertools.permutations(fruit))
    assert f"{expected} ways" in text_of("forbidden B1:apple; B2:apple; B3:banana,mango; B4:orange")


def test_coefficient_assignment2_q4():
    x = sp.Symbol("x")
    expected = sp.Poly(sp.expand((x ** 2 + x ** 3 + x ** 4 + x ** 5 + x ** 6) ** 5), x).coeff_monomial(x ** 20)
    assert f"= {expected}" in text_of("coefficient of x^20 in (x^2+x^3+x^4+x^5+x^6)^5")


def test_coefficient_rational_assignment2_q5i():
    x = sp.Symbol("x")
    series = sp.series((1 + x) ** 4 / (1 - x) ** 4, x, 0, 16).removeO()
    assert f"= {sp.Poly(series, x).coeff_monomial(x ** 15)}" in text_of("coefficient of x^15 in (1+x)^4/(1-x)^4")


def test_coefficient_infinite_series_assignment2_q5ii():
    x = sp.Symbol("x")
    series = sp.series((1 + x + x ** 2) * (x / (1 - x) ** 2) ** 4, x, 0, 13).removeO()
    assert f"= {sp.Poly(series, x).coeff_monomial(x ** 12)}" in text_of("coefficient of x^12 in (1+x+x^2)(x+2x^2+3x^3+4x^4+...)^4")


@pytest.mark.parametrize("question, value", [
    ("gf 1,2,3,4,...", lambda x: 1 / (1 - x) ** 2),
    ("gf 0,1,-2,3,-4,...", lambda x: x / (1 + x) ** 2),
    ("gf 1^3,2^3,3^3,4^3,...", lambda x: (1 + 4 * x + x ** 2) / (1 - x) ** 4),
    ("gf 0^3,1^3,2^3,3^3,...", lambda x: x * (1 + 4 * x + x ** 2) / (1 - x) ** 4),
])
def test_generating_functions_assignment2(question, value):
    x = sp.Symbol("x")
    line = answer_question(question).answer[0].split("=", 1)[1].replace("·", "*").replace("^", "**")
    assert sp.simplify(sp.sympify(line) - value(x)) == 0


def test_sequence_from_generating_function_assignment2_q1():
    assert "27, 27, 9, 1, 0" in text_of("gf sequence of (3+x)^3")


def test_sequence_from_exponential_function_assignment2_q9():
    answer = text_of("egf sequence of 6e^(5x)-3e^(2x)")
    # a(r) = 6·5^r - 3·2^r
    assert "3, 24, 138, 726" in answer


def test_partitions_assignment2_q8():
    assert "p(6) = 11" in text_of("partitions of 6")


def test_engine_arrangements_end_sem_3b():
    expected = len(set(itertools.permutations("ENGINE", 4)))
    assert f"{expected} arrangements" in text_of("arrangements ENGINE 4")


def test_marbles_assignment2_q12():
    marbles = "RBBBGGOOWK"
    assert f"{len(set(itertools.permutations(marbles, 6)))} arrangements" in text_of("arrangements counts=1,3,2,2,1,1; r=6")


def test_recurrence_first_order_assignment3_q1():
    assert "3·4^n" in text_of("recurrence a(n+1)=4a(n); a(0)=3")


def test_recurrence_end_sem_4b():
    assert "2·7^n" in text_of("recurrence a(n)=7a(n-1); a(2)=98")


def test_recurrence_fibonacci_assignment3_q7():
    answer = text_of("recurrence F(n+2)=F(n+1)+F(n); F(0)=0; F(1)=1")
    function = answer.split("=", 1)[1].replace("·", "*").replace("^", "**").replace("√(", "sqrt(")
    expression = sp.sympify(function, locals={"n": sp.Symbol("n", integer=True)})
    values = [sp.simplify(expression.subs(sp.Symbol("n", integer=True), k)) for k in range(8)]
    assert values == [0, 1, 1, 2, 3, 5, 8, 13]


def _check_recurrence(answer, step, starts, count=8):
    function = answer.split("=", 1)[1].replace("·", "*").replace("^", "**")
    n = sp.Symbol("n", integer=True)
    expression = sp.sympify(function, locals={"n": n})
    values = [expression.subs(n, k) for k in range(count)]
    for start, value in enumerate(starts):
        assert sp.simplify(values[start] - value) == 0
    for k in range(len(starts), count):
        assert sp.simplify(values[k] - step(values, k)) == 0


def test_recurrence_constant_right_side_assignment3_q8():
    _check_recurrence(text_of("recurrence a(n)+4a(n-1)+4a(n-2)=8; a(0)=1; a(1)=2"), lambda v, k: 8 - 4 * v[k - 1] - 4 * v[k - 2], [1, 2])


def test_recurrence_power_right_side_assignment3_q3():
    _check_recurrence(text_of("recurrence a(n)-3a(n-1)=5*7^n; a(0)=2"), lambda v, k: 5 * 7 ** k + 3 * v[k - 1], [2])


def test_recurrence_exponential_assignment3_q12():
    _check_recurrence(text_of("recurrence a(n+2)+3a(n+1)+2a(n)=3^n; a(0)=0; a(1)=1"), lambda v, k: 3 ** (k - 2) - 3 * v[k - 1] - 2 * v[k - 2], [0, 1])


def test_virus_growth_end_sem_4a():
    # up 250% every two hours is ×3.5, twelve times in a day
    shown = answer_question("growth start=1000 percent=250 periods=12").answer[0].split(":")[1]
    assert float(shown) == pytest.approx(1000 * 3.5 ** 12)


def test_hanoi_and_handshakes_assignment3():
    assert "2^n - 1" in text_of("hanoi")
    assert "n(n-1)/2" in text_of("handshakes")
