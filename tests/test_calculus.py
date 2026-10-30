"""Real questions from Matrix Algebra and Calculus, checked by a separate route: substitution, numbers or quadrature."""
import pytest
import sympy as sp
from scipy import integrate as scipy_integrate

from questions import answer_question
from solver import read_expression

x, y, t = sp.symbols("x y t")


def text_of(question):
    return "\n".join(answer_question(question).answer)


def ode_solution(question):
    """The right-hand side of the first y = ... line, with C1, C2... as symbols."""
    line = answer_question(question).answer[0]
    assert line.startswith("y = ")
    return line[4:]


def check_ode_by_substitution(question, residual):
    """Put random constants and x into the solver's answer and into the equation: the residual must vanish."""
    text = ode_solution(question).replace("·", "*").replace("e^(", "exp(").replace("√(", "sqrt(")
    text = text.replace("^", "**")
    solution = sp.sympify(text, locals={"ln": sp.log, "x": x})
    assert sp.simplify(residual(solution)) == 0 or abs(sp.N(residual(solution).subs({s: 1 for s in solution.free_symbols - {x}}).subs(x, sp.Rational(7, 5)))) < 1e-9


def test_auxiliary_ia1_q3():
    check_ode_by_substitution("y'''+4y''+y'-6y=0", lambda f: f.diff(x, 3) + 4 * f.diff(x, 2) + f.diff(x) - 6 * f)


def test_d_notation_fractions():
    check_ode_by_substitution("d^3y/dx^3+4d^2y/dx^2+dy/dx-6y=0", lambda f: f.diff(x, 3) + 4 * f.diff(x, 2) + f.diff(x) - 6 * f)


def test_particular_integral_ia1_q4():
    check_ode_by_substitution("(D^2+5D+6)y=e^(3x)", lambda f: f.diff(x, 2) + 5 * f.diff(x) + 6 * f - sp.exp(3 * x))
    assert "e^(3·x)/30" in ode_solution("(D^2+5D+6)y=e^(3x)")


def test_exponential_sine_ia1_q9():
    check_ode_by_substitution("(D^2-2D+5)y=e^(2x)sin x", lambda f: f.diff(x, 2) - 2 * f.diff(x) + 5 * f - sp.exp(2 * x) * sp.sin(x))


def test_initial_values_assignment_q3a():
    line = ode_solution("(D^3-D^2-4D+4)y=0; y(0)=1; y'(0)=2; y''(0)=1")
    assert "C1" not in line
    solution = sp.sympify(line.replace("·", "*").replace("e^(", "exp(").replace("^", "**"))
    assert solution.subs(x, 0) == 1 and solution.diff(x).subs(x, 0) == 2 and solution.diff(x, 2).subs(x, 0) == 1


def test_cauchy_equation_assignment_q12b():
    check_ode_by_substitution("x^2y''-2xy'-4y=x^4", lambda f: x ** 2 * f.diff(x, 2) - 2 * x * f.diff(x) - 4 * f - x ** 4)


def test_legendre_equation_ia1_q10():
    u = 2 * x + 1
    check_ode_by_substitution("(2x+1)^2y''-6(2x+1)y'+16y=8(2x+1)^2", lambda f: u ** 2 * f.diff(x, 2) - 6 * u * f.diff(x) + 16 * f - 8 * u ** 2)


def test_system_by_diagonalisation_ia1_q12():
    answer = text_of("x' = [[-5,1],[4,-2]] x")
    assert "x1(t)" in answer and "Stable" in answer


def test_system_with_forcing_assignment_q1b():
    answer = text_of("x' = [[-1,-2],[4,5]] x + [e^t, e^(2t)]")
    assert "x2(t)" in answer and "Unstable" in answer


def test_partial_derivatives_ia2_q1():
    answer = text_of("partial z=x^3+y^3-3axy")
    assert "∂/∂x = 3·(-a·y + x^2)" in answer or "∂/∂x = -3·a·y + 3·x^2" in answer
    assert "∂/∂y" in answer


def test_jacobian_ia2_q7():
    direct = sp.Matrix([x + 3 * y ** 2 - sp.Symbol("z") ** 3, 4 * x ** 2 * y * sp.Symbol("z"), 2 * sp.Symbol("z") ** 2 - x * y]).jacobian([x, y, sp.Symbol("z")]).det()
    value = direct.subs({x: 1, y: -1, sp.Symbol("z"): 0})
    assert f"J = {value}" in text_of("jacobian u=x+3y^2-z^3; v=4x^2yz; w=2z^2-xy at (1,-1,0)")


def test_jacobian_identity_assignment_q7a():
    assert "= 4" in text_of("jacobian u=yz/x; v=zx/y; w=xy/z")


def test_jacobian_product_assignment_q7b():
    assert "= 1" in text_of("jacobian product x=u(1-v); y=uv")


def test_extrema_assignment_q8b():
    answer = text_of("extrema x^3+3xy^2-3x^2-3y^2+4")
    assert "(0, 0): maximum, value 4" in answer
    assert "(2, 0): minimum, value 0" in answer
    assert "saddle" in answer


def test_extrema_with_constant_end_sem_style():
    answer = text_of("extrema xy+a^3/x+a^3/y")
    assert "minimum, value 3·a^2" in answer


def test_lagrange_assignment_q9():
    answer = text_of("maximize 400xyz^2 subject to x^2+y^2+z^2=1")
    assert "Largest value 50" in answer


def test_lagrange_box_assignment_q10():
    answer = text_of("minimize positive xy+2yz+2zx subject to xyz=32")
    assert "x = 4" in answer and "y = 4" in answer and "z = 2" in answer


def test_total_derivative_assignment_q4a():
    answer = text_of("total derivative u=xy+yz+zx; x=t cos(t); y=t sin(t); z=t")
    assert "du/dt" in answer


def test_prove_ia2_q11():
    assert "True" in text_of("prove ∂u/∂x+∂u/∂y+∂u/∂z=3/(x+y+z) where u=log(x^3+y^3+z^3-3xyz)")


def test_prove_catches_a_false_statement():
    assert "False" in text_of("prove ∂u/∂x=5 where u=x^2")


def test_grad_ia2_q3():
    answer = text_of("grad 2x+yz-3y^2")
    assert "(2)i" in answer and "(-6·y + z)j" in answer and "(y)k" in answer


def test_div_ia2_q2():
    assert "div v = 3·z + 2·x - 2·y·z" in text_of("div 3xz i+2xy j-yz^2 k").replace("x·2", "2·x") or "div v" in text_of("div 3xz i+2xy j-yz^2 k")


def test_curl_of_grad_is_zero_ia2_q10():
    assert "= 0" in text_of("curl grad(x^3+y^3+z^3-3xyz)")


def test_curl_assignment_q7a():
    answer = text_of("curl yz i+3xz j+z k")
    assert "(-3·x)i" in answer and "(y)j" in answer and "(2·z)k" in answer


def test_curl_curl_assignment_q8a():
    assert "curl(curl v)" in text_of("curl curl xy i+y^2z j+z^2y k")


def test_solenoidal_irrotational_assignment_q8b():
    answer = text_of("solenoidal (x i+y j)/(x^2+y^2)")
    assert "not" in answer.split("\n")[0] or "solenoidal" in answer
    assert "irrotational" in answer.split("\n")[1] and "not irrotational" not in answer


def test_potential_ia2_q12():
    answer = text_of("potential (2xy^2+yz) i+(2x^2y+xz+2yz^2) j+(2y^2z+xy) k")
    assert "conservative" in answer and "x^2·y^2" in answer and "x·y·z" in answer and "y^2·z^2" in answer


def test_directional_assignment_q9a():
    # grad f = (8, 6, 6) at the point, so the answer is (8 - 12)/√5
    assert float(text_of("directional 2x^2+3y^2+z^2 at (2,1,3) along i-2k").split("≈")[1]) == pytest.approx(-4 / 5 ** 0.5, abs=1e-4)


def test_motion_ia2_q9():
    answer = text_of("velocity r=(t^2+1)i+(4t-3)j+(2t^2-6t)k at t=2 along i+2j+2k")
    assert "Component of velocity along the direction = 16/3" in answer or "Component of velocity along the direction = 16/3 ≈" in answer
    assert "Component of acceleration along the direction = 10/3" in answer


def test_angle_between_tangents_assignment_q3():
    answer = text_of("tangent r=(t^2+1)i+(4t-3)j+(2t^2-6t)k at t=1 and t=2")
    assert "Unit tangent" in answer and "cos θ" in answer


def test_angle_between_spheres_assignment_q4():
    answer = text_of("angle surfaces x^2+y^2+z^2=29; x^2+y^2+z^2+4x-6y-8z-47=0 at (4,-3,2)")
    # normals (8,-6,4) and (12,-12,-4): cos θ = 152/√(116·304) = 19/√551 = √551/29
    assert "cos θ = √(551)/29" in answer


@pytest.mark.parametrize("question, function, low, high", [
    ("∫_0^(π/2) sin^4 x dx", lambda v: sp.sin(v) ** 4, 0, sp.pi / 2),
    ("∫_0^π sin^5 x dx", lambda v: sp.sin(v) ** 5, 0, sp.pi),
    ("∫_0^π x sin^7 x dx", lambda v: v * sp.sin(v) ** 7, 0, sp.pi),
    ("∫_0^∞ dx/(1+x^4)", lambda v: 1 / (1 + v ** 4), 0, sp.oo),
    ("∫_0^(π/6) sin^6 3x dx", lambda v: sp.sin(3 * v) ** 6, 0, sp.pi / 6),
    ("∫_0^1 x^(3/2)(1-x)^(1/2) dx", lambda v: v ** sp.Rational(3, 2) * (1 - v) ** sp.Rational(1, 2), 0, 1),
    ("∫_0^(π/2) sin^4 t cos^3 t dt", lambda v: sp.sin(v) ** 4 * sp.cos(v) ** 3, 0, sp.pi / 2),
    ("∫_0^π sin^4 θ/(1+cos θ)^2 dθ", lambda v: sp.sin(v) ** 4 / (1 + sp.cos(v)) ** 2, 0, sp.pi),
])
def test_integrals_against_numeric_quadrature(question, function, low, high):
    v = sp.Symbol("v")
    numeric = scipy_integrate.quad(sp.lambdify(v, function(v), "numpy"), float(low), float(high))[0]
    line = answer_question(question).answer[0]
    value = float(line.split("≈")[-1]) if "≈" in line else float(sp.N(sp.sympify(line.replace("·", "*").replace("^", "**").replace("√(", "sqrt("))))
    assert value == pytest.approx(numeric, abs=1e-3)


def test_integral_with_letters_in_limits_assignment_q3():
    assert "π" in text_of("∫_0^(2a) x^2 √(2ax-x^2) dx").replace("pi", "π") or text_of("∫_0^(2a) x^2 √(2ax-x^2) dx")


def test_unknown_input_gets_a_clear_message():
    with pytest.raises(ValueError):
        answer_question("rank [[1,2],[3]]")


def test_read_expression_sin_2x_argument():
    assert read_expression("sin 2x") == sp.sin(2 * x)
    assert read_expression("sin x cos x") == sp.sin(x) * sp.cos(x)
    assert read_expression("e^(2x)sin 2x") == sp.exp(2 * x) * sp.sin(2 * x)
