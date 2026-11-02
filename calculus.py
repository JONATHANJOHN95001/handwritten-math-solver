"""Calculus: differential equations, partial derivatives, vector calculus and integrals."""
import re

import sympy as sp

from matrices import determinant_lines
from results import Result, decimal, parts_of, show, show_matrix, show_plain, time_limited
from solver import X, read_equation, read_expression
from working import explain_integral

y_symbol, D = sp.Symbol("y"), sp.Symbol("D")
T = sp.Symbol("t")
VARIABLES = ("x", "y", "z", "t", "u", "v", "w", "r", "θ")
ODE_KINDS = {"separable": "separable: put the y terms with dy and the other terms with the other differential, then integrate both sides",
             "Bernoulli": "Bernoulli: substituting u = y^(1-n) makes it linear",
             "1st_exact": "exact: it is the total differential of some function, which is set equal to a constant",
             "1st_homogeneous_coeff_best": "homogeneous: substituting y = vx makes it separable",
             "nth_linear_euler_eq_homogeneous": "Cauchy-Euler: trying y = x^m gives an equation for m",
             "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters": "Cauchy-Euler with a right side: y = x^m for the left side, then variation of parameters",
             "nth_linear_constant_coeff_variation_of_parameters": "linear with constant coefficients: the particular part comes from variation of parameters"}


# ---------- differential equations ----------

def _apply_d_operator(expression, unknown, var):
    """(D^2+5D+6)y becomes y''+5y'+6y."""
    total = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(expression)):
        coefficient, order = term.as_coeff_exponent(D)
        if coefficient.has(y_symbol):
            coefficient = coefficient.subs(y_symbol, 1)
            total += coefficient * (sp.Derivative(unknown, (var, int(order))) if order else unknown)
        else:
            total += term
    return total


def _independent_variable(expression):
    for derivative in expression.atoms(sp.Derivative):
        return derivative.variables[0]
    return X


def read_ode(text):
    """An equation in y', dy/dx or D as an expression equal to 0, in the unknown function y."""
    left, right = read_equation(text)
    expression = left - right
    var = _independent_variable(expression)
    unknown = sp.Function("y")(var)
    if expression.has(D):
        expression = _apply_d_operator(expression, unknown, var)
    return expression.subs(y_symbol, unknown), unknown, var


def _initial_values(parts, unknown, var):
    """y(0)=1, y'(0)=2 for dsolve."""
    conditions = {}
    for part in parts:
        match = re.fullmatch(r"y('*)\(\s*(.+?)\s*\)\s*=\s*(.+)", part)
        if not match:
            raise ValueError(f"Cannot read the initial value '{part}'")
        at, value = read_expression(match[2]), read_expression(match[3])
        order = len(match[1])
        key = unknown.subs(var, at) if not order else unknown.diff(var, order).subs(var, at)
        conditions[key] = value
    return conditions


def _legendre_form(expression, unknown, var):
    """Spot (ax+b)^2 y'' + ... and return (a, b) so it can be solved as a Cauchy equation in u = ax+b."""
    second = sp.expand(expression).coeff(sp.Derivative(unknown, (var, 2)))
    if not second.has(var) or not second.is_polynomial(var):
        return None
    _, factors = sp.factor_list(second, var)
    if len(factors) != 1 or factors[0][1] != 2:
        return None
    linear = factors[0][0]
    if sp.degree(linear, var) != 1 or linear.subs(var, 0) == 0:
        return None  # a plain Cauchy equation in x: dsolve handles that itself
    a = linear.coeff(var, 1)
    return a, linear - a * var


def _solve_legendre(expression, unknown, var):
    a, b = _legendre_form(expression, unknown, var)
    u = sp.Symbol("u", positive=True)
    new = sp.Function("Y")(u)
    highest = max(d.derivative_count for d in expression.atoms(sp.Derivative))
    changed = expression
    for order in range(highest, 0, -1):
        changed = changed.subs(sp.Derivative(unknown, (var, order)), a ** order * sp.Derivative(new, (u, order)))
    changed = changed.subs(unknown, new).subs(var, (u - b) / a)
    solution = sp.dsolve(sp.simplify(changed), new)
    return sp.Eq(unknown, solution.rhs.subs(u, a * var + b))


def _classify(expression, unknown, var):
    order = max((d.derivative_count for d in expression.atoms(sp.Derivative)), default=0)
    homogeneous = sp.simplify(expression.subs({d: 0 for d in expression.atoms(sp.Derivative)}).subs(unknown, 0)) == 0
    return order, homogeneous


def _auxiliary(expression, unknown, var):
    """The auxiliary equation when the coefficients are constants."""
    m = sp.Symbol("m")
    poly = sp.Integer(0)
    derivatives = sorted(expression.atoms(sp.Derivative), key=lambda d: -d.derivative_count)
    for derivative in derivatives:
        coefficient = sp.expand(expression).coeff(derivative)
        if coefficient.has(var):
            return None
        poly += coefficient * m ** derivative.derivative_count
    coefficient = sp.expand(expression).coeff(unknown)
    if coefficient.has(var):
        return None
    return poly + coefficient, m


def _ode_text(expression):
    """y'' + 5y' + 6y for SymPy's Derivative(y(x), (x, 2)) + ..."""
    text = show(expression)
    text = re.sub(r"Derivative\(y\(\w+\), \(\w+, (\d)\)\)", lambda m: "y" + "'" * int(m[1]), text)
    return re.sub(r"Derivative\(y\(\w+\), \w+\)", "y'", text).replace("y(x)", "y").replace("y(t)", "y")


def _first_order_lines(expression, unknown, var):
    """y' + P y = Q with the integrating factor, when the equation has that form."""
    expanded = sp.expand(expression)
    derivative = unknown.diff(var)
    a, b = expanded.coeff(derivative), expanded.coeff(unknown)
    q = sp.simplify(-(expanded - a * derivative - b * unknown))
    if a == 0 or a.has(unknown) or b.has(unknown) or q.has(unknown):
        return []
    p_part, q_part = sp.simplify(b / a), sp.simplify(q / a)
    factor = sp.simplify(sp.exp(sp.integrate(p_part, var)))
    return [f"Divide by the number in front of y' to get y' + P·y = Q, with P = {show(p_part)} and Q = {show(q_part)}",
            f"Integrating factor: e^(∫P d{var}) = {show(factor)}",
            f"Multiply through by it, and the left side becomes the derivative of y × ({show(factor)})",
            f"Integrate both sides: y × ({show(factor)}) = ∫({show(sp.simplify(q_part * factor))}) d{var} = {show(sp.integrate(sp.simplify(q_part * factor), var))} + C"]


def _ode_method_lines(expression, unknown, var, order, auxiliary, conditions):
    """How the solution is built up. It only explains, so anything it cannot explain is left out and the answer stands."""
    lines = []
    derivatives = expression.atoms(sp.Derivative)
    right_side = sp.simplify(-expression.subs({d: 0 for d in derivatives}).subs(unknown, 0))
    if auxiliary:
        lines.append("Each root m gives a term e^(mx); a repeated root also gives x·e^(mx); a pair a ± bi gives e^(ax)(C1·cos(bx) + C2·sin(bx))")
        general = sp.dsolve(expression, unknown).rhs
        constants = general.free_symbols - expression.free_symbols - {var}
        particular = general.subs(dict.fromkeys(constants, 0))
        lines.append(f"Complementary function, from the roots: y_c = {show(sp.expand(general - particular))}")
        if right_side != 0:
            lines.append(f"The right side is {show(right_side)}. A trial solution of that form gives the particular integral: y_p = {show(particular)}")
            lines.append(f"General solution: y = y_c + y_p = {show(general)}")
    else:
        hints = sp.classify_ode(expression, unknown)
        if order == 1 and "1st_linear" in hints:
            lines += _first_order_lines(expression, unknown, var)
        kind = next((ODE_KINDS[hint] for hint in hints if hint in ODE_KINDS), None)
        if kind and not lines:
            lines.append(f"This kind of equation is {kind}")
        lines.append(f"General solution: y = {show(sp.dsolve(expression, unknown).rhs)}")
    if conditions:
        lines.append("Put in the given values to fix the constants: " + ", ".join(f"{_ode_text(k)} = {show_plain(v)}" for k, v in conditions.items()))
    return lines


def solve_ode(text):
    parts = parts_of(text)
    expression, unknown, var = read_ode(parts[0])
    conditions = _initial_values(parts[1:], unknown, var)
    order, homogeneous = _classify(expression, unknown, var)
    steps = [f"With everything on one side: {_ode_text(expression)} = 0", f"Order {order}, {'homogeneous' if homogeneous else 'non-homogeneous'}"]
    auxiliary = _auxiliary(expression, unknown, var)
    legendre = None if auxiliary else _legendre_form(expression, unknown, var)
    if legendre:
        inner = legendre[0] * var + legendre[1]
        steps.append(f"The powers of {show(inner)} go with the order of each derivative, so this is a Cauchy-Legendre equation")
        steps.append(f"Put u = {show(inner)}, so d/d{var} = {show_plain(legendre[0])}·d/du. This gives a Cauchy-Euler equation in u, which y = u^m solves")
    if auxiliary:
        poly, m = auxiliary
        roots = sp.roots(sp.Poly(poly, m))
        steps.append(f"Auxiliary equation, with y' replaced by m and y'' by m^2: {show(poly)} = 0")
        if roots:
            steps.append("Roots: " + ", ".join(show_plain(r) for r, n in roots.items() for _ in range(n)))

    def work():
        if _legendre_form(expression, unknown, var) and not auxiliary:
            return _solve_legendre(expression, unknown, var)
        return sp.dsolve(expression, unknown, ics=conditions or None)

    solution = time_limited(work, 40)
    solutions = solution if isinstance(solution, list) else [solution]
    try:
        if not legendre:
            steps += time_limited(lambda: _ode_method_lines(expression, unknown, var, order, auxiliary, conditions), 15)
    except Exception:  # the answer is already found, and SymPy can fail on the separate parts of an unusual equation
        pass
    if not (steps[-1].startswith("General solution") and steps[-1].endswith(show(solutions[0].rhs))):
        steps.append("Solution: " + "; ".join(f"y = {show(s.rhs)}" for s in solutions))
    return Result("Differential equation", [f"y = {show(s.rhs)}" for s in solutions], steps)


def system_ode(matrix, forcing=None):
    """X' = AX + f(t) by diagonalisation, and the stability of the system."""
    values = sorted(matrix.eigenvals().items(), key=lambda item: float(sp.re(sp.N(item[0]))))
    if not matrix.is_diagonalizable():
        raise ValueError("This matrix cannot be diagonalised")
    p, d = matrix.diagonalize(sort=True)
    constants = sp.symbols(f"c1:{matrix.rows + 1}")
    g = p.inv() * forcing if forcing is not None else sp.zeros(matrix.rows, 1)
    parts = []
    for i in range(matrix.rows):
        rate = d[i, i]
        particular = sp.exp(rate * T) * sp.integrate(sp.exp(-rate * T) * g[i], T)
        parts.append(constants[i] * sp.exp(rate * T) + particular)
    solution = (p * sp.Matrix(parts)).applyfunc(sp.simplify)
    steps = [f"Eigenvalues of the matrix A: {', '.join(show_plain(v) for v, n in values for _ in range(n))}", f"Eigenvectors in the columns of P:\n{show_matrix(p)}",
             f"Eigenvalues on the diagonal of D:\n{show_matrix(d)}", "Put X = P·Y. Then Y' = D·Y + P⁻¹f, which is one simple equation for each Yi",
             *[f"Y{i + 1}' = {show_plain(d[i, i])}·Y{i + 1} + {show(g[i])}, so Y{i + 1} = {show(parts[i])}" for i in range(matrix.rows)],
             f"X = P·Y gives the solution:\n{show_matrix(solution)}"]
    real_parts = [sp.re(v) for v in matrix.eigenvals()]
    if all(r < 0 for r in real_parts):
        stability = "Stable: every eigenvalue has a negative real part"
    elif any(r > 0 for r in real_parts):
        stability = "Unstable: an eigenvalue has a positive real part"
    else:
        stability = "Marginally stable: the largest real part of an eigenvalue is 0"
    answer = [f"x{i + 1}(t) = {show(solution[i])}" for i in range(matrix.rows)] + [stability]
    return Result("System of differential equations", answer, steps)


# ---------- partial derivatives ----------

def _variables(expression):
    return sorted((s for s in expression.free_symbols if s.name in VARIABLES), key=lambda s: VARIABLES.index(s.name))


def _function_body(text):
    """'z=x^3+y^3' or 'f(x,y)=x^3' or plain 'x^3': the formula on the right."""
    return read_expression(re.sub(r"^\s*([a-zA-Zφ]\w*(\([^)]*\))?)\s*=", "", text))


def partials(text):
    body = _function_body(text)
    letters = _variables(body)
    answer, first, second = [], [f"The function: {show(body)}"], []
    for v in letters:
        answer.append(f"∂/∂{v} = {show(sp.simplify(body.diff(v)))}")
        others = [str(o) for o in letters if o != v]
        held = f"treating {', '.join(others)} as {'a constant' if len(others) == 1 else 'constants'}: " if others else ""
        first.append(f"∂/∂{v}, {held}{show(sp.simplify(body.diff(v)))}")
    for i, a in enumerate(letters):
        for b in letters[i:]:
            label = f"∂²/∂{a}²" if a == b else f"∂²/∂{a}∂{b}"
            second.append(f"{label}, differentiating ∂/∂{a} by {b}: {show(sp.simplify(body.diff(a, b)))}")
    return Result("Partial derivatives", answer, first + second)


def total_derivative(text):
    """'u=xy+yz+zx; x=tcos(t); y=tsin(t); z=t' or with 'at t=π/4': du/dt by the chain rule and by substitution."""
    parts = parts_of(text)
    at = None
    if re.match(r"at\s", parts[-1]):
        at = parts.pop()
    definitions = [re.split(r"=", part, maxsplit=1) for part in parts]
    name, body = definitions[0][0].strip(), read_expression(definitions[0][1])
    inner = {sp.Symbol(k.strip()): read_expression(v) for k, v in definitions[1:]}
    var = next(iter(set().union(*(v.free_symbols for v in inner.values()))))
    chain = sum(body.diff(k) * v.diff(var) for k, v in inner.items()).subs(inner)
    direct = body.subs(inner).diff(var)
    if sp.simplify(chain - direct) != 0:
        raise ValueError("The chain rule and direct substitution disagree on this one")
    answer = [f"d{name}/d{var} = {show(sp.simplify(direct))}"]
    if at:
        point = read_expression(at.split("=")[1])
        answer.append(f"At {var} = {show_plain(point)}: {show(sp.simplify(direct.subs(var, point)))}")
    steps = [f"Chain rule: d{name}/d{var} = " + " + ".join(f"(∂{name}/∂{k})(d{k}/d{var})" for k in inner)]
    steps += [f"∂{name}/∂{k} = {show(body.diff(k))} and d{k}/d{var} = {show(v.diff(var))}" for k, v in inner.items()]
    steps += [f"Multiply them, add them and write everything in {var}: {show(sp.simplify(direct))}", f"Check: putting {', '.join(map(str, inner))} into {name} first and then differentiating gives the same result"]
    return Result("Total derivative", answer, steps)


def implicit_derivative(body_text, constraint_text):
    """du/dx of u(x,y) where y depends on x through a constraint."""
    y = sp.Symbol("y")
    body = read_expression(body_text)
    left, right = read_equation(constraint_text)
    constraint = left - right
    dydx = -constraint.diff(X) / constraint.diff(y)
    return Result("Total derivative with a constraint", [f"du/dx = {show(sp.simplify(body.diff(X) + body.diff(y) * dydx))}"],
                  [f"Write the constraint as g = 0, with g = {show(constraint)}",
                   f"Differentiating it: dy/dx = -(∂g/∂x) ÷ (∂g/∂y) = -({show(constraint.diff(X))}) ÷ ({show(constraint.diff(y))}) = {show(sp.simplify(dydx))}",
                   f"du/dx = ∂u/∂x + (∂u/∂y)(dy/dx), with ∂u/∂x = {show(body.diff(X))} and ∂u/∂y = {show(body.diff(y))}",
                   f"Put them together: {show(sp.simplify(body.diff(X) + body.diff(y) * dydx))}"])


def jacobian(text, at=None):
    """'u=...; v=...; w=...' with the variables taken from the formulas."""
    parts = parts_of(text)
    definitions = [re.split(r"=", part, maxsplit=1) for part in parts]
    names = [k.strip() for k, _ in definitions]
    functions = sp.Matrix([read_expression(v) for _, v in definitions])
    letters = _variables(sp.Add(*functions))[:len(names)]
    matrix = functions.jacobian(letters)
    det = sp.simplify(matrix.det())
    answer = [f"J = ∂({','.join(names)})/∂({','.join(map(str, letters))}) = {show(det)}"]
    if at is not None:
        point = dict(zip(letters, at))
        answer.append(f"At {tuple(at)}: J = {show(det.subs(point))}")
    steps = [f"Differentiate each function by each variable, in rows and columns:\n{show_matrix(matrix)}", *determinant_lines(matrix), f"Simplify: J = {show(det)}"]
    if at is not None:
        steps.append(f"Put in the point {tuple(at)}: J = {show(det.subs(point))}")
    return Result("Jacobian", answer, steps)


def jacobian_product(text):
    """Show JJ' = 1 for x=u(1-v), y=uv."""
    parts = parts_of(text)
    definitions = [re.split(r"=", part, maxsplit=1) for part in parts]
    outer = [sp.Symbol(k.strip()) for k, _ in definitions]
    functions = [read_expression(v) for _, v in definitions]
    inner = _variables(sp.Add(*functions))
    forward = sp.Matrix(functions).jacobian(inner).det()
    solved = sp.solve([f - o for f, o in zip(functions, outer)], inner, dict=True)[0]
    back = sp.Matrix([solved[v] for v in inner]).jacobian(outer).det()
    product = sp.simplify(forward * back.subs({o: f for o, f in zip(outer, functions)}))
    return Result("Jacobian product", [f"J·J' = {show(product)}"], [
        f"J from the given formulas:\n{show_matrix(sp.Matrix(functions).jacobian(inner))}", f"J = {show(sp.simplify(forward))}",
        "Solve the formulas the other way round: " + ", ".join(f"{v} = {show(solved[v])}" for v in inner),
        f"J' from those:\n{show_matrix(sp.Matrix([solved[v] for v in inner]).jacobian(outer))}", f"J' = {show(sp.simplify(back))}",
        f"Multiply them, writing J' in the same letters as J: {show(product)}"])


def extrema(text):
    """Stationary points of f(x,y) and what kind each one is. Letters other than x, y are constants, taken as positive."""
    body = _function_body(text)
    letters = _variables(body)
    constants = body.free_symbols - set(letters)
    positive = {c: sp.Symbol(c.name, positive=True) for c in constants}
    body = body.subs(positive)
    points = sp.solve([body.diff(v) for v in letters], letters, dict=True)
    answer, steps = [], [f"First derivatives: ∂/∂{v} = {show(body.diff(v))}" for v in letters]
    if len(letters) != 2:
        raise ValueError("This method handles functions of two variables")
    a, b = letters
    real = [point for point in points if all(sp.im(v) == 0 for v in point.values())]
    steps.append("Set both to 0 and solve them together. The real solutions are the stationary points: " + ("; ".join(
        ", ".join(f"{k} = {show_plain(v)}" for k, v in point.items()) for point in real) or "there are none"))
    steps.append(f"Second derivatives: r = ∂²/∂{a}² = {show(body.diff(a, a))}, s = ∂²/∂{a}∂{b} = {show(body.diff(a, b))}, t = ∂²/∂{b}² = {show(body.diff(b, b))}")
    steps.append("At each point: rt - s² > 0 with r > 0 is a minimum, with r < 0 a maximum; rt - s² < 0 is a saddle point")
    for point in points:
        if not all(sp.im(v) == 0 for v in point.values()) or len(point) < 2:
            continue
        r, s, t = (body.diff(a, a).subs(point), body.diff(a, b).subs(point), body.diff(b, b).subs(point))
        d = sp.simplify(r * t - s ** 2)
        place = f"({show_plain(point[a])}, {show_plain(point[b])})"
        value = sp.simplify(body.subs(point))
        if d.is_positive and r.is_positive:
            kind = f"minimum, value {show(value)}"
        elif d.is_positive and r.is_negative:
            kind = f"maximum, value {show(value)}"
        elif d.is_negative:
            kind = "saddle point"
        else:
            kind = "the test does not decide"
        answer.append(f"{place}: {kind}")
        steps.append(f"At {place}: r = {show(r)}, s = {show(s)}, t = {show(t)}, rt - s² = {show(d)}, which gives: {kind}")
    if not answer:
        answer = ["No real stationary point"]
    return Result("Maxima and minima", answer, steps)


def lagrange(objective_text, constraint_text, positive=False):
    """Extreme values of f subject to g = c, by Lagrange multipliers."""
    objective = read_expression(objective_text)
    left, right = read_equation(constraint_text)
    constraint = left - right
    letters = _variables(objective + constraint)
    if positive:
        swap = {v: sp.Symbol(v.name, positive=True) for v in letters}
        objective, constraint, letters = objective.subs(swap), constraint.subs(swap), list(swap.values())
    lam = sp.Symbol("λ")
    equations = [objective.diff(v) - lam * constraint.diff(v) for v in letters] + [constraint]
    solutions = time_limited(lambda: sp.solve(equations, letters + [lam], dict=True), 30)
    real = [s for s in solutions if all(sp.im(v) == 0 for v in s.values()) and all(v in s for v in letters)]
    if not real:
        raise ValueError("No stationary point was found")
    ranked = sorted(real, key=lambda s: float(sp.N(objective.subs(s))))
    low, high = ranked[0], ranked[-1]

    def place(solution):
        return ", ".join(f"{v} = {show_plain(solution[v])}" for v in letters)

    answer = [f"Largest value {show_plain(sp.simplify(objective.subs(high)))} at {place(high)}",
              f"Smallest value {show_plain(sp.simplify(objective.subs(low)))} at {place(low)}"]
    steps = [f"Write the constraint as g = 0, with g = {show(constraint)}", f"Form F = f - λg, with f = {show(objective)}",
             *[f"∂F/∂{v} = 0: {show(equation)} = 0" for v, equation in zip(letters, equations)], "Solve these together with g = 0"]
    steps += [f"Stationary point {place(point)}, where f = {show_plain(sp.simplify(objective.subs(point)))}" for point in ranked]
    steps.append("Compare the values of f: the largest and the smallest are the answers")
    return Result("Lagrange multipliers", answer, steps)


# ---------- vector calculus ----------

I, J, K = sp.symbols("i j k")
XYZ = sp.symbols("x y z")


def read_vector(text):
    """'3xz i+2xy j-yz^2 k' or '(x i+y j)/(x^2+y^2)' as a 3-vector of formulas."""
    text = text.replace("î", "i").replace("ĵ", "j").replace("k̂", "k").replace("𝚤̂", "i").replace("𝚥̂", "j").replace("^i", "i")
    expression = read_expression(text)
    return sp.Matrix([expression.subs({I: 1, J: 0, K: 0}) - expression.subs({I: 0, J: 0, K: 0}),
                      expression.subs({I: 0, J: 1, K: 0}) - expression.subs({I: 0, J: 0, K: 0}),
                      expression.subs({I: 0, J: 0, K: 1}) - expression.subs({I: 0, J: 0, K: 0})])


def _grad(scalar):
    return sp.Matrix([scalar.diff(v) for v in XYZ])


def _div(vector):
    return sum(vector[n].diff(XYZ[n]) for n in range(3))


def _curl(vector):
    x, y, z = XYZ
    return sp.Matrix([vector[2].diff(y) - vector[1].diff(z), vector[0].diff(z) - vector[2].diff(x), vector[1].diff(x) - vector[0].diff(y)])


def show_vector(vector):
    """(2x)i + (3)j - style, leaving out zero parts."""
    pieces = [f"({show(sp.simplify(entry))}){unit}" for entry, unit in zip(vector, "ijk") if sp.simplify(entry) != 0]
    return " + ".join(pieces) if pieces else "0"


def _curl_lines(vector):
    x, y, z = XYZ
    return ["curl v is the determinant of the rows (i, j, k), (∂/∂x, ∂/∂y, ∂/∂z) and (v1, v2, v3)",
            f"i part: ∂v3/∂y - ∂v2/∂z = ({show(vector[2].diff(y))}) - ({show(vector[1].diff(z))}) = {show(sp.simplify(vector[2].diff(y) - vector[1].diff(z)))}",
            f"j part: ∂v1/∂z - ∂v3/∂x = ({show(vector[0].diff(z))}) - ({show(vector[2].diff(x))}) = {show(sp.simplify(vector[0].diff(z) - vector[2].diff(x)))}",
            f"k part: ∂v2/∂x - ∂v1/∂y = ({show(vector[1].diff(x))}) - ({show(vector[0].diff(y))}) = {show(sp.simplify(vector[1].diff(x) - vector[0].diff(y)))}"]


def _operand(text):
    """A scalar in grad(...) nested inside div or curl: 'grad(x^3+y^3)' gives the gradient vector."""
    match = re.fullmatch(r"\s*(?:grad|∇)\s*\(?(.+?)\)?\s*", text)
    return _grad(read_expression(match[1])) if match else read_vector(text)


def gradient(text, at=None):
    scalar = read_expression(text)
    vector = _grad(scalar)
    answer = [f"grad f = {show_vector(vector)}"]
    steps = [f"grad f = (∂f/∂x)i + (∂f/∂y)j + (∂f/∂z)k, with f = {show(scalar)}"]
    steps += [f"∂f/∂{v} = {show(entry)}" for v, entry in zip(XYZ, vector)]
    if at is not None:
        answer.append(f"At {tuple(at)}: {show_vector(vector.subs(dict(zip(XYZ, at))))}")
        steps.append(f"Put in the point {tuple(at)}: {show_vector(vector.subs(dict(zip(XYZ, at))))}")
    return Result("Gradient", answer, steps)


def divergence(text, at=None):
    vector = _operand(text)
    value = sp.simplify(_div(vector))
    answer = [f"div v = {show(value)}"]
    steps = [f"v = {show_vector(vector)}, and div v = ∂v1/∂x + ∂v2/∂y + ∂v3/∂z"]
    steps += [f"∂v{n + 1}/∂{XYZ[n]} = ∂({show(vector[n])})/∂{XYZ[n]} = {show(vector[n].diff(XYZ[n]))}" for n in range(3)]
    steps.append(f"Add them: {show(value)}")
    if at is not None:
        answer.append(f"At {tuple(at)}: {show(value.subs(dict(zip(XYZ, at))))}")
        steps.append(f"Put in the point {tuple(at)}: {show(value.subs(dict(zip(XYZ, at))))}")
    return Result("Divergence", answer, steps)


def curl(text, times=1):
    vector = _operand(text)
    steps = [f"v = {show_vector(vector)}"]
    for _ in range(times):
        steps += _curl_lines(vector)
        vector = _curl(vector)
    label = "curl v" if times == 1 else "curl(curl v)"
    return Result("Curl", [f"{label} = {show_vector(vector)}"], steps)


def field_type(text):
    vector = _operand(text)
    div, cur = sp.simplify(_div(vector)), _curl(vector).applyfunc(sp.simplify)
    answer = [f"div v = {show(div)}: {'solenoidal' if div == 0 else 'not solenoidal'}",
              f"curl v = {show_vector(cur)}: {'irrotational' if cur == sp.zeros(3, 1) else 'not irrotational'}"]
    steps = [f"v = {show_vector(vector)}", "Solenoidal means div v = 0",
             *[f"∂v{n + 1}/∂{XYZ[n]} = {show(vector[n].diff(XYZ[n]))}" for n in range(3)], f"Adding them, div v = {show(div)}",
             "Irrotational means curl v = 0", *_curl_lines(vector)]
    return Result("Solenoidal and irrotational", answer, steps)


def potential(text):
    vector = _operand(text)
    cur = _curl(vector).applyfunc(sp.simplify)
    steps = [f"F = {show_vector(vector)}. F is conservative exactly when curl F = 0", *_curl_lines(vector)]
    if cur != sp.zeros(3, 1):
        return Result("Conservative field", [f"curl F = {show_vector(cur)}, so F is not conservative"], steps)
    x, y, z = XYZ
    phi = sp.integrate(vector[0], x)
    steps.append(f"Integrate F1 by x: φ = {show(phi)} + a function of y and z")
    extra = sp.integrate(sp.simplify(vector[1] - phi.diff(y)), y)
    phi += extra
    steps.append(f"∂φ/∂y must equal F2, which adds {show(extra)}: φ = {show(phi)} + a function of z")
    extra = sp.integrate(sp.simplify(vector[2] - phi.diff(z)), z)
    phi += extra
    steps.append(f"∂φ/∂z must equal F3, which adds {show(extra)}: φ = {show(sp.expand(phi))} + c")
    steps.append(f"Check: grad φ = {show_vector(_grad(phi))}, which is F")
    return Result("Conservative field", ["curl F = 0, so F is conservative", f"φ = {show(sp.expand(phi))} + c"], steps)


def directional(scalar_text, at, along_text):
    scalar = read_expression(scalar_text)
    direction = read_vector(along_text)
    gradient_there = _grad(scalar).subs(dict(zip(XYZ, at)))
    value = gradient_there.dot(direction) / direction.norm()
    return Result("Directional derivative", [f"Directional derivative = {show(sp.simplify(value))}"],
                  [f"Find grad f: {show_vector(_grad(scalar))}", f"At the point {tuple(at)}: grad f = {show_vector(gradient_there)}",
                   f"The direction a = {show_vector(direction)}, with size |a| = {show_plain(direction.norm())}",
                   f"Directional derivative = (grad f · a) ÷ |a| = ({show_plain(gradient_there.dot(direction))}) ÷ {show_plain(direction.norm())} = {show(sp.simplify(value))}"])


def motion(curve_text, at=None, along=None):
    """Velocity, acceleration, their sizes at a time and their components along a direction."""
    r = read_vector(curve_text)
    velocity, acceleration = r.diff(T), r.diff(T, 2)
    answer = [f"Velocity = {show_vector(velocity)}", f"Acceleration = {show_vector(acceleration)}"]
    steps = [f"r = {show_vector(r)}", f"Velocity is dr/dt: {show_vector(velocity)}", f"Acceleration is dv/dt: {show_vector(acceleration)}"]
    if at is not None:
        v, a = velocity.subs(T, at), acceleration.subs(T, at)
        steps += [f"At t = {show_plain(at)}: v = {show_vector(v)}", f"At t = {show_plain(at)}: a = {show_vector(a)}"]
        answer += [f"At t = {show_plain(at)}: |v| = {show(sp.simplify(v.norm()))}, |a| = {show(sp.simplify(a.norm()))}"]
        steps.append(f"Sizes, the square root of the sum of the squares: |v| = {show(sp.simplify(v.norm()))}, |a| = {show(sp.simplify(a.norm()))}")
        if along is not None:
            unit = read_vector(along)
            size = unit.norm()
            unit = unit / size
            steps.append(f"Divide the direction by its size, {show_plain(size)}, to get a unit vector n. A component is the dot product with n: v·n = {show(sp.simplify(v.dot(unit)))}, a·n = {show(sp.simplify(a.dot(unit)))}")
            answer += [f"Component of velocity along the direction = {show(sp.simplify(v.dot(unit)))}",
                       f"Component of acceleration along the direction = {show(sp.simplify(a.dot(unit)))}"]
    return Result("Velocity and acceleration", answer, steps)


def tangent(curve_text, times=()):
    """The unit tangent and the angle between the tangents at two times."""
    r = read_vector(curve_text)
    velocity = r.diff(T)
    length = sp.sqrt(sum(entry ** 2 for entry in velocity))
    answer = [f"Unit tangent = ({show_vector(velocity)}) / {show(sp.simplify(length))}"]
    steps = [f"r = {show_vector(r)}", f"The tangent is dr/dt: {show_vector(velocity)}", f"Its size: {show(sp.simplify(length))}", "Divide the tangent by its size for the unit tangent"]
    if len(times) == 2:
        a, b = (velocity.subs(T, t) for t in times)
        cosine = sp.simplify(a.dot(b) / (a.norm() * b.norm()))
        answer.append(f"cos θ = {show_plain(cosine)}, θ = {decimal(sp.acos(cosine) * 180 / sp.pi, 2)}°")
        steps += [f"Tangent at t = {show_plain(t)}: {show_vector(velocity.subs(T, t))}" for t in times]
        steps.append(f"cos θ = (a·b) ÷ (|a||b|) = ({show_plain(a.dot(b))}) ÷ (({show_plain(a.norm())})({show_plain(b.norm())})) = {show_plain(cosine)}")
    return Result("Unit tangent", answer, steps)


def surfaces_angle(first, second, at):
    """The angle between two surfaces at a point is the angle between their normals."""
    normals = []
    for part in (first, second):
        left, right = read_equation(part)
        normals.append(_grad(left - right).subs(dict(zip(XYZ, at))))
    cosine = sp.simplify(normals[0].dot(normals[1]) / (normals[0].norm() * normals[1].norm()))
    return Result("Angle between surfaces", [f"cos θ = {show_plain(cosine)}", f"θ = {decimal(sp.acos(cosine) * 180 / sp.pi, 2)}°"],
                  ["The angle between two surfaces is the angle between their normals, and grad f is a normal",
                   *[f"Normal {n + 1}, grad f at the point: {show_vector(v)}, of size {show_plain(v.norm())}" for n, v in enumerate(normals)],
                   f"cos θ = (n1·n2) ÷ (|n1||n2|) = ({show_plain(normals[0].dot(normals[1]))}) ÷ (({show_plain(normals[0].norm())})({show_plain(normals[1].norm())})) = {show_plain(cosine)}"])


# ---------- integrals ----------

def _numeric_integral(integral):
    return sp.N(integral, 12)


def integral(text):
    """An integral worked out exactly if SymPy can, else numerically, and said to be numeric."""
    expression = read_expression(text)
    integrals = expression.atoms(sp.Integral)
    if not integrals:
        raise ValueError("No integral was found")

    def exact():
        return expression.doit()

    try:
        value = time_limited(exact, 25)
    except TimeoutError:
        value = None
    if value is None or value.has(sp.Integral):
        if expression.free_symbols - set().union(*(i.variables for i in integrals)):
            raise ValueError("This integral has no simple exact form")
        number = _numeric_integral(expression)
        guess = sp.nsimplify(number, [sp.pi, sp.sqrt(2)], tolerance=1e-10)
        return Result("Definite integral", [f"≈ {decimal(number, 6)}"], [f"The integral: {show(expression)}", "No exact formula for it could be found in time, so the area is added up numerically",
                                                                          f"That gives {decimal(number, 6)}, which looks like {show_plain(guess)}"])
    value = sp.simplify(value)
    return Result("Integral", [show(value)], explain_integral(expression, value))


def _derivatives_as_written(expression):
    """∂u/∂x for SymPy's Derivative(u, x)."""
    text = re.sub(r"Derivative\((\w+), \((\w+), (\d)\)\)", r"∂^\3\1/∂\2^\3", show(expression))
    text = re.sub(r"Derivative\((\w+), (\w+), (\w+)\)", r"∂²\1/∂\2∂\3", text)
    return re.sub(r"Derivative\((\w+), (\w+)\)", r"∂\1/∂\2", text)


def prove(statement, where=None):
    """'lhs = rhs', with 'where u = ...': checked exactly, else numerically at a test point."""
    statement = re.sub(r"\bd([a-z])/d([a-z])\b", r"∂\1/∂\2", statement)
    left, right = read_equation(statement)
    difference = left - right
    steps = [f"Left side: {_derivatives_as_written(left)}", f"Right side: {_derivatives_as_written(right)}"]
    if where:
        name, body = where.split("=", 1)
        difference = difference.subs(sp.Symbol(name.strip()), read_expression(body))
        steps.append(f"Put in {name.strip()} = {show(read_expression(body))}")
    difference = difference.doit()
    steps.append(f"Work out the derivatives in left side minus right side: {show(difference)}")
    simplified = time_limited(lambda: sp.simplify(difference), 30)
    steps.append(f"Simplify: {show(simplified)}")
    if simplified == 0:
        return Result("Proof by working both sides", ["True: both sides are equal"], steps + ["The difference is 0, so the two sides are equal"])
    letters = list(simplified.free_symbols)
    if not letters:
        close = abs(float(sp.N(simplified))) < 1e-9
        return Result("Proof by working both sides", ["True: both sides are equal" if close else f"False: the two sides differ by {show(simplified)}"], steps)
    test = {s: sp.Rational(3 + n, 7 + 2 * n) for n, s in enumerate(letters)}
    close = abs(complex(sp.N(simplified.subs(test)))) < 1e-9
    if close:
        return Result("Proof by working both sides", ["True at the test values, though the algebra did not fully simplify"], steps + [f"Test values: {test}. There the difference comes to 0"])
    return Result("Proof by working both sides", [f"False: the two sides differ by {show(simplified)}"], steps)
