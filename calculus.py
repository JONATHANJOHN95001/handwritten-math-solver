"""Calculus: differential equations, partial derivatives, vector calculus and integrals."""
import re

import sympy as sp

from results import Result, decimal, parts_of, show, show_matrix, show_plain, time_limited
from solver import X, read_equation, read_expression

y_symbol, D = sp.Symbol("y"), sp.Symbol("D")
T = sp.Symbol("t")
VARIABLES = ("x", "y", "z", "t", "u", "v", "w", "r", "θ")


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


def solve_ode(text):
    parts = parts_of(text)
    expression, unknown, var = read_ode(parts[0])
    conditions = _initial_values(parts[1:], unknown, var)
    order, homogeneous = _classify(expression, unknown, var)
    steps = [f"Order {order}, {'homogeneous' if homogeneous else 'non-homogeneous'}"]
    auxiliary = _auxiliary(expression, unknown, var)
    if auxiliary:
        poly, m = auxiliary
        roots = sp.roots(sp.Poly(poly, m))
        steps.append(f"Auxiliary equation: {show(poly)} = 0")
        if roots:
            steps.append("Roots: " + ", ".join(show_plain(r) for r, n in roots.items() for _ in range(n)))

    def work():
        if _legendre_form(expression, unknown, var) and not auxiliary:
            return _solve_legendre(expression, unknown, var)
        return sp.dsolve(expression, unknown, ics=conditions or None)

    solution = time_limited(work, 40)
    solutions = solution if isinstance(solution, list) else [solution]
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
    steps = [f"Eigenvalues: {', '.join(show_plain(v) for v, n in values for _ in range(n))}", f"P =\n{show_matrix(p)}", f"D =\n{show_matrix(d)}"]
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
    answer, steps = [], []
    for v in letters:
        answer.append(f"∂/∂{v} = {show(sp.simplify(body.diff(v)))}")
    for i, a in enumerate(letters):
        for b in letters[i:]:
            label = f"∂²/∂{a}²" if a == b else f"∂²/∂{a}∂{b}"
            steps.append(f"{label} = {show(sp.simplify(body.diff(a, b)))}")
    return Result("Partial derivatives", answer, ["Second order:"] + steps)


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
    return Result("Total derivative", answer, ["The chain rule and direct substitution give the same result"])


def implicit_derivative(body_text, constraint_text):
    """du/dx of u(x,y) where y depends on x through a constraint."""
    y = sp.Symbol("y")
    body = read_expression(body_text)
    left, right = read_equation(constraint_text)
    constraint = left - right
    dydx = -constraint.diff(X) / constraint.diff(y)
    return Result("Total derivative with a constraint", [f"du/dx = {show(sp.simplify(body.diff(X) + body.diff(y) * dydx))}"],
                  [f"dy/dx = {show(sp.simplify(dydx))}"])


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
    return Result("Jacobian", answer, [f"Jacobian matrix:\n{show_matrix(matrix)}"])


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
    return Result("Jacobian product", [f"J·J' = {show(product)}"], [f"J = {show(sp.simplify(forward))}", f"J' = {show(sp.simplify(back))}"])


def extrema(text):
    """Stationary points of f(x,y) and what kind each one is. Letters other than x, y are constants, taken as positive."""
    body = _function_body(text)
    letters = _variables(body)
    constants = body.free_symbols - set(letters)
    positive = {c: sp.Symbol(c.name, positive=True) for c in constants}
    body = body.subs(positive)
    points = sp.solve([body.diff(v) for v in letters], letters, dict=True)
    answer, steps = [], [f"∂/∂{v} = {show(body.diff(v))}" for v in letters]
    if len(letters) != 2:
        raise ValueError("This method handles functions of two variables")
    a, b = letters
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
        steps.append(f"At {place}: r = {show(r)}, s = {show(s)}, t = {show(t)}, rt - s² = {show(d)}")
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
    return Result("Lagrange multipliers", answer, [f"Stationary points found: {len(real)}"])


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


def _operand(text):
    """A scalar in grad(...) nested inside div or curl: 'grad(x^3+y^3)' gives the gradient vector."""
    match = re.fullmatch(r"\s*(?:grad|∇)\s*\(?(.+?)\)?\s*", text)
    return _grad(read_expression(match[1])) if match else read_vector(text)


def gradient(text, at=None):
    scalar = read_expression(text)
    vector = _grad(scalar)
    answer = [f"grad f = {show_vector(vector)}"]
    if at is not None:
        answer.append(f"At {tuple(at)}: {show_vector(vector.subs(dict(zip(XYZ, at))))}")
    return Result("Gradient", answer, [])


def divergence(text, at=None):
    value = sp.simplify(_div(_operand(text)))
    answer = [f"div v = {show(value)}"]
    if at is not None:
        answer.append(f"At {tuple(at)}: {show(value.subs(dict(zip(XYZ, at))))}")
    return Result("Divergence", answer, [])


def curl(text, times=1):
    vector = _operand(text)
    for _ in range(times):
        vector = _curl(vector)
    label = "curl v" if times == 1 else "curl(curl v)"
    return Result("Curl", [f"{label} = {show_vector(vector)}"], [])


def field_type(text):
    vector = _operand(text)
    div, cur = sp.simplify(_div(vector)), _curl(vector).applyfunc(sp.simplify)
    answer = [f"div v = {show(div)}: {'solenoidal' if div == 0 else 'not solenoidal'}",
              f"curl v = {show_vector(cur)}: {'irrotational' if cur == sp.zeros(3, 1) else 'not irrotational'}"]
    return Result("Solenoidal and irrotational", answer, [])


def potential(text):
    vector = _operand(text)
    cur = _curl(vector).applyfunc(sp.simplify)
    if cur != sp.zeros(3, 1):
        return Result("Conservative field", [f"curl F = {show_vector(cur)}, so F is not conservative"], [])
    x, y, z = XYZ
    phi = sp.integrate(vector[0], x)
    phi += sp.integrate(sp.simplify(vector[1] - phi.diff(y)), y)
    phi += sp.integrate(sp.simplify(vector[2] - phi.diff(z)), z)
    return Result("Conservative field", ["curl F = 0, so F is conservative", f"φ = {show(sp.expand(phi))} + c"],
                  ["Check: grad φ = F"])


def directional(scalar_text, at, along_text):
    scalar = read_expression(scalar_text)
    direction = read_vector(along_text)
    gradient_there = _grad(scalar).subs(dict(zip(XYZ, at)))
    value = gradient_there.dot(direction) / direction.norm()
    return Result("Directional derivative", [f"Directional derivative = {show(sp.simplify(value))}"],
                  [f"grad f at the point = {show_vector(gradient_there)}", f"|a| = {show(direction.norm())}"])


def motion(curve_text, at=None, along=None):
    """Velocity, acceleration, their sizes at a time and their components along a direction."""
    r = read_vector(curve_text)
    velocity, acceleration = r.diff(T), r.diff(T, 2)
    answer = [f"Velocity = {show_vector(velocity)}", f"Acceleration = {show_vector(acceleration)}"]
    steps = []
    if at is not None:
        v, a = velocity.subs(T, at), acceleration.subs(T, at)
        steps += [f"At t = {show_plain(at)}: v = {show_vector(v)}", f"At t = {show_plain(at)}: a = {show_vector(a)}"]
        answer += [f"At t = {show_plain(at)}: |v| = {show(sp.simplify(v.norm()))}, |a| = {show(sp.simplify(a.norm()))}"]
        if along is not None:
            unit = read_vector(along)
            unit = unit / unit.norm()
            answer += [f"Component of velocity along the direction = {show(sp.simplify(v.dot(unit)))}",
                       f"Component of acceleration along the direction = {show(sp.simplify(a.dot(unit)))}"]
    return Result("Velocity and acceleration", answer, steps)


def tangent(curve_text, times=()):
    """The unit tangent and the angle between the tangents at two times."""
    r = read_vector(curve_text)
    velocity = r.diff(T)
    length = sp.sqrt(sum(entry ** 2 for entry in velocity))
    answer = [f"Unit tangent = ({show_vector(velocity)}) / {show(sp.simplify(length))}"]
    steps = []
    if len(times) == 2:
        a, b = (velocity.subs(T, t) for t in times)
        cosine = sp.simplify(a.dot(b) / (a.norm() * b.norm()))
        answer.append(f"cos θ = {show_plain(cosine)}, θ = {decimal(sp.acos(cosine) * 180 / sp.pi, 2)}°")
        steps = [f"Tangent at t = {show_plain(t)}: {show_vector(velocity.subs(T, t))}" for t in times]
    return Result("Unit tangent", answer, steps)


def surfaces_angle(first, second, at):
    """The angle between two surfaces at a point is the angle between their normals."""
    normals = []
    for part in (first, second):
        left, right = read_equation(part)
        normals.append(_grad(left - right).subs(dict(zip(XYZ, at))))
    cosine = sp.simplify(normals[0].dot(normals[1]) / (normals[0].norm() * normals[1].norm()))
    return Result("Angle between surfaces", [f"cos θ = {show_plain(cosine)}", f"θ = {decimal(sp.acos(cosine) * 180 / sp.pi, 2)}°"],
                  [f"Normal {n + 1} = {show_vector(v)}" for n, v in enumerate(normals)])


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
        return Result("Definite integral", [f"≈ {decimal(number, 6)}"], [f"Worked out numerically. It looks like {show_plain(guess)}"])
    value = sp.simplify(value)
    return Result("Integral", [show(value)], [])


def prove(statement, where=None):
    """'lhs = rhs', with 'where u = ...': checked exactly, else numerically at a test point."""
    statement = re.sub(r"\bd([a-z])/d([a-z])\b", r"∂\1/∂\2", statement)
    left, right = read_equation(statement)
    difference = left - right
    steps = []
    if where:
        name, body = where.split("=", 1)
        difference = difference.subs(sp.Symbol(name.strip()), read_expression(body))
    difference = difference.doit()
    simplified = time_limited(lambda: sp.simplify(difference), 30)
    if simplified == 0:
        return Result("Proof by working both sides", ["True: both sides are equal"], ["Left side minus right side simplifies to 0"])
    letters = list(simplified.free_symbols)
    if not letters:
        close = abs(float(sp.N(simplified))) < 1e-9
        return Result("Proof by working both sides", ["True: both sides are equal" if close else f"False: the two sides differ by {show(simplified)}"], steps)
    test = {s: sp.Rational(3 + n, 7 + 2 * n) for n, s in enumerate(letters)}
    close = abs(complex(sp.N(simplified.subs(test)))) < 1e-9
    if close:
        return Result("Proof by working both sides", ["True at the test values, though the algebra did not fully simplify"], [f"Test values: {test}"])
    return Result("Proof by working both sides", [f"False: the two sides differ by {show(simplified)}"], steps)
