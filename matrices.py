"""Matrix Algebra: rank, determinant, eigenvalues, diagonalisation, linear systems and Gauss-Seidel."""
import re

import sympy as sp

from results import Result, decimal, parts_of, show, show_matrix, show_plain
from solver import read_equation

LAM = sp.Symbol("λ")
UNKNOWN = re.compile(r"^(x|y|z|w|x\d+)$")


def _whole_numbers(vector):
    """An eigenvector scaled so that its entries are whole numbers where possible."""
    denominators = [sp.fraction(entry)[1] for entry in vector]
    scale = sp.ilcm(*denominators) if all(d.is_Integer for d in denominators) else 1
    return vector * scale


def _list(values):
    return ", ".join(show_plain(v) for v in values)


def rank(matrix):
    r = matrix.rank()
    steps = [f"Matrix:\n{show_matrix(matrix)}", f"Row echelon form:\n{show_matrix(matrix.echelon_form())}",
             f"Non-zero rows in the echelon form: {r}"]
    return Result("Rank of a matrix", [f"Rank = {r}"], steps)


def echelon(matrix):
    return Result("Row echelon form", [show_matrix(matrix.echelon_form())], [f"Matrix:\n{show_matrix(matrix)}"])


def determinant(matrix):
    return Result("Determinant", [f"det = {show(matrix.det())}"], [f"Matrix:\n{show_matrix(matrix)}"])


def inverse(matrix):
    if matrix.det() == 0:
        raise ValueError("The determinant is 0, so this matrix has no inverse")
    return Result("Inverse of a matrix", [show_matrix(matrix.inv())], [f"det = {show(matrix.det())}"])


def characteristic(matrix):
    poly = matrix.charpoly(LAM).as_expr()
    return Result("Characteristic polynomial", [f"{show(poly)} = 0"], [])


def eigen(matrix):
    poly = matrix.charpoly(LAM).as_expr()
    pairs = sorted(matrix.eigenvects(), key=lambda item: float(sp.re(sp.N(item[0]))))
    answer, steps = [], [f"Characteristic equation: {show(poly)} = 0"]
    for value, multiplicity, vectors in pairs:
        repeat = f" (repeated {multiplicity} times)" if multiplicity > 1 else ""
        answer.append(f"λ = {show_plain(value)}{repeat}: eigenvector " + ", ".join(
            "(" + _list(_whole_numbers(v)) + ")" for v in vectors))
    return Result("Eigenvalues and eigenvectors", answer, steps)


def eigenvalues(matrix):
    full = eigen(matrix)
    values = []
    for value, multiplicity in sorted(matrix.eigenvals().items(), key=lambda item: float(sp.re(sp.N(item[0])))):
        values += [value] * multiplicity
    return Result("Eigenvalues", [f"λ = {_list(values)}"], full.steps)


def diagonalize(matrix, power=None):
    if not matrix.is_diagonalizable():
        raise ValueError("This matrix cannot be diagonalised: it does not have enough independent eigenvectors")
    p, d = matrix.diagonalize(sort=True)
    answer = [f"P =\n{show_matrix(p)}", f"D =\n{show_matrix(d)}"]
    steps = eigen(matrix).answer + ["A = P D P⁻¹"]
    if power is not None:
        answer.append(f"A^{power} =\n{show_matrix((p * d ** power * p.inv()).applyfunc(sp.simplify))}")
        steps.append(f"A^{power} = P D^{power} P⁻¹")
    return Result("Diagonalisation", answer, steps)


def matrix_power(matrix, power):
    return Result("Power of a matrix", [f"A^{power} =\n{show_matrix((matrix ** power).applyfunc(sp.simplify))}"], [])


def summary(matrix):
    """Everything the MATLAB questions ask for: determinant, rank, size, characteristic polynomial, eigenvalues."""
    answer = [f"Determinant = {show(matrix.det())}", f"Rank = {matrix.rank()}", f"Size = {matrix.rows} × {matrix.cols}"]
    if matrix.is_square:
        answer.append(f"Characteristic polynomial: {show(matrix.charpoly(LAM).as_expr())}")
        answer += eigenvalues(matrix).answer
    return Result("Matrix summary", answer, [])


def _unknowns(expressions):
    letters = set().union(*(e.free_symbols for e in expressions))
    return sorted((s for s in letters if UNKNOWN.match(s.name)), key=lambda s: (s.name[0] == "w", s.name))


def _equations(parts):
    expressions = []
    for part in parts:
        left, right = read_equation(part)
        expressions.append(left - right)
    return expressions


def solve_system(text):
    """Consistency test and solution of equations given as 'x+y=3; x-y=1'."""
    expressions = _equations(parts_of(text))
    letters = _unknowns(expressions)
    try:
        coefficients, constants = sp.linear_eq_to_matrix(expressions, letters)
    except sp.solvers.solveset.NonlinearError:
        return _nonlinear(expressions, letters)
    augmented = coefficients.row_join(constants)
    r_a, r_b = coefficients.rank(), augmented.rank()
    steps = [f"Coefficient matrix A:\n{show_matrix(coefficients)}", f"Augmented matrix [A|B] in echelon form:\n{show_matrix(augmented.echelon_form())}",
             f"rank(A) = {r_a}, rank([A|B]) = {r_b}, unknowns = {len(letters)}"]
    if r_a != r_b:
        return Result("Linear system", ["The system is inconsistent: it has no solution"], steps)
    solution = next(iter(sp.linsolve((coefficients, constants), *letters)))
    lines = [f"{name} = {show_plain(value)}" for name, value in zip(letters, solution)]
    if r_a == len(letters):
        kind = "consistent, with a unique solution"
    else:
        kind = f"consistent, with infinitely many solutions ({len(letters) - r_a} free unknown)"
    return Result("Linear system", [f"The system is {kind}"] + lines, steps)


def _nonlinear(expressions, letters):
    solutions = sp.solve(expressions, letters, dict=True)
    if not solutions:
        return Result("System of equations", ["No solution was found"], [])
    return Result("System of equations", [", ".join(f"{k} = {show_plain(v)}" for k, v in s.items()) for s in solutions], [])


def gauss_seidel(text, iterations=4, start=None, places=4):
    """Gauss-Seidel: the rows are first put in the order that makes each diagonal entry largest."""
    expressions = _equations(parts_of(text))
    letters = _unknowns(expressions)
    coefficients, constants = sp.linear_eq_to_matrix(expressions, letters)
    n = len(letters)
    order, free = [], list(range(n))
    for column in range(n):
        best = max(free, key=lambda row: abs(coefficients[row, column]))
        order.append(best)
        free.remove(best)
    a = [[float(coefficients[row, c]) for c in range(n)] for row in order]
    b = [float(constants[row]) for row in order]
    values = [float(v) for v in start] if start else [0.0] * n
    steps = ["Rows arranged so that each diagonal entry is the largest: " + "; ".join(
        show(sum(coefficients[row, c] * letters[c] for c in range(n))) + f" = {show_plain(constants[row])}" for row in order)]
    steps.append("Start: " + ", ".join(f"{name} = {decimal(v, places)}" for name, v in zip(letters, values)))
    for number in range(1, iterations + 1):
        for i in range(n):
            if a[i][i] == 0:
                raise ValueError("A diagonal entry is 0, so Gauss-Seidel cannot run on these equations")
            values[i] = (b[i] - sum(a[i][j] * values[j] for j in range(n) if j != i)) / a[i][i]
        steps.append(f"Iteration {number}: " + ", ".join(f"{name} = {decimal(v, places)}" for name, v in zip(letters, values)))
    answer = [f"{name} = {decimal(v, places)}" for name, v in zip(letters, values)]
    return Result(f"Gauss-Seidel method, {iterations} iterations", answer, steps)


def investigate(text):
    """For which values of the parameters does the system have a unique, infinite or no solution?"""
    expressions = _equations(parts_of(text))
    letters = _unknowns(expressions)
    parameters = sorted(set().union(*(e.free_symbols for e in expressions)) - set(letters), key=lambda s: s.name)
    if not parameters:
        return solve_system(text)
    coefficients, constants = sp.linear_eq_to_matrix(expressions, letters)
    if not coefficients.is_square:
        raise ValueError("This needs as many equations as unknowns")
    det = sp.factor(coefficients.det())
    steps = [f"Determinant of the coefficients: {show(det)}"]
    answer = []
    roots = [r for s in det.free_symbols for r in [(s, v) for v in sp.solve(det, s)]]
    if not roots:
        return Result("Investigating a system", [f"A unique solution for every value of {_list(parameters)}"], steps)
    condition = " and ".join(f"{s} ≠ {show_plain(v)}" for s, v in roots)
    answer.append(f"Unique solution: {condition}")
    for symbol, value in roots:
        reduced = coefficients.subs(symbol, value).row_join(constants.subs(symbol, value)).echelon_form()
        last = [reduced[r, -1] for r in range(reduced.rows) if all(reduced[r, c] == 0 for c in range(reduced.cols - 1))]
        steps.append(f"With {symbol} = {show_plain(value)}, echelon form:\n{show_matrix(reduced)}")
        conditions = [sp.solve(entry, others) for entry in last for others in entry.free_symbols]
        others = [(s, v) for entry in last for s in entry.free_symbols for v in sp.solve(entry, s)]
        if others:
            for other, other_value in others:
                answer.append(f"Infinitely many solutions: {symbol} = {show_plain(value)} and {other} = {show_plain(other_value)}")
                answer.append(f"No solution: {symbol} = {show_plain(value)} and {other} ≠ {show_plain(other_value)}")
        elif any(entry != 0 for entry in last):
            answer.append(f"No solution: {symbol} = {show_plain(value)}")
        else:
            answer.append(f"Infinitely many solutions: {symbol} = {show_plain(value)}")
        del conditions
    return Result("Investigating a system", answer, steps)
