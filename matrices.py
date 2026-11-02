"""Matrix Algebra: rank, determinant, eigenvalues, diagonalisation, linear systems and Gauss-Seidel."""
import re

import sympy as sp
from sympy.solvers.solveset import NonlinearError

from results import Result, decimal, parts_of, show, show_matrix, show_plain
from solver import read_equation

LAM = sp.Symbol("λ")
UNKNOWN = re.compile(r"^(x|y|z|w|x\d+)$")
POWERS_SHOWN = 8  # a matrix power up to this is multiplied out one step at a time; above it, by repeated squaring


def _whole_numbers(vector):
    """An eigenvector scaled so that its entries are whole numbers where possible."""
    denominators = [sp.fraction(entry)[1] for entry in vector]
    scale = sp.ilcm(*denominators) if all(d.is_Integer for d in denominators) else 1
    return vector * scale


def _list(values):
    return ", ".join(show_plain(v) for v in values)


def _row_operation(target, factor, pivot):
    """'R2 → R2 − 2·R1'."""
    negative = factor.could_extract_minus_sign()
    size = -factor if negative else factor
    written = "" if size == 1 else f"{show_plain(size)}·" if size.is_Rational else f"({show_plain(size)})·"
    return f"R{target + 1} → R{target + 1} {'+' if negative else '−'} {written}R{pivot + 1}"


def _row_reduce(matrix):
    """Echelon form, one row operation at a time. Returns the form, one line per operation and how many rows were swapped."""
    work = matrix.as_mutable()
    lines, swaps, row = [], 0, 0
    for column in range(work.cols):
        if row >= work.rows:
            break
        found = next((r for r in range(row, work.rows) if work[r, column] != 0), None)
        if found is None:
            continue
        if found != row:
            work.row_swap(found, row)
            swaps += 1
            lines.append(f"Swap R{row + 1} and R{found + 1}, to get a non-zero entry at the top of column {column + 1}:\n{show_matrix(work)}")
        for below in range(row + 1, work.rows):
            if work[below, column] != 0:
                factor = sp.simplify(work[below, column] / work[row, column])
                for c in range(work.cols):
                    work[below, c] = sp.simplify(work[below, c] - factor * work[row, c])
                lines.append(f"{_row_operation(below, factor, row)}, to clear column {column + 1} below the pivot {show_plain(work[row, column])}:\n{show_matrix(work)}")
        row += 1
    return work.as_immutable(), lines, swaps


def rank(matrix):
    r = matrix.rank()
    form, operations, _ = _row_reduce(matrix)
    steps = [f"Matrix:\n{show_matrix(matrix)}", "Reduce it to row echelon form:", *operations,
             f"Count the rows that are not all zero: {r}, so the rank is {r}"]
    return Result("Rank of a matrix", [f"Rank = {r}"], steps)


def echelon(matrix):
    form, operations, _ = _row_reduce(matrix)
    return Result("Row echelon form", [show_matrix(form)], [f"Matrix:\n{show_matrix(matrix)}", *operations,
                                                            "Every entry below the pivots is now 0, so this is row echelon form"])


def _two_by_two(matrix):
    """'(1)(4) − (2)(3) = -2'."""
    a, b, c, d = matrix
    return f"({show_plain(a)})({show_plain(d)}) − ({show_plain(b)})({show_plain(c)}) = {show_plain(matrix.det())}"


def determinant_lines(matrix):
    if not matrix.is_square:
        raise ValueError("Only a square matrix has a determinant")
    if matrix.rows == 1:
        return [f"A 1 × 1 matrix has its one entry as determinant: {show_plain(matrix[0, 0])}"]
    if matrix.rows == 2:
        return [f"For a 2 × 2 matrix, det = ad − bc = {_two_by_two(matrix)}"]
    if matrix.rows == 3:
        lines, terms = ["Expand along the first row: det = a11·M11 − a12·M12 + a13·M13, where each M is the determinant left after crossing out that row and column"], []
        for column in range(3):
            minor = matrix.minor_submatrix(0, column)
            lines.append(f"M1{column + 1} = {_two_by_two(minor)}")
            terms.append(f"{'' if column == 0 else ' − ' if column == 1 else ' + '}({show_plain(matrix[0, column])})({show_plain(minor.det())})")
        return lines + [f"det = {''.join(terms)} = {show_plain(matrix.det())}"]
    form, operations, swaps = _row_reduce(matrix)
    diagonal = [form[k, k] for k in range(form.rows)]
    sign = "-" if swaps % 2 else ""
    return ["Reduce it to a triangle: adding a multiple of one row to another leaves the determinant alone, and each swap changes its sign", *operations,
            f"det = {sign}(product of the diagonal) = {sign}" + " × ".join(f"({show_plain(v)})" for v in diagonal) + f" = {show_plain(matrix.det())}"]


def determinant(matrix):
    return Result("Determinant", [f"det = {show(matrix.det())}"], [f"Matrix:\n{show_matrix(matrix)}", *determinant_lines(matrix)])


def inverse(matrix):
    if matrix.det() == 0:
        raise ValueError("The determinant is 0, so this matrix has no inverse")
    steps = [f"Matrix:\n{show_matrix(matrix)}", *determinant_lines(matrix),
             f"The determinant is not 0, so the inverse exists: A⁻¹ = adj(A) ÷ det(A)",
             f"Cofactor of each entry:\n{show_matrix(matrix.cofactor_matrix())}",
             f"adj(A) is that matrix with rows and columns swapped:\n{show_matrix(matrix.adjugate())}",
             f"Divide every entry by the determinant, {show_plain(matrix.det())}:\n{show_matrix(matrix.inv())}"]
    return Result("Inverse of a matrix", [show_matrix(matrix.inv())], steps)


def _characteristic_lines(matrix):
    poly = matrix.charpoly(LAM).as_expr()
    shifted = matrix - LAM * sp.eye(matrix.rows)
    factored = sp.factor(poly)
    lines = [f"Subtract λ from each diagonal entry, A − λI:\n{show_matrix(shifted)}", f"Set its determinant to 0: {show(poly)} = 0"]
    if factored.is_Mul or factored.is_Pow:
        lines.append(f"Factor it: {show(factored)} = 0")
    return lines


def characteristic(matrix):
    poly = matrix.charpoly(LAM).as_expr()
    return Result("Characteristic polynomial", [f"{show(poly)} = 0"], [f"Matrix:\n{show_matrix(matrix)}", *_characteristic_lines(matrix)])


def eigen(matrix):
    pairs = sorted(matrix.eigenvects(), key=lambda item: float(sp.re(sp.N(item[0]))))
    answer, steps = [], [f"Matrix:\n{show_matrix(matrix)}", *_characteristic_lines(matrix),
                         "Its roots are the eigenvalues: " + _list([value for value, _, _ in pairs])]
    for value, multiplicity, vectors in pairs:
        repeat = f" (repeated {multiplicity} times)" if multiplicity > 1 else ""
        found = ", ".join("(" + _list(_whole_numbers(v)) + ")" for v in vectors)
        answer.append(f"λ = {show_plain(value)}{repeat}: eigenvector " + found)
        if value.is_Rational and all(entry.is_Rational for entry in matrix):
            form, _, _ = _row_reduce(matrix - value * sp.eye(matrix.rows))
            steps.append(f"For λ = {show_plain(value)}, solve (A − λI)v = 0. Row reduced, A − λI is:\n{show_matrix(form)}")
            steps.append(f"The rows that are left give the eigenvector {found}")
        else:  # the roots are not simple numbers, so row operations on them could not be read or even told from zero
            steps.append(f"For λ = {show_plain(value)}, solving (A − λI)v = 0 gives the eigenvector {found}")
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
    steps = eigen(matrix).steps + [f"Put the eigenvectors in the columns of P:\n{show_matrix(p)}",
                                   f"Put the eigenvalues on the diagonal of D, in the same order:\n{show_matrix(d)}", "Then A = P D P⁻¹",
                                   f"P⁻¹:\n{show_matrix(p.inv())}"]
    if power is not None:
        answer.append(f"A^{power} =\n{show_matrix((p * d ** power * p.inv()).applyfunc(sp.simplify))}")
        steps += [f"A^{power} = P D^{power} P⁻¹, and a power of D only raises its diagonal entries:\n{show_matrix(d ** power)}",
                  f"Multiply P, D^{power} and P⁻¹:\n{show_matrix((p * d ** power * p.inv()).applyfunc(sp.simplify))}"]
    return Result("Diagonalisation", answer, steps)


def matrix_power(matrix, power):
    steps = [f"A:\n{show_matrix(matrix)}"]
    current, reached = matrix, 1
    if power <= POWERS_SHOWN:
        while reached < power:
            current, reached = current * matrix, reached + 1
            steps.append(f"A^{reached} = A^{reached - 1} × A:\n{show_matrix(current.applyfunc(sp.simplify))}")
    else:
        while reached * 2 <= power:
            current, reached = current * current, reached * 2
            steps.append(f"A^{reached} = A^{reached // 2} × A^{reached // 2}:\n{show_matrix(current.applyfunc(sp.simplify))}")
        steps.append(f"Multiply the powers that add up to {power}: {' + '.join(str(2 ** k) for k in range(power.bit_length()) if power >> k & 1)}")
    return Result("Power of a matrix", [f"A^{power} =\n{show_matrix((matrix ** power).applyfunc(sp.simplify))}"], steps)


def summary(matrix):
    """Everything the MATLAB questions ask for: determinant, rank, size, characteristic polynomial, eigenvalues."""
    answer = [f"Determinant = {show(matrix.det())}", f"Rank = {matrix.rank()}", f"Size = {matrix.rows} × {matrix.cols}"]
    steps = [f"Matrix, with {matrix.rows} rows and {matrix.cols} columns:\n{show_matrix(matrix)}", "Rank:", *rank(matrix).steps[1:]]
    if matrix.is_square:
        answer.append(f"Characteristic polynomial: {show(matrix.charpoly(LAM).as_expr())}")
        answer += eigenvalues(matrix).answer
        steps += ["Determinant:", *determinant_lines(matrix), "Eigenvalues:", *eigen(matrix).steps[1:]]
    return Result("Matrix summary", answer, steps)


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
    except NonlinearError:
        return _nonlinear(expressions, letters)
    augmented = coefficients.row_join(constants)
    r_a, r_b = coefficients.rank(), augmented.rank()
    form, operations, _ = _row_reduce(augmented)
    steps = [f"Write the equations as a matrix [A|B], with the unknowns {_list(letters)} in columns and the constants last:\n{show_matrix(augmented)}",
             "Reduce it to row echelon form:", *operations,
             f"Compare the ranks: rank(A) = {r_a}, rank([A|B]) = {r_b}, unknowns = {len(letters)}"]
    if r_a != r_b:
        steps.append("A row now says 0 = a number that is not 0, so the ranks differ and the equations cannot all hold")
        return Result("Linear system", ["The system is inconsistent: it has no solution"], steps)
    solution = next(iter(sp.linsolve((coefficients, constants), *letters)))
    lines = [f"{name} = {show_plain(value)}" for name, value in zip(letters, solution)]
    steps += _back_substitution(form, letters, solution)
    if r_a == len(letters):
        kind = "consistent, with a unique solution"
    else:
        kind = f"consistent, with infinitely many solutions ({len(letters) - r_a} free unknown)"
    return Result("Linear system", [f"The system is {kind}"] + lines, steps)


def _back_substitution(form, letters, solution):
    """Read the echelon form from the bottom row up, each row giving one unknown."""
    lines = []
    for row in range(form.rows - 1, -1, -1):
        pivot = next((c for c in range(form.cols - 1) if form[row, c] != 0), None)
        if pivot is None:
            continue
        written = show(sum(form[row, c] * letters[c] for c in range(form.cols - 1)))
        lines.append(f"R{row + 1} says {written} = {show_plain(form[row, form.cols - 1])}, so {letters[pivot]} = {show_plain(solution[pivot])}")
    return lines


def _nonlinear(expressions, letters):
    solutions = sp.solve(expressions, letters, dict=True)
    steps = ["The equations are not linear, so solve one for an unknown and put it into the others: " + "; ".join(f"{show(e)} = 0" for e in expressions)]
    if not solutions:
        return Result("System of equations", ["No solution was found"], steps + ["This leaves no value that fits every equation"])
    for solution in solutions:
        values = ", ".join(f"{k} = {show_plain(v)}" for k, v in solution.items())
        steps.append(f"This gives {values}. Putting it back, the equations come to " + ", ".join(show_plain(sp.simplify(e.subs(solution))) for e in expressions))
    return Result("System of equations", [", ".join(f"{k} = {show_plain(v)}" for k, v in s.items()) for s in solutions], steps)


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
    steps.append("Make each diagonal unknown the subject of its row: " + "; ".join(
        f"{letters[i]} = ({show(constants[row] - sum(coefficients[row, c] * letters[c] for c in range(n) if c != i))}) ÷ {show_plain(coefficients[row, i])}" for i, row in enumerate(order)))
    steps.append("Each new value is used at once in the next row")
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
