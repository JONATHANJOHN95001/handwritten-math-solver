"""What every course method returns, and how numbers, formulas and matrices are written in an answer."""
import ctypes
import re
import threading
from dataclasses import dataclass, field

import sympy as sp

from solver import format_value, read_expression


@dataclass
class Result:
    topic: str                                 # 'Rank of a matrix'
    answer: list                               # the final answer, one line per item
    steps: list = field(default_factory=list)  # the working
    read_as: str = None                        # the command an English question was understood as
    parsed: object = None                      # what was read, for display as a formula


def show(value):
    """A number or formula as plain text: 3·x^2, 7/3 ≈ 2.33333, ln(x), e^(-3·x)."""
    value = sp.sympify(value)
    if value.is_number:
        return format_value(value).replace("log(", "ln(").replace("sqrt(", "√(")
    text = sp.sstr(value).replace("**", "^").replace("*", "·").replace("log(", "ln(").replace("exp(", "e^(").replace("sqrt(", "√(").replace("pi", "π")
    return re.sub(r"\bE\b", "e", text)


def decimal(value, places=4):
    """A rounded decimal, without trailing zeros: 2.5, 0.0333."""
    return f"{float(sp.N(value)):.{places}f}".rstrip("0").rstrip(".")


def show_matrix(matrix):
    """One text line per row, in lined-up columns."""
    cells = [[show_plain(matrix[r, c]) for c in range(matrix.cols)] for r in range(matrix.rows)]
    widths = [max(len(row[c]) for row in cells) for c in range(matrix.cols)]
    return "\n".join("[ " + "  ".join(cell.rjust(w) for cell, w in zip(row, widths)) + " ]" for row in cells)


def show_plain(value):
    """Like show, without the decimal after a fraction: for inside matrices and lists."""
    value = sp.sympify(value)
    if value.is_Rational:
        return str(value)
    if value.is_number:
        return sp.sstr(value).replace("**", "^").replace("*", "·").replace("sqrt(", "√(")
    return show(value)


def read_numbers(text):
    """'4, 1, 5' or '20%, 15%' or '1/2 1/3' as exact numbers."""
    numbers = []
    for item in re.split(r"[,\s;]+", text.strip().strip("[](){}")):
        if not item:
            continue
        percent = item.endswith("%")
        value = read_expression(item.rstrip("%"))
        numbers.append(value / 100 if percent else value)
    if not numbers:
        raise ValueError("No numbers were found")
    return numbers


def read_matrix(text):
    """[[1,2],[3,4]] or [1 2; 3 4] as a SymPy matrix."""
    text = text.strip()
    rows = re.findall(r"\[([^\[\]]*)\]", text) if "[[" in text else text.strip("[] ").split(";")
    table = [[read_expression(cell) for cell in re.split(r"[,\s]+", row.strip()) if cell] for row in rows if row.strip()]
    if not table or len({len(row) for row in table}) != 1:
        raise ValueError("Every row of a matrix must have the same number of entries")
    return sp.Matrix(table)


def parts_of(text):
    """Split a question at semicolons and new lines."""
    return [part.strip() for part in re.split(r"[;\n]+", text) if part.strip()]


def options_of(text):
    """'mean=70 sd=5, n=4' as {'mean': '70', 'sd': '5', 'n': '4'}."""
    return {key.lower(): value.strip().rstrip(",") for key, value in re.findall(r"\b([A-Za-z_]+)\s*=\s*([^\s;,]+)", text)}


class _Stopped(BaseException):
    """Raised inside work that ran out of time. Not an Exception, because SymPy catches those in places."""


def time_limited(work, seconds=30):
    """Run work() but give up after a while: SymPy can run for ever on a hard integral."""
    outcome = {}

    def run():
        try:
            outcome["value"] = work()
        except Exception as error:  # handed back to the caller's thread
            outcome["error"] = error
        except _Stopped:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(_Stopped))
        thread.join(5)  # stops at its next line of Python, so it does not go on using the CPU
        raise TimeoutError(f"This took more than {seconds} seconds, so I stopped")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
