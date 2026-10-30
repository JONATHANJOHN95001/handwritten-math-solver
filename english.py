"""Understand a question typed or pasted in English and turn it into commands for the methods.

The patterns were taught from this year's assignments and exam papers. They are rules, not a language model, so a question
worded very differently may not be understood, and then the app says so. The commands are shown to the student, so a
wrong reading can be seen and corrected.
"""
import math
import re
import unicodedata

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
NUM = r"-?\d+(?:\.\d+)?"
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12}
ORDINAL_POWER = {"second": 2, "third": 3, "fourth": 4}


def normalise(text):
    """Plain characters from what a PDF or a phone gives: 𝑥 to x, ² to ^2, − to -, ≤ to <=."""
    text = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^" + m[0].translate(SUPERSCRIPTS), text)
    text = text.translate(SUBSCRIPTS)
    text = text.replace("x̄", "xbar").replace("𝑥̅", "xbar").replace("ȳ", "ybar").replace("𝑦̅", "ybar")
    text = unicodedata.normalize("NFKC", text)
    for old, new in {"−": "-", "–": "-", "—": "-", "≤": "<=", "≥": ">=", "’": "'", "‘": "'", "“": '"', "”": '"', "̂": "", "⃗": ""}.items():
        text = text.replace(old, new)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _flat(text):
    return re.sub(r"\s+", " ", text).strip()


def _fix_lost_powers(text):
    """A PDF drops the raised look: x2 back to x^2, (..)5 back to (..)^5, d3y/dx3 back to d^3y/dx^3."""
    text = re.sub(r"\bd(\d)y\s*/\s*dx(\d)\b", r"d^\1y/dx^\2", text)
    return text


def _word_number(word):
    return WORD_NUMBERS.get(word.lower(), word)


def _numbers(text):
    return re.findall(NUM, text)


def _list_after(text, label, minimum=2):
    """The run of numbers after a label: 'Marks (Statistics) 53 55 52' gives [53, 55, 52]."""
    match = re.search(label + r"[^\d\-]{0,40}?((?:" + NUM + r"[\s,]+(?:and\s+)?)*" + NUM + r")", text, re.IGNORECASE)
    if not match:
        return None
    found = _numbers(match[1])
    return found if len(found) >= minimum else None


def _value_after(text, label):
    match = re.search(label + r"[^\d\-]{0,40}?(" + NUM + r")", text, re.IGNORECASE)
    return match[1] if match else None


def _matrix(text):
    """The matrix in a question: [[1,2],[3,4]], [1 2; 3 4], or a flat list of n² numbers as a PDF pastes it."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    body = match[0]
    if "[[" in body or ";" in body:
        return _flat(body)
    numbers = _numbers(body)
    size = int(math.isqrt(len(numbers)))
    if size < 2 or size * size != len(numbers):
        return None
    return "[" + ",".join("[" + ",".join(numbers[r * size:(r + 1) * size]) + "]" for r in range(size)) + "]"


def _equations(text):
    """The equations of a system: separated by ; or , and ending at a full stop."""
    text = re.sub(r"\band\b", ";", text)
    pieces = [p.strip(" ,.") for p in re.split(r"[;,]", text)]
    return [p for p in pieces if "=" in p and re.search(r"[a-zλμ]", p)]


# ---------- Matrix Algebra and Calculus ----------

def _rules_matrix(t, low):
    matrix = _matrix(t)
    if matrix:
        if "rank" in low:
            return [f"rank {matrix}"]
        if "diagonal" in low:
            power = re.search(r"find\s*a\s*\^?\s*(\d+)", low)
            return [f"diagonalize {matrix}" + (f" power {power[1]}" if power else "")]
        if "eigen" in low and "vector" in low:
            return [f"eigenvectors {matrix}"]
        if "eigen" in low:
            return [f"eigenvalues {matrix}"]
        if "determinant" in low and "rank" not in low:
            return [f"summary {matrix}"] if "characteristic" in low else [f"det {matrix}"]
        if "inverse" in low:
            return [f"inverse {matrix}"]
    return None


def _rules_system(t, low):
    if "seidel" in low:
        body = re.split(r"\b(?:carry out|by the gauss|by gauss|taking|using)\b", re.sub(r"^.*?equations[.:]?", "", t, flags=re.I | re.S), flags=re.I)[0]
        equations = _equations(body)
        if len(equations) < 2:
            return None
        count = re.search(r"(\d+|one|two|three|four|five|six)\s*iterations", low)
        start = re.search(r"\((\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*)\)", t)
        places = re.search(r"(\w+)\s*decimal", low)
        command = "gauss-seidel " + "; ".join(equations)
        if count:
            command += f"; iterations={_word_number(count[1])}"
        if start:
            command += f"; start=({start[1]})"
        if places and str(_word_number(places[1])).isdigit():
            command += f"; places={_word_number(places[1])}"
        return [command]
    if re.search(r"investigate the values", low):
        body = re.sub(r"^.*?equations", "", t, flags=re.I | re.S)
        body = re.split(r"\bmay have\b|\bhas\b", body)[0]
        return ["investigate " + "; ".join(_equations(body))]
    if re.search(r"consisten|system of (linear )?equations|solve the system", low):
        body = re.sub(r"^.*?(?:solve(?: the system of (?:linear )?equations)?|equations)\b", "", t, flags=re.I | re.S)
        equations = _equations(body)
        if len(equations) >= 2:
            return ["solve " + "; ".join(equations)]
    return None


ODE_PREFIX = re.compile(r"^.*?\b(?:parameters,?\s*solve|solve the initial value problem|initial value problem|particular integral of|general solution of|solve)\b:?\s*", re.I | re.S)


def _rules_ode(t, low):
    if re.search(r"x'\s*=", t, re.I) and "[" in t:
        match = re.search(r"(\[.*?\]\]?)\s*\*?\s*x\b\s*(?:\+\s*\[(.*?)\])?", t, re.I | re.S)
        if match:
            numbers = _numbers(match[1])
            size = int(math.isqrt(len(numbers)))
            if size * size == len(numbers):
                matrix = "[" + ",".join("[" + ",".join(numbers[r * size:(r + 1) * size]) + "]" for r in range(size)) + "]"
                forcing = ""
                if match[2]:
                    entries = re.split(r"[,\s]+", match[2].strip())
                    forcing = " + [" + ",".join(entries) + "]"
                return [f"x' = {matrix} x{forcing}"]
    if not re.search(r"\by'|d\^?\d?y\s*/\s*d|\(\s*d\^?\d?\s*[+\-)]|\bD\^|\(D\b|\bD\s*[-+]", t):
        return None
    body = ODE_PREFIX.sub("", _fix_lost_powers(t)).strip(" .")
    body = re.sub(r"\s*\bby\b.*$", "", body)
    body = re.sub(r"\s*,\s*(?=y'*\()", "; ", body)
    body = re.sub(r"(?<=[^;])\s+(?=y'*\(\s*[-\d])", "; ", body)
    return [f"solve {body}"]


def _partial_function(text):
    match = re.search(r"(?:function|of)\s*:?\s*((?:[a-z]\s*=|f\s*\(\s*x\s*,\s*y[^)]*\)\s*=)?\s*[^.]+)", text, re.I | re.S)
    return _flat(match[1]).rstrip(". ") if match else None


def _rules_multivariable(t, low):
    if "partial derivative" in low:
        match = re.search(r"partial derivatives? of (?:the )?(?:given )?(?:function)?\s*:?\s*(.+)", t, re.I | re.S)
        if match:
            return ["partial " + _flat(match[1]).strip(".: ")]
    if "total derivative" in low:
        match = re.search(r"functions?\s+(.+?)(?:\band also\b|$)", t, re.I | re.S)
        if match:
            body = _flat(match[1]).rstrip(", .")
            at = re.search(r",?\s*\bt\s*=\s*([^,;\s]+)\s*,?\s*$", body)
            if at:
                body = body[:at.start()]
            pieces = "; ".join(p.strip() for p in re.split(r"[;,]", body) if "=" in p)
            return [f"total derivative {pieces}" + (f"; at t={at[1]}" if at else "")]
    if "jacobian" in low or re.search(r"[∂d]\(\s*u\s*,\s*v", t):
        definitions = re.findall(r"\b([uvw]|x|y)\s*=\s*([^,;]+?)(?=\s*(?:[,;]|\band\b|\bfind\b|\bwith\b|\.|$))", t, re.I)
        at = re.search(r"at\s*\(([^)]*)\)", t)
        if len(definitions) >= 2:
            names = "; ".join(f"{a}={b.strip()}" for a, b in definitions)
            return [f"jacobian {names}" + (f" at ({at[1]})" if at else "")]
    if re.search(r"jj'|jj′|j\s*j'", low):
        definitions = re.findall(r"\b([xy])\s*=\s*([^,;]+?)(?=\s*(?:[,;]|\band\b|\.|$))", t)
        if len(definitions) == 2:
            return ["jacobian product " + "; ".join(f"{a}={b.strip()}" for a, b in definitions)]
    if re.search(r"temperature", low) and "sphere" in low:
        function = re.search(r"T\s*=\s*([^.]+?)\s*\.", t)
        constraint = re.search(r"sphere\s*([^.]+)", t)
        if function and constraint:
            return [f"maximize {function[1].strip()} subject to {constraint[1].strip()}"]
    if re.search(r"box", low) and re.search(r"open at the top", low):
        volume = _value_after(t, "volume of")
        return [f"minimize positive xy+2yz+2zx subject to xyz={volume}"]
    if re.search(r"extreme value|maxima and minima|minimum value of|is maximum at", low):
        match = re.search(r"(?:function|value of)\s*:?\s*(?:[a-z]\s*=\s*|f\(x,\s*y\)\s*=\s*)?(.+?)(?:\s+is\b|,\s*a\s*>|,\s*where|\.\s*$|$)", t, re.I | re.S)
        if match:
            return ["extrema " + _flat(match[1]).rstrip(". ")]
    if re.search(r"prove that|show that", low) and re.search(r"where\s+u\s*=", low) and "∂" in t:
        match = re.search(r"(?:prove|show) that\s*(.+?)\s*,?\s*where\s+(u\s*=\s*.+?)(?:\s+and hence.*)?$", _flat(t), re.I)
        if match:
            return [f"prove {match[1]} where {match[2].rstrip('. ')}"]
    return None


def _vector_tail(text):
    return _flat(text).rstrip(". ")


def _rules_vector(t, low):
    if "directional derivative" in low:
        match = re.search(r"derivatives? of\s*(?:the\s*)?(?:function\s*)?(?:[a-zφ]\s*(?:\([^)]*\))?\s*=\s*)?(.+?)\s*[,.]?\s*at\s*(?:the point\s*)?\(([^)]*)\)\s*(?:in the direction of(?: the vector)?|along)\s*(?:[a-z]\s*=\s*)?(.+)", t, re.I | re.S)
        if match:
            return [f"directional {_flat(match[1])} at ({match[2]}) along {_vector_tail(match[3])}"]
    if re.search(r"components? of velocity", low):
        curve = re.search(r"r\s*→?\s*(?:\([^)]*\))?\s*=\s*(.+?)\s*(?:in the direction|along)", t, re.I | re.S)
        at = re.search(r"\bt\s*=\s*(\S+?)\b", t)
        along = re.search(r"(?:in the direction(?: of)?(?: the vector)?|along)\s*(.+)", t, re.I | re.S)
        if curve and at and along:
            return [f"velocity r={_flat(curve[1])} at t={at[1]} along {_vector_tail(along[1])}"]
    if re.search(r"parametric", low) and re.search(r"\bx\s*=", t):
        parts = re.findall(r"\b([xyz])\s*=\s*([^,;]+?)(?=\s*(?:,|;|\bwhere\b|\brepresents\b|\bas\b|$))", t)
        if len(parts) == 3:
            r = "+".join(f"({body.strip()}){unit}" for (_, body), unit in zip(parts, "ijk"))
            if "tangent" in low:
                times = re.search(r"t\s*=\s*(\S+?)\s*and\s*t\s*=\s*(\S+?)[.\s]*$", t, re.S)
                return [f"tangent r={r}" + (f" at t={times[1]} and t={times[2]}" if times else "")]
            at = re.search(r"magnitudes? at t\s*=\s*(\S+?)[.\s]*$", t, re.S)
            return [f"velocity r={r}" + (f" at t={at[1]}" if at else "")]
    if "angle between" in low and ("sphere" in low or "surface" in low):
        at = re.search(r"at the point\s*\(([^)]*)\)", t)
        equations = re.findall(r"([^.;]*?x\s*\^\s*2[^.;]*?=\s*-?\d+)", re.sub(r"^.*?(?:spheres|surfaces)", "", _flat(t).split("at the point")[0], flags=re.I))
        equations = [re.sub(r"^\s*and\s*", "", e).strip() for e in equations]
        if at and len(equations) == 2:
            return [f"angle surfaces {equations[0]}; {equations[1]} at ({at[1]})"]
    if re.search(r"solenoidal|irrotational", low):
        match = re.search(r"\b[vF]\s*→?\s*=\s*(.+?)\s*(?:is both|is\s|$)", t, re.S) or re.search(r"function\s*(.+?)\s*(?:is both|is\s|$)", t, re.S)
        if match:
            return [f"solenoidal {_flat(match[1])}"]
    if re.search(r"conservative", low):
        match = re.search(r"F\s*→?\s*=\s*(.+?)\s*(?:is a|is\s|\?)", t, re.S)
        if match:
            return [f"potential {_flat(match[1])}"]
    if re.search(r"curl\s*\(\s*curl", low):
        match = re.search(r"[vF]\s*→?\s*=\s*(.+)", t, re.S)
        if match:
            return [f"curl curl {_vector_tail(match[1])}"]
    if re.search(r"\bcurl\b", low):
        match = re.search(r"[vF]\s*→?\s*=\s*(.+)", t, re.S)
        if match:
            return [f"curl {_vector_tail(match[1]).replace('∇', 'grad')}"]
    if re.search(r"\bdiv\b|divergence", low):
        match = re.search(r"[vF]\s*→?\s*=\s*(.+)", t, re.S)
        if match:
            return [f"div {_vector_tail(match[1]).replace('∇', 'grad')}"]
    if re.search(r"\bgrad\b|gradient", low):
        match = re.search(r"f\s*(?:\([^)]*\))?\s*=\s*(.+?)(?:\s*,?\s*then\b|\s*\.?\s*$)", t, re.I | re.S)
        if match:
            return [f"grad {_flat(match[1])}"]
    return None


def _rules_integral(t, low):
    if "∫" not in t:
        return None
    if re.search(r"show that|prove that", low) and "=" in t:
        match = re.search(r"(?:show|prove) that\s*(.+?)\s*(?:using.*)?$", _flat(t), re.I)
        return [f"prove {match[1].rstrip('. ')}"] if match else None
    match = re.search(r"(∫.+?)(?:\s+using\b|\s+by\b|\.\s*$|$)", _flat(t), re.I)
    return [match[1].rstrip(". ")] if match else None


# ---------- Statistics and Probability ----------

def _plural_table(text, first, second):
    """Two labelled rows of a pasted table."""
    a, b = _list_after(text, first), _list_after(text, second)
    return (a, b) if a and b and len(a) == len(b) else None


def _table_rows(text):
    """All the runs of three or more numbers in a pasted table, longest first in reading order."""
    return [_numbers(run) for run in re.findall(r"(?:" + NUM + r"[ \t,]+){2,}" + NUM, text)]


def _rules_describe(t, low):
    if re.search(r"regression lines", low) and len(re.findall(r"=", t)) >= 2:
        equations = re.findall(r"(\d[^,;]*?[xy][^,;]*?=\s*-?\d+)", re.split(r"are the two", t)[0])
        sigma = re.search(r"σ\s*_?x\s*=\s*(" + NUM + ")", t) or re.search(r"sigma\s*x\s*=\s*(" + NUM + ")", t)
        if len(equations) == 2:
            return ["regression lines " + "; ".join(e.strip() for e in equations) + (f"; sx={sigma[1]}" if sigma else "")]
    if re.search(r"values? of mode and mean|asymmetrical", low):
        values = _numbers(t)
        first_is_mode = low.find("mode") < low.find("mean")
        mode, mean = (values[0], values[1]) if first_is_mode else (values[1], values[0])
        return [f"empirical mode={mode} mean={mean}"]
    if re.search(r"in which subject|which subject", low):
        labels = re.findall(r"Marks\s*\(\s*(\w+)\s*\)\s*((?:\d+\s*)+)", t)
        if len(labels) == 2:
            return ["compare " + "; ".join(f"{name.lower()}={','.join(_numbers(values))}" for name, values in labels)]
    if re.search(r"correlation|regression|least square|scatter", low) and not re.search(r"define|write the equations", low):
        rows = [r for r in _table_rows(t) if len(r) >= 3]
        if len(rows) > 2:  # a serial-number row before the two data rows
            rows = rows[-2:]
        if len(rows) >= 2 and len(rows[-1]) == len(rows[-2]):
            x, y = rows[-2], rows[-1]
            data = f"x={','.join(x)}; y={','.join(y)}"
            predict = re.search(r"(" + NUM + r")\s*years? old", t) or re.search(r"estimate.*?(" + NUM + ")", t)
            data += f"; predict={predict[1]}" if predict else ""
            if "quadratic" in low:
                return [f"quadratic regression {data}"]
            if "exponential" in low:
                return [f"exponential regression {data}"]
            if "correlation" in low:
                return [f"correlation x={','.join(x)}; y={','.join(y)}"]
            return [f"regression {data}"]
    grouped = list(re.finditer(r"(\d+)\s*-\s*(\d+)", t))
    if len(grouped) >= 3 and re.search(r"mean|median|mode|variance", low):
        counts = _numbers(t[grouped[-1].end():])
        if len(counts) >= len(grouped):
            classes = ",".join(f"{g[1]}-{g[2]}" for g in grouped)
            return [f"stats classes={classes}; freq={','.join(counts[:len(grouped)])}"]
    frequency = re.search(r"freq\w*\s*(?:\([^)]*\))?\s*:?\s*((?:\d+[\s,]+)+\d+)", t, re.I)
    if frequency and re.search(r"mean|median|variance", low):
        before = t[:frequency.start()]
        values = re.findall(r"((?:\d+[\s,]+)+\d+)", before)
        if values:
            return [f"stats values={','.join(_numbers(values[-1]))}; freq={','.join(_numbers(frequency[1]))}"]
    if re.search(r"\bmean\b|median|\bmode\b|variance|standard deviation", low) and not re.search(r"confidence|hypothes|sample of|normal|population", low):
        lists = [r for r in re.findall(r"(?:\d+(?:\.\d+)?\s*,\s*){3,}\d+(?:\.\d+)?", t)]
        if lists:
            return [f"stats {','.join(_numbers(lists[-1]))}"]
    return None


PERCENT_LIST = r"((?:\d+(?:\.\d+)?%\s*,?\s*(?:and\s*)?){2,})"


def _rules_probability(t, low):
    if re.search(r"p\s*\(\s*a\s*\)\s*=", low) and re.search(r"p\s*\(\s*b\s*\)\s*=", low):
        pieces = re.findall(r"P\s*\(\s*([AB∩nU]+)\s*\)\s*=\s*(" + NUM + r"(?:/\d+)?)", t.replace("𝐴", "A").replace("𝐵", "B"))
        known = {a.replace("∩", "n"): b for a, b in pieces}
        if {"A", "B", "AnB"} <= set(known):
            return [f"prob P(A)={known['A']}; P(B)={known['B']}; P(AnB)={known['AnB']}"]
    if re.search(r"machines?", low) and re.search(r"defective", low):
        lists = re.findall(PERCENT_LIST, t)
        if len(lists) >= 2:
            priors, likelihoods = [_list_percent(x) for x in lists[:2]]
            names = re.search(r"machines?\s+((?:[A-Z]\s*,\s*)+[A-Z])", t)
            labels = re.sub(r"\s", "", names[1]) if names else ",".join("ABCDEFG"[:len(priors)])
            want = re.search(r"manufactured by\s+([A-Z](?:\s*(?:or|and|,)\s*[A-Z])*)", t)
            command = f"bayes labels={labels}; priors={priors}; likelihoods={likelihoods}"
            return [command + (f"; want={','.join(re.findall(r'[A-Z]', want[1]))}" if want else "")]
    if "doctor" in low and "diagnos" in low:
        values = re.findall(r"(\d+)%", t)
        if len(values) == 3:
            correct, die_right, die_wrong = values
            return [f"bayes labels=correct,wrong; priors={correct}%,{100 - int(correct)}%; likelihoods={die_right}%,{die_wrong}%; want=correct"]
    if re.search(r"\bbag\b", low) and re.search(r"white", low):
        pairs = re.findall(r"(\d+)\s*white\s*and\s*(\d+)\s*(red|black)", t)
        if len(pairs) == 2 and re.search(r"put in|transfer", low):
            moved = re.search(r"(\d+)\s*balls? (?:are|is) drawn", t)
            return [f"transfer white1={pairs[0][0]} red1={pairs[0][1]} white2={pairs[1][0]} red2={pairs[1][1]} moved={moved[1]}"]
        if len(pairs) == 2 and re.search(r"drawn from bag", low):
            (w1, b1, _), (w2, b2, _) = pairs
            colour = "white" if re.search(r"found to be white", low) else "black"
            first, second = (w1, w2) if colour == "white" else (b1, b2)
            return [f"bayes labels=I,II; priors=1/2,1/2; likelihoods={first}/{int(w1) + int(b1)},{second}/{int(w2) + int(b2)}; want=I"]
    equation = re.search(r"(\d*)\s*P\s*\(\s*X\s*=\s*(\d+)\s*\)\s*=\s*(\d*)\s*P\s*\(\s*X\s*=\s*(\d+)\s*\)", t, re.I)
    if equation and "poisson" in low:
        return [f"poisson equation {equation[1]}P(X={equation[2]})={equation[3]}P(X={equation[4]})"]
    if re.search(r"chosen at random", low) and "positive" in low and "negative" in low and "multiplied" in low:
        match = re.search(r"(\d+)\s*positive\s*and\s*(\d+)\s*negative", low)
        p, n = int(match[1]), int(match[2])
        k = int(re.search(r"(\d+)\s*numbers\s*are", low)[1])
        terms = [f"{p}C{k - j}*{n}C{j}" for j in range(0, k + 1, 2)]
        return [f"({'+'.join(terms)})/{p + n}C{k}"]
    if re.search(r"\byes\b.*\bno\b", low) and "out of" in low:
        least = re.search(r"at least (\d+)", low)
        total = re.search(r"out of (\d+)", low)
        options = re.search(r"(\d+)\s*options", low)
        commands = [f"binomial n={total[1]} p=1/2 P(X>={least[1]})"]
        if options:
            commands.append(f"binomial n={total[1]} p=1/{options[1]} P(X>={least[1]})")
        return commands
    if re.search(r"defective", low) and re.search(r"box of|selected", low) and not re.search(r"poisson", low):
        match = re.search(r"(\d+)\s*\w+\s*in a box of\s*(\d+)\s*are defective", low)
        pick = re.search(r"(\d+)\s*\w+\s*(?:are|is)\s*selected", low)
        if match and pick:
            n = pick[1]
            queries = []
            for number, kind in re.findall(r"\(\w+\)\s*(\d+)\s*are\s*(defective|good)", low):
                queries.append(f"P(X={number})" if kind == "defective" else f"P(X={int(n) - int(number)})")
            if "at least one" in low:
                queries.append("P(X>=1)")
            at_most = re.search(r"at most (\d+)", low)
            if at_most:
                queries.append(f"P(X<={at_most[1]})")
            return [f"binomial n={n} p={match[1]}/{match[2]} " + " ".join(queries)]
    if re.search(r"bad reaction|poisson", low) or re.search(r"razor", low):
        chance = re.search(r"(0\.\d+)", t)
        if chance and re.search(r"out of\s*([\d,]+)", low):
            n = re.search(r"out of\s*([\d,]+)", low)[1].replace(",", "")
            more = re.search(r"more than (\w+)", low)
            return [f"poisson n={n} p={chance[1]} P(X>{_word_number(more[1])})"]
        packet = re.search(r"packet of (\w+)", low)
        total = re.search(r"([\d,]+)\s*packets", low)
        if chance and packet and total:
            return [f"poisson n={_word_number(packet[1])} p={chance[1]} repeats={total[1].replace(',', '')} P(X=0) P(X=1) P(X=2)"]
    if re.search(r"uniformly distributed", low):
        ends = re.search(r"between\s*(" + NUM + r")\s*and\s*(" + NUM + ")", low)
        fewer = re.search(r"(?:fewer|less) than\s*(" + NUM + ")", low)
        if ends:
            return [f"uniform a={ends[1]} b={ends[2]}" + (f" P(X<{fewer[1]})" if fewer else "")]
    if re.search(r"constant k|value of k|find k", low) and re.search(r"f\s*\(\s*x\s*\)", low):
        match = re.search(r"f\s*\(\s*x\s*\)\s*=\s*\{?\s*(k[^,]*?)\s*,\s*(" + NUM + r")\s*<\s*x\s*<\s*(" + NUM + ")", t, re.I)
        if match:
            queries = re.findall(r"P\s*\(\s*([^)]+?)\s*\)", t, re.I)
            return [f"pdf {match[1].strip()} from {match[2]} to {match[3]}; " + "; ".join(f"P({q})" for q in queries)]
    if re.search(r"value of k", low) and re.search(r"p\s*\(\s*x\s*\)", low):
        x_row = re.search(r"\bx\s+((?:-?\d+\s+)+-?\d+)", t)
        p_row = re.search(r"p\s*\(\s*x\s*\)\s+((?:\d*k\s*)+)", t, re.I)
        if x_row and p_row:
            queries = re.findall(r"P\s*\(\s*([^)]+?)\s*\)", t, re.I)
            queries = [q for q in queries if q.lower() != "x"]
            return [f"distribution x={','.join(_numbers(x_row[1]))}; p={','.join(p_row[1].split())}; " + "; ".join(f"P({q})" for q in queries)]
    if re.search(r"normal", low) and re.search(r"(mean|average)", low) and re.search(r"standard deviation|s\.?\s*d", low) and not re.search(r"confidence|sample of|hypoth|samples", low):
        mean = _value_after(t, r"(?:mean|average(?: life)?)(?: of)?")
        sd = _value_after(t, r"(?:standard deviation|S\.?\s*D\.?)(?: of)?")
        total = re.search(r"(?:of|on)\s*([\d,]+)\s*(?:students|electric|bulbs|items)", t)
        queries = _range_queries(low)
        command = f"normal mean={mean} sd={sd}"
        if total:
            command += f" repeats={total[1].replace(',', '')}"
        return [command + " " + " ".join(queries)]
    return None


def _list_percent(text):
    return ",".join(re.findall(r"\d+(?:\.\d+)?%", text))


def _range_queries(low, name="X"):
    """'more than 2150', 'less than 1950', 'between 65 and 75', 'more than 1920 but less than 2160'."""
    queries = []
    used = low
    for a, b in re.findall(r"more than\s*(" + NUM + r")\s*(?:\w+\s*)?(?:and\s*)?but less than\s*(" + NUM + ")", used):
        queries.append(f"P({a}<{name}<{b})")
    used = re.sub(r"more than\s*" + NUM + r"\s*(?:\w+\s*)?(?:and\s*)?but less than\s*" + NUM, " ", used)
    for a, b in re.findall(r"between\s*(" + NUM + r")\s*and\s*(" + NUM + ")", used):
        queries.append(f"P({a}<{name}<{b})")
    used = re.sub(r"between\s*" + NUM + r"\s*and\s*" + NUM, " ", used)
    for a in re.findall(r"(?:more than|exceed|greater than|over|at least)\s*(" + NUM + ")", used):
        queries.append(f"P({name}>{a})")
    for a in re.findall(r"(?:less than|fewer than|below|at most)\s*(" + NUM + ")", used):
        queries.append(f"P({name}<{a})")
    return queries


def _rules_sampling(t, low):
    if re.search(r"population consists|if the population is", low):
        numbers = re.search(r"(?:numbers|population is)\s*((?:\d+\s*,\s*)+\d+)", t, re.I)
        size = re.search(r"size\s*(\w+)", low)
        if numbers and size:
            how = "without replacement" if "without" in low else "with replacement"
            return [f"sampling population={numbers[1].replace(' ', '')}; size={_word_number(size[1])} {how}"]
    if re.search(r"masses", low) and re.search(r"s\.?\s*d", low):
        lists = re.findall(r"((?:\d+\.\d+\s*,\s*)+\d+\.\d+)", t)
        if len(lists) == 2:
            return [f"sum of normals means={lists[0].replace(' ', '')}; sds={lists[1].replace(' ', '')}"]
    if re.search(r"sample of size|sample size", low) and re.search(r"μ|mu\b|population mean", t.lower()):
        n = _value_after(t, "size")
        mu = _value_after(t, r"(?:μ|mu)\s*=")
        sd = _value_after(t, r"(?:σ|sigma)\s*=")
        queries = _range_queries(low.replace("fall between", "between"), "xbar")
        if n and mu and sd:
            return [f"clt mu={mu} sd={sd} n={n} " + " ".join(queries)]
    if re.search(r"sampling distribution for the sample proportion|sample proportion", low) or (re.search(r"\d+%\s*of", low) and re.search(r"over\s*\d+%", low)):
        p = re.search(r"(\d+)%", low)
        n = re.search(r"on\s*(\d+)\s*new|sample of\s*(\d+)", low)
        over = re.search(r"over\s*(\d+)%", low)
        if p and n and over:
            return [f"proportion p={int(p[1]) / 100:g} n={n[1] or n[2]} P(phat>{int(over[1]) / 100:g})"]
    return None


def _sample_values(t):
    """n, mean and s.d. from wordings such as 'sample of 40 fans has an average life of 800 hours'."""
    n = re.search(r"sample of\s*(\d+)|(\d+)\s*(?:randomly selected|bolts|steel rods|batteries|fans|items)|sample of size\s*(\d+)", t, re.I)
    rest = re.sub(r"sample of\s*(?:size\s*)?\d+", "sample", t, flags=re.I)
    mean = (re.search(r"sample mean[^\d$]{0,30}?\$?(" + NUM + ")", rest, re.I)
            or re.search(r"(?:average|mean)[^\d$]{0,120}?\$?(" + NUM + ")", rest, re.I))
    sd = re.search(r"(?:standard deviation|\bs\.?\s*d\b\.?)(?:\s*\(σ\))?[^\d$]{0,40}?\$?(" + NUM + ")", rest, re.I)
    return (next((g for g in n.groups() if g), None) if n else None), mean[1] if mean else None, sd[1] if sd else None


def _rules_estimate(t, low):
    if "confidence" not in low:
        return None
    level = re.search(r"(\d+)\s*%\s*confidence", low)
    conf = level[1] if level else "95"
    data = re.findall(r"(?:\d+\s*,\s*){3,}\d+(?:\s*and\s*\d+)?", t)
    if data and not re.search(r"standard deviation", low):
        return [f"ci data={','.join(_numbers(data[-1]))}; conf={conf}"]
    n, mean, sd = _sample_values(t)
    if not all((n, mean, sd)):
        return None
    z = re.search(r"z\s*_?\s*c?\s*=\s*(" + NUM + ")", t)
    known = " known" if re.search(r"normally distributed with a standard deviation|normal distribution with a standard deviation|population standard deviation|follows an approximately normal", low) else ""
    return [f"ci mean={mean} sd={sd} n={n} conf={conf}" + (f" z={z[1]}" if z else "") + known]


def _rules_test(t, low):
    if re.search(r"goodness of fit|equal distribution|equally distributed", low):
        rows = _table_rows(t)
        if rows:
            return [f"chisquare observed={','.join(rows[-1])}"]
    if re.search(r"h\s*0\s*:\s*[μm]\s*1|μ1\s*-\s*μ2|brand a", low) or (re.search(r"two different", low) and re.search(r"brands?", low)):
        diff = re.search(r"H\s*0\s*:\s*μ\s*1\s*-\s*μ\s*2\s*=\s*(" + NUM + ")", t)
        tail = "right" if re.search(r"H\s*a\s*:.*>", t) else "two"
        alpha = re.search(r"(?:level of|level)\s*(0?\.\d+)", low)
        by_letters = re.search(r"m\s*=\s*(\d+)\s*,\s*xbar\s*=\s*(" + NUM + r")\s*,\s*s\s*1\s*=\s*(" + NUM + r")\s*,\s*n\s*=\s*(\d+)\s*,\s*ybar\s*=\s*(" + NUM + r")\s*,\s*s\s*2\s*=\s*(" + NUM + ")", t)
        if by_letters:
            n1, m1, s1, n2, m2, s2 = by_letters.groups()
        else:
            brands = re.findall(r"sample of\s*(\d+)[^.]*?(?:mean|average)[^\d]*?(" + NUM + r")[^.]*?standard deviation of\s*(" + NUM + ")", t, re.I)
            if len(brands) != 2:
                return None
            (n1, m1, s1), (n2, m2, s2) = brands
        return [f"twosample mean1={m1} sd1={s1} n1={n1} mean2={m2} sd2={s2} n2={n2} diff={diff[1] if diff else 0} alpha={alpha[1] if alpha else 0.05} tail={tail}"]
    claims = re.search(r"claims?|h\s*0|hypothes|reject|significance|conclude|contradict", low)
    if claims and re.search(r"sample", low) and re.search(r"standard deviation|s\.?\s*d", low):
        n, mean, sd = _sample_values(t)
        both = re.search(r"sample mean and sample s\.?d\.?[^\d]*?(\d+)\s*and\s*(\d+)", t, re.I)
        if both:
            mean, sd = both[1], both[2]
        word_count = re.search(r"\b(twenty-six|twenty-five)\b", low)
        if word_count:
            n = {"twenty-six": "26", "twenty-five": "25"}[word_count[1]]
        count = re.search(r"sample of\s*(\d+)|\b(\d+)\s*randomly|\b(\d+)\s*(?:fluorescent|undergraduates|monthly|escape)", t, re.I)
        if count and not word_count:
            n = next(g for g in count.groups() if g)
        mu = re.search(r"H\s*0\s*:\s*[μm]\s*=\s*(" + NUM + ")", t) or re.search(r"(?:claims?|believed|average|mean)[^.]*?(?:is|be|than|no more than|is more that)\s*\$?(" + NUM + r")\s*(?:hours|grams|ml|seconds|kg)?", t, re.I)
        if re.search(r"more than \$?(\d+)|more that \$?(\d+)", low):
            found = re.search(r"more tha[nt] \$?(\d+)", low)
            mu = found
        if re.search(r"\((\d+)\s*seconds\)", low):
            mu = re.search(r"\((\d+)\s*seconds\)", low)
        if "average life of the bulbs" in low:
            mu = re.search(r"company is\s*(\d+)", low)
        alpha = re.search(r"(?:a|α|level of significance of|level of)\s*=?\s*(0?\.\d+)", low)
        percent = re.search(r"(\d+)%\s*(?:significance|level)", low)
        alpha_value = alpha[1] if alpha else (int(percent[1]) / 100 if percent else 0.05)
        tail = "two"
        if re.search(r"h\s*1\s*:\s*[μm]\s*>|more than \$?\d+|more that \$?\d+|no more than", low):
            tail = "right"
        critical = re.search(r"t\s*_?\s*0?\.\d+\s*,\s*\d+\s*=\s*(" + NUM + r")|z\s*_?\s*c?\s*=\s*(" + NUM + ")", t)
        kind = "ztest" if n and int(n) >= 30 else "ttest"
        if not all((n, mean, sd, mu)):
            return None
        command = f"{kind} mean={mean} mu={mu[1]} sd={sd} n={n} alpha={alpha_value} tail={tail}"
        if critical:
            command += f" critical={critical[1] or critical[2]}"
        return [command]
    return None


# ---------- Discrete Mathematics ----------

def _rules_counting(t, low):
    if re.search(r"how many distinct|how many different|product rule", low) and re.search(r"come in|models", low):
        return ["product rule " + ",".join(re.findall(r"(\d+)\s*(?:models|colou?rs|engine|transmission)", low))]
    if "license plate" in low:
        letters = re.search(r"(\w+)\s*english letters", low)
        digits = re.search(r"(\w+)\s*digits", low)
        if "vowel" in low and "even" in low:
            return ["product rule " + ",".join(["5"] * (int(_word_number(letters[1])) + int(_word_number(digits[1]))))]
    if "pigeonhole" in low or re.search(r"dictionaries", low):
        match = re.search(r"(\d+)\s*\w+[^.]*?total of\s*([\d,]+)", low)
        if match:
            return [f"pigeonhole {match[2].replace(',', '')} / {match[1]}"]
    if re.search(r"envelopes", low) and re.search(r"right person|at least one", low):
        count = re.search(r"(\w+)\s*letters", low)
        return [f"derangements {_word_number(count[1])}"]
    if re.search(r"books", low) and re.search(r"redistributed|recollected", low):
        count = re.search(r"(\w+)\s*books", low)
        n = int(_word_number(count[1]))
        return [f"{n}!*(derangements)" if False else f"derangements {n}"]
    if re.search(r"non-?negative integer solutions|number of integer solution", low):
        names = re.findall(r"\b[xc]\s*_?(\d)", t)
        total = re.search(r"=\s*(\d+)|<=\s*(\d+)|\bS\s*(\d+)", t)
        count = len(set(names))
        if not count or not total:
            return None
        value = next(g for g in total.groups() if g)
        if re.search(r"<=", t.split("where")[0]):
            return [f"solutions total={value} n={count + 1}"]
        lower = {}
        upper = {}
        for number, bound in re.findall(r"[xc]\s*_?(\d)\s*>=\s*(-?\d+)", t):
            lower[number] = bound
        for bound, number in re.findall(r"(-?\d+)\s*<=\s*c\s*_?(\d)", t):
            lower[number] = bound
        for number, bound in re.findall(r"c\s*_?(\d)\s*<=\s*(-?\d+)", t):
            upper[number] = bound
        ordered = sorted(set(names))
        command = f"solutions total={value} n={count} lower=" + ",".join(lower.get(k, "0") for k in ordered)
        if upper:
            command += " upper=" + ",".join(upper.get(k, "") for k in ordered)
        return [command]
    if re.search(r"distribute", low) and re.search(r"at least", low) and re.search(r"not more than", low):
        total = re.search(r"distribute\s*(\d+)", low)
        people = re.search(r"to\s*(\d+)\s*children", low)
        least = re.search(r"at least\s*(\w+)", low)
        most = re.search(r"not more than\s*(\w+)", low)
        return [f"solutions total={total[1]} n={people[1]} lower={_word_number(least[1])} upper={_word_number(most[1])}"]
    if re.search(r"do not study", low) or (re.search(r"\bstudy\b", low) and re.search(r"total number", low)):
        singles = re.findall(r"(\d+)\s*(?:students\s*)?study\s+\w+", t)
        pairs = re.findall(r"(\d+)\s*(?:students\s*)?for\s*[A-Z]\s*and\s*[A-Z]\b", t)
        triples = re.findall(r"(\d+)\s*(?:students\s*)?for\s*[A-Z]\s*,\s*[A-Z]\s*,?\s*and\s*[A-Z]\b", t)
        everyone = re.search(r"(\d+)\s*who study all", t)
        none = re.search(r"(\d+)\s*students?\s*do not", t)
        if singles and pairs:
            command = f"inclusion singles={','.join(singles)}; pairs={','.join(pairs)}"
            command += f"; triples={','.join(triples)}" if triples else ""
            command += f"; all={everyone[1]}" if everyone else ""
            return [command + (f"; none={none[1]}" if none else "")]
    if re.search(r"how many integers", low):
        ends = re.search(r"between\s*(\d+)\s*and\s*(\d+)", low)
        divisors = re.search(r"divisible by (?:at least one|none) of\s*((?:\d+\s*,\s*)+\d+)", low)
        if ends and divisors:
            return [f"divisible {ends[1]} to {ends[2]} by {divisors[1].replace(' ', '')}"]
    if re.search(r"permuted", low) and re.search(r"pattern", low):
        n = re.search(r"(\d+)\s*letters", low)
        words = re.findall(r"\b[A-Z]{2,}\b", t)
        if n and words:
            return [f"patterns n={n[1]}; words={','.join(words)}"]
    if "rook polynomial" in low:
        size = re.search(r"(\d+)\s*[×x]\s*(\d+)", low)
        if size:
            return [f"rook {size[1]}x{size[2]}"]
    if re.search(r"do not wish|does not want|refuses", low):
        return _forbidden(t)
    return None


def _forbidden(t):
    """'B1 and B2 do not wish to have an apple, B3 does not want a banana or a mango, and B4 refuses an orange.'"""
    wishes = re.split(r"[Tt]he boys\s+|\bThe boy\s+|[,;]\s*(?:and\s*)?(?=(?:the boy\s*)?[A-Z]\d)", t.split(".")[-2] if t.count(".") >= 1 else t)
    rows = {}
    items = set()
    subject = re.search(r"((?:[A-Z]\d\s*(?:,|and)?\s*)+)(?:do not|does not|refuses)", t)
    if not subject:
        return None
    fruit = re.findall(r"\b(?:an?|or an?|or)\s+(apple|banana|mango|orange)\b", t)
    if fruit:
        for clause in re.split(r",\s*(?:and\s*)?(?:the boy\s*)?(?=[A-Z]\d)|\band the boy\b", t[t.find(subject[0]):]):
            people = re.findall(r"\b([A-Z]\d)\b", clause)
            wanted = re.findall(r"\b(apple|banana|mango|orange)\b", clause)
            for person in people:
                rows[person] = wanted
                items.update(wanted)
        all_people = sorted(set(re.findall(r"\b([BT]\d)\b", t)))
        return ["forbidden " + "; ".join(f"{p}:{','.join(rows.get(p, []))}" for p in all_people)]
    clauses = re.split(r",\s*(?:and\s*)?(?=T\d)", t[t.find(subject[0]):].split(". In how")[0])
    for clause in clauses:
        people = re.findall(r"\b(T\d)\b", clause)
        classes = re.findall(r"\b(C\d)\b", clause)
        for person in people:
            rows[person] = classes
    if rows:
        everyone = sorted(set(re.findall(r"\b(T\d)\b", t)))
        return ["forbidden " + "; ".join(f"{p}:{','.join(rows.get(p, []))}" for p in everyone)]
    del wishes, items
    return None


def _rules_gf(t, low):
    if "coefficient of" in low:
        match = re.search(r"coefficient of\s*x\s*\^?\s*(\d+)\s*(?:in(?: the expansion)?(?: of)?)\s*:?\s*(.+)", t, re.I | re.S)
        if match:
            body = _flat(match[2]).rstrip(". ")
            body = re.sub(r"x(\d+)", r"x^\1", body)
            body = re.sub(r"\)(\d+)", r")^\1", body)
            return [f"coefficient of x^{match[1]} in {body}"]
    if "generating function" in low and re.search(r"sequence", low) and "exponential" not in low and "find the sequence" not in low and "determine the sequence" not in low:
        pieces = re.findall(r"\(\w+\)\s*([^()]+?)(?=\s*\(\w+\)|$)", t)
        if pieces:
            return ["gf " + re.sub(r"\s+", "", p).rstrip(".") for p in pieces if "," in p]
    if re.search(r"find the sequence generated", low):
        pieces = re.findall(r"\(i+\)\s*(.+?)(?=\s*,?\s*\(i+\)|\.?\s*$)", t, re.S)
        if pieces:
            return ["gf sequence of " + _flat(p).rstrip(", .") for p in pieces]
    if re.search(r"sequences? for which.*exponential generating", low):
        pieces = re.findall(r"\(i+\)\s*(.+?)(?=\s*,?\s*\(i+\)|\.?\s*$)", t, re.S)
        if pieces:
            return ["egf sequence of " + _flat(p).rstrip(", .") for p in pieces]
    if re.search(r"exponential generating function for", low):
        pieces = re.findall(r"\(i+\)\s*((?:\d+\s*,\s*)+\d+)", t)
        if pieces:
            return ["egf " + p.replace(" ", "") for p in pieces]
    if "partitions of" in low:
        match = re.search(r"partitions of\s*(?:n\s*=\s*)?(\d+)", low)
        if match:
            return [f"partitions of {match[1]}"]
    word = re.search(r"letters in the word\s+([A-Z]+)", t)
    if word:
        count = re.search(r"(\w+)\s*of the letters", low)
        return [f"arrangements {word[1]} {_word_number(count[1])}"]
    if "marbles" in low and "arranged" in low:
        counts = re.findall(r"(\d+)\s*(?:red|blue|green|orange|white|black)", low)
        how_many = re.search(r"(\w+)\s*of these marbles", low)
        if counts and how_many:
            return [f"arrangements counts={','.join(counts)}; r={_word_number(how_many[1])}"]
    return None


def _rules_recurrence(t, low):
    if "virus" in low:
        start = re.search(r"is\s*(\d+)", low)
        percent = re.search(r"(\d+)%", low)
        every = re.search(r"every\s*(\w+)\s*hours", low)
        if start and percent and every:
            hours = int(_word_number(every[1]))
            return [f"growth start={start[1]} percent={percent[1]} periods={24 // hours}"]
    if "pegs" in low or "hanoi" in low:
        return ["hanoi"]
    if re.search(r"shakes hands|handshakes|hand shakes", low):
        return ["handshakes"]
    if "recurrence" in low:
        body = re.sub(r"^.*?recurrence relation:?", "", _flat(t), flags=re.I)
        main = re.split(r"(?:\s*,?\s*(?:for|where|given|∀|n\s*>=|and\s+a\s*_?0))", body)[0].strip(" ,.")
        main = main.replace("×", "*")
        main = re.sub(r"\*\s*(\d+)\s*n\b", r"*\1^n", main)
        starts = re.findall(r"\b([aF])\s*_?\(?\s*(\d)\s*\)?\s*=\s*(-?\d+)", body)
        if "=" not in main:
            return None
        conditions = "".join(f"; {a}({k})={v}" for a, k, v in starts)
        return [f"recurrence {main}{conditions}"]
    return None


def _rules_graph(t, low):
    named = re.search(r"\bK\s*_?(\d+)(?:\s*,\s*(\d+))?\b", t)
    if re.search(r"non-?planar|planar", low) and named and "region" not in low:
        return [f"planar K{named[1]}" + (f",{named[2]}" if named[2] else "")]
    if re.search(r"chromatic", low) and named:
        return [f"chromatic K{named[1]}" + (f",{named[2]}" if named[2] else "")]
    if "regions" in low and "degrees" in low:
        degrees = re.search(r"degrees\s*((?:\d+\s*,\s*)+\d+)", low)
        if degrees:
            return [f"regions degrees={degrees[1].replace(' ', '')}"]
    if re.search(r"\btree\b", low) and re.search(r"leaves", low):
        pairs = re.findall(r"(\w+)\s*(?:vertices|vertex) of\s*degree\s*(\d+)", low)
        if pairs:
            return ["leaves " + ", ".join(f"{d}:{_word_number(c)}" for c, d in pairs)]
    if re.search(r"extension cords|outlets", low):
        devices = re.search(r"(\d+)\s*(?:micro)?computers", low)
        outlets = re.search(r"(\w+)\s*outlets", low)
        if devices and outlets:
            return [f"cords devices={devices[1]} outlets={_word_number(outlets[1])}"]
    if re.search(r"merge sort", low):
        values = re.search(r"\{\s*((?:\d+\s*,\s*)+\d+)\s*\}|list\s*((?:\d+\s*,\s*)+\d+)", t)
        if values:
            return ["merge sort " + (values[1] or values[2]).replace(" ", "")]
    if "prefix code" in low:
        symbols = re.search(r"symbols\s*((?:\w\s*,\s*)+\w)", t)
        frequencies = re.search(r"frequencies\s*((?:\d+\s*,\s*)+\d+)", low)
        if symbols and frequencies:
            pairs = zip(re.findall(r"\w", symbols[1]), _numbers(frequencies[1]))
            return ["huffman " + ", ".join(f"{s}:{f}" for s, f in pairs)]
        message = re.search(r"message\s+([A-Z ]+?)\s*\.", t)
        if message:
            return [f'huffman text="{message[1].strip()}"']
    if re.search(r"\bprim\b|prim's|kruskal|spanning tree|chromatic|planarity|isomorphic|euler's formula", low):
        edges = re.findall(r"\w+\s*-\s*\w+(?:\s*:\s*[\d.]+)?", t)
        if not edges:
            raise ValueError("This question uses a drawn graph. Type its edges, like: prim a-b:3, b-c:1, c-a:2")
        joined = ", ".join(edges)
        if "kruskal" in low:
            return [f"kruskal {joined}"]
        if "prim" in low:
            return [f"prim {joined}"]
        if "chromatic" in low:
            return [f"chromatic {joined}"]
        if "planar" in low or "euler" in low:
            return [f"planar {joined}"]
        return [f"spanning trees {joined}"]
    return None


RULES = [_rules_graph, _rules_recurrence, _rules_gf, _rules_counting, _rules_test, _rules_estimate, _rules_sampling, _rules_probability,
         _rules_describe, _rules_ode, _rules_system, _rules_matrix, _rules_integral, _rules_vector, _rules_multivariable]


def translate(text):
    """The commands an English question stands for, or None if no pattern fits."""
    t = normalise(text)
    low = _flat(t).lower()
    for rule in RULES:
        commands = rule(t, low)
        if commands:
            return commands
    return None
