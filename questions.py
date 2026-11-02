"""Work out what kind of question this is and send it to the method that solves it.

The command language reads like the paper: 'rank [[1,2],[3,4]]', 'solve x+y=3; x-y=1', 'grad 2x+yz-3y^2'. Anything that
is not a command goes to the plain solver in solver.py. A question it cannot do raises ValueError with a clear message.
"""
import re

import sympy as sp

import calculus
import discrete
import english
import glossary
import graphs
import matrices
import probability
from results import Result, parts_of, read_matrix, read_numbers, time_limited
from solver import describe, read_expression, solve
from working import explain

MATRIX = r"(\[.*\])"


def _int(text):
    return int(read_expression(text))


def _huffman(match):
    message = re.search(r"text\s*=\s*[\"'](.+)[\"']", match[1])
    if message:
        return graphs.huffman_text(message[1])
    return graphs.huffman({symbol: float(weight) for symbol, weight in re.findall(r"(\w+)\s*:\s*([\d.]+)", match[1])})


def _chromatic(match):
    colours = re.search(r"colou?rs?\s*=\s*(\d+)", match[1])
    return graphs.chromatic(re.sub(r"colou?rs?\s*=\s*\d+", "", match[1]), int(colours[1]) if colours else None)


def _isomorphic(match):
    first, second = parts_of(match[1])
    return graphs.isomorphic(first, second)


def _prim(match):
    start = re.search(r"start\s*=\s*(\w+)", match[1])
    return graphs.prim(re.sub(r"start\s*=\s*\w+", "", match[1]), start[1] if start else None)


def _poisson_equation(match):
    return probability.poisson_equation(match[1] or "1", int(match[2]), match[3] or "1", int(match[4]))


def _divisible(match):
    return discrete.divisible(int(match[1]), int(match[2]), [int(v) for v in re.split(r"[,\s]+", match[3].strip()) if v])


def _patterns(match):
    return discrete.patterns(int(match[1]), [w.strip().upper() for w in match[2].split(",")])


def _word_arrangements(match):
    return discrete.word_arrangements(match[1], int(match[2]))


def _counts_arrangements(match):
    return discrete.arrangements([int(v) for v in read_numbers(match[1])], int(match[2]))


def _point(text):
    """'(1,-1,0)' as a list of numbers."""
    return [read_expression(item) for item in text.strip("() ").split(",")]


def _split_at(text):
    """'grad f at (2,1,3)' as ('grad f', [2,1,3]) or (text, None)."""
    match = re.search(r"\bat\s*\(([^)]*)\)\s*$", text)
    return (text[:match.start()].strip(), _point(match[1])) if match else (text.strip(), None)


def _gauss_seidel(match):
    text = match[1]
    iterations = re.search(r"iterations?\s*=\s*(\d+)", text)
    start = re.search(r"start\s*=\s*\(([^)]*)\)", text)
    places = re.search(r"places\s*=\s*(\d+)", text)
    text = re.sub(r"(iterations?|start|places)\s*=\s*(\([^)]*\)|\d+)\s*;?", "", text)
    return matrices.gauss_seidel(text, int(iterations[1]) if iterations else 4, _point(start[1]) if start else None,
                                 int(places[1]) if places else 4)


def _diagonalize(match):
    power = re.search(r"power\s*=?\s*(\d+)|\^\s*(\d+)", match[2] or "")
    return matrices.diagonalize(read_matrix(match[1]), int(power[1] or power[2]) if power else None)


def _motion(match):
    text = match[1]
    along = re.search(r"\balong\s+(.+)$", text)
    if along:
        text = text[:along.start()]
    at = re.search(r"\bat\s*t\s*=\s*([^\s;]+)", text)
    if at:
        text = text[:at.start()]
    curve = text.replace("r=", "", 1).strip(" ;")
    return calculus.motion(curve, read_expression(at[1]) if at else None, along[1] if along else None)


def _directional(match):
    return calculus.directional(match[1], _point(match[2]), match[3])


def _jacobian(match):
    text, at = _split_at(match[1])
    return calculus.jacobian(text, at)


def _grad(match):
    text, at = _split_at(match[1])
    return calculus.gradient(text, at)


def _div(match):
    text, at = _split_at(match[1])
    return calculus.divergence(text, at)


def _surfaces(match):
    text, at = _split_at(match[1])
    first, second = parts_of(text)
    return calculus.surfaces_angle(first, second, at)


def _tangent(match):
    text = match[1]
    times = re.search(r"\bat\s*t\s*=\s*([^\s;]+)\s*(?:and|,)\s*(?:t\s*=\s*)?([^\s;]+)", text)
    if times:
        text = text[:times.start()]
    curve = text.replace("r=", "", 1).strip(" ;")
    return calculus.tangent(curve, [read_expression(times[1]), read_expression(times[2])] if times else ())


def _lagrange(match):
    return calculus.lagrange(match[1], match[2], positive=bool(re.search(r"\bpositive\b", match[0])))


def _prove(match):
    text = match[1]
    pieces = re.split(r"\bwhere\b", text)
    return calculus.prove(pieces[0], pieces[1] if len(pieces) > 1 else None)


def _system_ode(match):
    forcing = None
    if match[2]:
        forcing = sp.Matrix([read_expression(item) for item in match[2].strip(" []").split(",")])
    return calculus.system_ode(read_matrix(match[1]), forcing)


ROUTES = [
    (r"(?:rank)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.rank(read_matrix(m[1]))),
    (r"(?:echelon)\s*(?:form\s*)?(?:of\s*)?" + MATRIX, lambda m: matrices.echelon(read_matrix(m[1]))),
    (r"(?:det|determinant)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.determinant(read_matrix(m[1]))),
    (r"(?:inverse|inv)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.inverse(read_matrix(m[1]))),
    (r"(?:charpoly|characteristic(?:\s*polynomial)?)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.characteristic(read_matrix(m[1]))),
    (r"eigenvalues\s*(?:of\s*)?" + MATRIX, lambda m: matrices.eigenvalues(read_matrix(m[1]))),
    (r"(?:eigen|eigenvectors)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.eigen(read_matrix(m[1]))),
    (r"diagonali[sz]e\s*(?:of\s*)?" + MATRIX + r"(.*)", _diagonalize),
    (r"(?:summary|matrix)\s*(?:of\s*)?" + MATRIX, lambda m: matrices.summary(read_matrix(m[1]))),
    (MATRIX + r"\s*\^\s*(\d+)", lambda m: matrices.matrix_power(read_matrix(m[1]), int(m[2]))),
    (r"gauss[\s-]*seidel\s*(.+)", _gauss_seidel),
    (r"(?:investigate)\s*(.+)", lambda m: matrices.investigate(m[1])),
    (r"(?:solve\s+)?(?:system\s+)?x\s*'\s*=\s*" + MATRIX + r"\s*\*?\s*x\s*(?:\+\s*(\[.*\]))?", _system_ode),
    (r"partial(?:\s*derivatives?)?\s*(?:of\s*)?(.+)", lambda m: calculus.partials(m[1])),
    (r"total\s*derivative\s*(?:of\s*)?(.+)", lambda m: calculus.total_derivative(m[1])),
    (r"implicit\s*(.+?)\s*;\s*(.+)", lambda m: calculus.implicit_derivative(m[1], m[2])),
    (r"jacobian\s*product\s*(.+)", lambda m: calculus.jacobian_product(m[1])),
    (r"jacobian\s*(?:of\s*)?(.+)", _jacobian),
    (r"extrema\s*(?:of\s*)?(.+)", lambda m: calculus.extrema(m[1])),
    (r"(?:maximi[sz]e|minimi[sz]e)\s*(?:positive\s*)?(.+?)\s*subject\s*to\s*(.+)", _lagrange),
    (r"grad\s*(?:of\s*)?(.+)", _grad),
    (r"div(?:ergence)?\b\s*(?:of\s*)?(.+)", _div),
    (r"curl\s*(?:\(\s*)?curl\s*(.+?)\)?", lambda m: calculus.curl(m[1], 2)),
    (r"curl\s*(?:of\s*)?(.+)", lambda m: calculus.curl(m[1])),
    (r"(?:solenoidal|irrotational)\s*(?:of\s*)?(.+)", lambda m: calculus.field_type(m[1])),
    (r"(?:potential|conservative)\s*(?:of\s*)?(.+)", lambda m: calculus.potential(m[1])),
    (r"directional\s*(?:derivative\s*)?(?:of\s*)?(.+?)\s*at\s*\(([^)]*)\)\s*along\s*(.+)", _directional),
    (r"(?:motion|velocity)\s*(.+)", _motion),
    (r"tangent\s*(.+)", _tangent),
    (r"angle\s*(?:between\s*)?surfaces\s*(.+)", _surfaces),
    (r"prove\s*(.+)", _prove),
    # statistics and probability
    (r"(?:stats?|statistics)\s+(.+)", lambda m: probability.describe_data(m[1])),
    (r"compare\s+(.+)", lambda m: probability.compare(m[1])),
    (r"empirical\s+(.+)", lambda m: probability.empirical(m[1])),
    (r"correlation\s+(.+)", lambda m: probability.correlation(m[1])),
    (r"regression\s*lines\s+(.+)", lambda m: probability.regression_lines(m[1])),
    (r"(?:linear\s*)?regression\s+(.+)", lambda m: probability.linear_fit(m[1])),
    (r"quadratic\s*(?:regression|fit)?\s+(.+)", lambda m: probability.linear_fit(m[1], "quadratic")),
    (r"exponential\s*(?:regression|fit)\s+(.+)", lambda m: probability.linear_fit(m[1], "exponential")),
    (r"prob(?:ability)?\s+(.+)", lambda m: probability.set_probabilities(m[1])),
    (r"bayes\s+(.+)", lambda m: probability.bayes(m[1])),
    (r"transfer\s+(.+)", lambda m: probability.transfer(m[1])),
    (r"distribution\s+(.+)", lambda m: probability.discrete_table(m[1])),
    (r"binomial\s+(.+)", lambda m: probability.binomial(m[1])),
    (r"poisson\s*equation\s*(\d*)\s*P\(\s*X\s*=\s*(\d+)\s*\)\s*=\s*(\d*)\s*P\(\s*X\s*=\s*(\d+)\s*\)", _poisson_equation),
    (r"poisson\s+(.+)", lambda m: probability.poisson(m[1])),
    (r"normal\s+(.+)", lambda m: probability.normal(m[1])),
    (r"uniform\s+(.+)", lambda m: probability.uniform(m[1])),
    (r"pdf\s+(.+)", lambda m: probability.density(m[1])),
    (r"sampling\s+(.+)", lambda m: probability.sampling(m[1])),
    (r"clt\s+(.+)", lambda m: probability.sample_mean(m[1])),
    (r"proportion\s+(.+)", lambda m: probability.sample_proportion(m[1])),
    (r"sum\s*of\s*normals\s+(.+)", lambda m: probability.sum_of_normals(m[1])),
    (r"(?:ci|confidence(?:\s*interval)?)\s+(.+)", lambda m: probability.confidence_interval(m[1])),
    (r"z-?test\s+(.+)", lambda m: probability.mean_test(m[1], "z")),
    (r"t-?test\s+(.+)", lambda m: probability.mean_test(m[1], "t")),
    (r"two-?sample\s+(.+)", lambda m: probability.two_sample_test(m[1])),
    (r"chi-?square\s+(.+)", lambda m: probability.chi_square(m[1])),
    # discrete mathematics
    (r"product\s*rule\s+(.+)", lambda m: discrete.product_rule(m[1])),
    (r"pigeonhole\s+(?:items\s*=\s*)?(\d+)\s*(?:/|boxes\s*=|\s)\s*(\d+)", lambda m: discrete.pigeonhole(int(m[1]), int(m[2]))),
    (r"derangements?\s+(\d+)", lambda m: discrete.derangements(int(m[1]))),
    (r"solutions\s+(.+)", lambda m: discrete.integer_solutions(m[1])),
    (r"inclusion(?:[\s-]*exclusion)?\s+(.+)", lambda m: discrete.inclusion_exclusion(m[1])),
    (r"divisible\s+(\d+)\s*(?:\.\.|to|-)\s*(\d+)\s*by\s*(.+)", _divisible),
    (r"patterns\s+n\s*=\s*(\d+)\s*;?\s*words\s*=\s*(.+)", _patterns),
    (r"rook\s+(.+)", lambda m: discrete.rook(m[1])),
    (r"forbidden\s+(.+)", lambda m: discrete.forbidden(m[1])),
    (r"coefficient\s*(?:of\s*)?x\s*\^?\s*(\d+)\s*in\s*(.+)", lambda m: discrete.coefficient(int(m[1]), m[2])),
    (r"(?:gf|generating\s*function)\s*sequence\s*(?:of\s*)?(.+)", lambda m: discrete.sequence_of(m[1])),
    (r"egf\s*sequence\s*(?:of\s*)?(.+)", lambda m: discrete.sequence_of(m[1], exponential=True)),
    (r"(?:gf|generating\s*function)\s+(?:of\s*)?(.+)", lambda m: discrete.generating_function(m[1])),
    (r"egf\s+(?:of\s*)?(.+)", lambda m: discrete.exponential_gf(m[1])),
    (r"partitions?\s*(?:of\s*)?(\d+)", lambda m: discrete.partitions(int(m[1]))),
    (r"arrangements\s+([A-Za-z]+)\s+(\d+)", _word_arrangements),
    (r"arrangements\s*counts\s*=\s*(.+?)\s*;?\s*r\s*=\s*(\d+)", _counts_arrangements),
    (r"recurrence\s+(.+)", lambda m: discrete.recurrence(m[1])),
    (r"growth\s+start\s*=\s*(\d+)\s+percent\s*=\s*(\d+)\s+periods\s*=\s*(\d+)", lambda m: discrete.growth(int(m[1]), m[2], int(m[3]))),
    (r"hanoi\s*(\d*)", lambda m: discrete.hanoi(int(m[1]) if m[1] else None)),
    (r"handshakes?\s*(\d*)", lambda m: discrete.handshakes(int(m[1]) if m[1] else None)),
    # graphs and trees
    (r"chromatic\s*(?:polynomial|number)?\s*(?:of\s*)?(.+)", _chromatic),
    (r"planar(?:ity)?\s*(?:of\s*)?(.+)", lambda m: graphs.planarity(m[1])),
    (r"regions\s*(?:degrees\s*=\s*)?(.+)", lambda m: graphs.regions(m[1])),
    (r"isomorphic\s+(.+)", _isomorphic),
    (r"spanning\s*trees?\s*(?:of\s*)?(.+)", lambda m: graphs.spanning_trees(m[1])),
    (r"prim\s+(.+)", _prim),
    (r"kruskal\s+(.+)", lambda m: graphs.kruskal(m[1])),
    (r"huffman\s+(.+)", _huffman),
    (r"merge\s*sort\s+(.+)", lambda m: graphs.merge_sort([int(v) for v in read_numbers(m[1])])),
    (r"leaves\s+(.+)", lambda m: graphs.tree_leaves(m[1])),
    (r"cords\s+devices\s*=\s*(\d+)\s+outlets\s*=\s*(\d+)", lambda m: graphs.extension_cords(int(m[1]), int(m[2]))),
    (r"walk\s+(.+)", lambda m: graphs.walk_kind(m[1])),
    (r"graph\s+(.+)", lambda m: graphs.summary(m[1])),
    (r"solve\s+(.+)", lambda m: _solve(m[1])),
]
COMPILED = [(re.compile(pattern, re.IGNORECASE | re.DOTALL), handler) for pattern, handler in ROUTES]
ODE_MARK = re.compile(r"y'|d\^?\d*\s*y\s*/\s*d[a-z]|\(d\^?\d*y\)/\(d[a-z]|\bD\b|\bD\^|\d?D\s*[+\-)]")


def _looks_like_system(text):
    parts = parts_of(text)
    return len(parts) > 1 and all("=" in part for part in parts)


def _solve(text):
    if ODE_MARK.search(text):
        return calculus.solve_ode(text)
    if _looks_like_system(text):
        return matrices.solve_system(text)
    return _plain(text)


def _plain(text):
    if _is_sentence(text) or _has_long_word(text):
        raise ValueError("I could not understand this question. Try writing it as a formula, or in the wording your paper uses")
    answer = solve(text)
    kind = "Equation" if answer.solutions is not None else "Calculation"
    return Result(kind, [describe(answer)], explain(text, answer), parsed=answer.parsed)


def _dispatch(text):
    for pattern, handler in COMPILED:
        match = pattern.fullmatch(text)
        if match:
            return handler(match)
    if re.search(r"(∫|\bint\b)", text):
        return calculus.integral(text)
    return _solve(text)


ENGLISH = re.compile(r"\b(the|that|what|which|how|many|with|where|using|from|this|these|is|are|if|in|of|to|for|and|by|its)\b", re.IGNORECASE)
OPENING = re.compile(r"^\s*(?:(?:find|show|prove that|determine|calculate|compute|evaluate|given|test|investigate the|employ|using|construct|consider|"
                     r"there (?:are|is)|how many|what|suppose|write|reduce|obtain|verify|check|solve the|diagonali[sz]e the)\b"
                     r"|(?:a|an|the|in|from|if|let)\s+[a-zA-Z]{2,})", re.IGNORECASE)


def _is_sentence(text):
    """A question in words: it opens like one, or has several ordinary words in it."""
    plain = re.sub(r"\[.*?\]", " ", text)
    return bool(OPENING.match(plain)) or len(ENGLISH.findall(plain)) >= 2


def _has_long_word(text):
    """A run of letters too long to be a product of variables."""
    return bool(re.search(r"\b(?!(?:cosec|arcsin|arccos|arctan|partial|gamma)\b)[a-zA-Z]{6,}\b", re.sub(r"\[.*?\]", " ", text)))


def _english_answer(text):
    """Word problems. Returns a Result or None."""
    commands = english.translate(text)
    if not commands:
        return None
    results = [_dispatch(command) for command in commands]
    merged = Result(results[0].topic, [line for r in results for line in r.answer], [line for r in results for line in r.steps])
    merged.read_as = "; then ".join(commands)
    return merged


def answer_question(text):
    """The answer to one question, within a time limit. Raises ValueError with a clear message if it cannot help."""
    text = text.strip()
    if text.endswith(".") and not re.search(r"(\d|\.)\.$", text):  # a full stop, not 5. or the dots in 1,2,3,...
        text = text[:-1].strip()
    if not text:
        raise ValueError("Nothing to solve")

    def work():
        meaning = glossary.define(text)
        if meaning:
            return Result("Definition", [meaning])
        if _is_sentence(text):
            result = _english_answer(text)
            if result is not None:
                return result
        return _dispatch(text)

    try:
        return time_limited(work, 60)
    except TimeoutError as error:
        raise ValueError(str(error)) from None
