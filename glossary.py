"""Short definitions and statements for the 'Define' and 'State' questions in the three courses."""
import re

ENTRIES = [
    # Matrix Algebra and Calculus
    (r"rank\b.*matrix|rank of a matrix", "The rank of a matrix is the largest number of linearly independent rows or columns. It equals the number of non-zero rows in its row echelon form, and the order of its largest non-zero minor."),
    (r"cauchy", "Cauchy's linear differential equation: xⁿ dⁿy/dxⁿ + a₁ xⁿ⁻¹ dⁿ⁻¹y/dxⁿ⁻¹ + ... + aₙ y = f(x), where a₁, ..., aₙ are constants. Putting x = eᵗ turns it into an equation with constant coefficients."),
    (r"reduction formula", "∫₀^(π/2) sinᵐx cosⁿx dx = [(m-1)(m-3)...][(n-1)(n-3)...] / [(m+n)(m+n-2)...] × k, where k = π/2 when m and n are both even, and k = 1 otherwise."),
    (r"eigen\s*values?\b.*\b(define|what)|define.*eigen", "An eigenvalue of a square matrix A is a number λ with AX = λX for some non-zero vector X, which is its eigenvector. The eigenvalues are the roots of |A - λI| = 0."),
    # Statistics and Probability
    (r"correlation coefficient", "The correlation coefficient r measures the strength and direction of the straight-line relationship between two variables: r = cov(x,y)/(σx σy). Its range is -1 ≤ r ≤ 1."),
    (r"regression lines?", "Regression line of y on x: y - ȳ = r(σy/σx)(x - x̄). Regression line of x on y: x - x̄ = r(σx/σy)(y - ȳ)."),
    (r"statistic\b.*parameter|parameter\b.*statistic", "A parameter describes a whole population, such as the true mean height of all students. A statistic describes a sample, such as the mean height of 50 students chosen from them."),
    (r"non-?probability sampling", "Non-probability sampling chooses units without known chances. Convenience: whoever is easy to reach. Judgement: chosen by an expert. Quota: fixed numbers from each group, chosen freely. Snowball: members suggest further members."),
    (r"probability sampling", "Probability sampling gives every unit a known chance. Simple random: every sample is equally likely. Systematic: every kth unit. Stratified: random samples from each group. Cluster: all units from randomly chosen groups."),
    (r"simple random sampling", "In simple random sampling every sample of the same size is equally likely. It is simple and unbiased, but it needs a full list of the population and can miss small groups."),
    (r"central limit theorem", "Central Limit Theorem: for large n, the mean of a random sample from any population with mean μ and finite s.d. σ is approximately normal with mean μ and s.d. σ/√n. It lets us use the normal distribution for means whatever the population's shape."),
    (r"type i\b.*type ii|type 1\b.*type 2", "A Type I error rejects a true null hypothesis, with chance α: convicting an innocent person. A Type II error keeps a false null hypothesis, with chance β: letting a guilty person go."),
    (r"null\b.*alternative", "The null hypothesis H0 is the claim being tested, usually of no effect: μ = 100. The alternative H1 is what we accept if H0 is rejected: μ ≠ 100, μ > 100 or μ < 100."),
    (r"procedure of hypothesis|hypothesis testing", "State H0 and H1. Choose the significance level α. Choose the test statistic. Find the critical value. Work out the statistic from the sample. Reject H0 if the statistic falls in the critical region, otherwise do not reject it."),
    # Discrete Mathematics and Graph Theory
    (r"product rule", "Product rule: if one task can be done in m ways and a second in n ways whatever the first choice was, the two together can be done in m × n ways."),
    (r"pigeonhole", "Pigeonhole principle: if m objects go into n boxes with m > n, at least one box holds two or more. More generally, some box holds at least ⌈m/n⌉ objects."),
    (r"complete graph", "A complete graph Kₙ has n vertices and an edge between every pair of them, so n(n-1)/2 edges. K₄ has 4 vertices and 6 edges."),
    (r"simple graph", "A simple graph has no loops and no two edges joining the same pair of vertices."),
    (r"multi\s?graph", "A multigraph may have more than one edge between the same pair of vertices."),
    (r"isolated vertex", "An isolated vertex has degree 0: no edge touches it."),
    (r"pend[ae]nt vertex", "A pendant vertex has degree 1: exactly one edge touches it."),
    (r"(vertex )?degree", "The degree of a vertex is the number of edges touching it, with a loop counted twice."),
    (r"\bwalk\b", "A walk is a sequence of vertices and edges where each edge joins the vertices on either side of it."),
    (r"\btrail\b", "A trail is a walk with no repeated edge."),
    (r"\bpath\b", "A path is a walk with no repeated vertex."),
    (r"\bcircuit\b", "A circuit is a closed trail: it ends where it starts and repeats no edge."),
    (r"\bcycle\b", "A cycle is a closed path: it ends where it starts and repeats no other vertex."),
    (r"chromatic number", "The chromatic number of a graph is the smallest number of colours that can colour its vertices so that adjacent vertices differ."),
    (r"euler'?s formula", "Euler's formula: for a connected planar graph, v - e + r = 2, with v vertices, e edges and r regions."),
    (r"second-?order linear recurrence|general solution of a homogeneous", "For a(n) + p a(n-1) + q a(n-2) = 0 with roots r₁, r₂ of its characteristic equation: real and distinct, a(n) = A r₁ⁿ + B r₂ⁿ. Real and repeated, a(n) = (A + Bn) r₁ⁿ. Complex, r = R(cos θ ± i sin θ), a(n) = Rⁿ(A cos nθ + B sin nθ)."),
    (r"spanning tree", "A spanning tree of a connected graph is a subgraph that is a tree and includes every vertex."),
]
ASKS = re.compile(r"^\s*(define|state|what is|what are|write the (general form|reduction formula|equations)|discuss|explain|describe|differentiate)\b", re.IGNORECASE)


def define(text):
    """The stored answer to a theory question, or None. Only for questions that ask for a definition or statement."""
    if not ASKS.match(text):
        return None
    for pattern, answer in ENTRIES:
        if re.search(pattern, text, re.IGNORECASE):
            return answer
    return None
