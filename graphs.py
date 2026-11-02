"""Graph Theory and Trees, for graphs typed as text. A drawn figure has to be typed as its list of edges."""
import heapq
import re

import networkx as nx
import numpy as np
import sympy as sp

from results import Result, parts_of, read_numbers, show

LAM = sp.Symbol("λ")
NAMED = re.compile(r"\b(K|C|P|W)\s*_?\s*(\d+)(?:\s*,\s*(\d+))?\b")


def read_graph(text):
    """'a-b, b-c, c-a' or 'K5', 'K2,3', 'C4', 'P4', 'W5'. Weights: 'a-b:3'."""
    named = NAMED.search(text)
    if named and "-" not in text:
        kind, first, second = named[1], int(named[2]), named[3]
        if kind == "K" and second:
            return nx.complete_bipartite_graph(first, int(second))
        return {"K": nx.complete_graph, "C": nx.cycle_graph, "P": nx.path_graph, "W": nx.wheel_graph}[kind](first)
    graph = nx.Graph()
    edges = re.findall(r"([\w]+)\s*-\s*([\w]+)(?:\s*:\s*([\d.]+))?", text)
    if not edges:
        raise ValueError("Type the graph as its edges, like a-b, b-c, c-a. A drawn figure cannot be read yet")
    for a, b, weight in edges:
        graph.add_edge(a, b, weight=float(weight) if weight else 1.0)
    return graph


def chromatic_polynomial(graph):
    """By deletion and contraction: P(G) = P(G - e) - P(G / e)."""
    def rec(g):
        if g.number_of_edges() == 0:
            return LAM ** g.number_of_nodes()
        u, v = next(iter(g.edges()))
        deleted = g.copy()
        deleted.remove_edge(u, v)
        contracted = nx.contracted_nodes(g, u, v, self_loops=False)
        return rec(deleted) - rec(nx.Graph(contracted))
    return sp.factor(rec(nx.Graph(graph)))


def chromatic(text, colours=None):
    graph = read_graph(text)
    poly = chromatic_polynomial(graph)
    number = next(k for k in range(1, graph.number_of_nodes() + 2) if poly.subs(LAM, k) > 0)
    answer = [f"P(λ) = {show(poly)}", f"Chromatic number = {number}"]
    if colours:
        answer.append(f"With {colours} colours: {poly.subs(LAM, colours)} proper colourings")
    values = [(k, poly.subs(LAM, k)) for k in range(1, number + 1)]
    steps = [f"{graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges",
             "Deletion and contraction: P(G) = P(G - e) - P(G / e), repeated until no edges are left, where P = λ^(number of vertices)", f"That gives P(λ) = {show(poly)}",
             "Try small numbers of colours: " + ", ".join(f"P({k}) = {v}" for k, v in values), f"The first number with a colouring is {number}, so that is the chromatic number"]
    if colours:
        steps.append(f"With {colours} colours, put λ = {colours}: {poly.subs(LAM, colours)}")
    return Result("Graph colouring", answer, steps)


def planarity(text):
    graph = read_graph(text)
    planar, _ = nx.check_planarity(graph)
    v, e = graph.number_of_nodes(), graph.number_of_edges()
    steps = [f"v = {v}, e = {e}"]
    if planar:
        answer = ["The graph is planar"]
        if nx.is_connected(graph):
            r = e - v + 2
            answer.append(f"Regions: r = e - v + 2 = {r}")
            steps.append(f"Euler's formula: v - e + r = 2, so r = e - v + 2 = {e} - {v} + 2 = {r}")
    else:
        answer = ["The graph is not planar"]
        if e > 3 * v - 6:
            steps.append(f"For a planar graph e ≤ 3v - 6 = {3 * v - 6}, but e = {e}")
        else:
            steps.append("It contains a subdivision of K5 or K3,3 (Kuratowski's theorem)")
    return Result("Planarity", answer, steps)


def regions(text):
    """From a list of vertex degrees of a connected planar graph."""
    degrees = [int(d) for d in read_numbers(text)]
    if sum(degrees) % 2:
        raise ValueError("The degrees must add up to an even number")
    v, e = len(degrees), sum(degrees) // 2
    return Result("Euler's formula", [f"Regions: r = {e - v + 2}"], [f"Sum of degrees = {sum(degrees)} = 2e, so e = {e}", f"There are v = {v} vertices", f"Euler's formula: v - e + r = 2, so r = e - v + 2 = {e} - {v} + 2 = {e - v + 2}"])


def isomorphic(first, second):
    a, b = read_graph(first), read_graph(second)
    same = nx.is_isomorphic(a, b)
    steps = [f"Vertices: {a.number_of_nodes()} and {b.number_of_nodes()}. Edges: {a.number_of_edges()} and {b.number_of_edges()}", f"Degrees: {sorted(d for _, d in a.degree())} and {sorted(d for _, d in b.degree())}"]
    if same:
        matcher = nx.isomorphism.GraphMatcher(a, b)
        matcher.is_isomorphic()
        steps.append("Every edge is kept when the vertices are matched: " + ", ".join(f"{u} → {v}" for u, v in matcher.mapping.items()))
    elif sorted(d for _, d in a.degree()) != sorted(d for _, d in b.degree()):
        steps.append("The degrees differ, and an isomorphism must keep degrees")
    else:
        steps.append("The counts agree, but no matching of the vertices keeps every edge")
    return Result("Isomorphism", ["The graphs are isomorphic" if same else "The graphs are not isomorphic"], steps)


def spanning_trees(text, show_count=4):
    graph = read_graph(text)
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    count = round(np.linalg.det(laplacian[1:, 1:]))
    found = []
    for tree in nx.SpanningTreeIterator(nx.Graph(graph)):
        found.append(", ".join(f"{u}-{v}" for u, v in tree.edges()))
        if len(found) == show_count:
            break
    return Result("Spanning trees", [f"Number of spanning trees = {count}"] + [f"Tree {n + 1}: {edges}" for n, edges in enumerate(found)], ["Kirchhoff's matrix-tree theorem: form the Laplacian, with degrees on the diagonal and -1 for each edge",
                   "\n".join(["Laplacian:"] + ["[ " + "  ".join(f"{int(v):>2}" for v in row) + " ]" for row in laplacian]),
                   f"Remove one row and the same column, and take the determinant: {count}"])


def _edge_text(a, b):
    return f"{a}-{b}"


def prim(text, start=None):
    graph = read_graph(text)
    root = start if start in graph else sorted(graph.nodes(), key=str)[0]
    seen, heap, chosen, total = {root}, [], [], 0
    for b in graph[root]:
        heapq.heappush(heap, (graph[root][b]["weight"], str(root), str(b), root, b))
    while heap:
        w, _, _, a, b = heapq.heappop(heap)
        if b in seen:
            continue
        seen.add(b)
        chosen.append((a, b, w))
        total += w
        for c in graph[b]:
            if c not in seen:
                heapq.heappush(heap, (graph[b][c]["weight"], str(b), str(c), b, c))
    if len(seen) != graph.number_of_nodes():
        raise ValueError("The graph is not connected, so it has no spanning tree")
    return Result("Prim's algorithm", [f"Minimal spanning tree weight = {total:g}", "Edges: " + ", ".join(_edge_text(a, b) for a, b, _ in chosen)],
                  [f"Start at {root}. Each time, add the lightest edge from the tree to a new vertex",
                   *[f"Step {n + 1}: add {_edge_text(a, b)} (weight {w:g})" for n, (a, b, w) in enumerate(chosen)],
                   f"Add the weights: {' + '.join(f'{w:g}' for _, _, w in chosen)} = {total:g}"])


def kruskal(text):
    graph = read_graph(text)
    parent = {v: v for v in graph}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    chosen, steps, total = [], [], 0
    for a, b, data in sorted(graph.edges(data=True), key=lambda e: (e[2]["weight"], str(e[0]), str(e[1]))):
        w = data["weight"]
        if find(a) == find(b):
            steps.append(f"Skip {_edge_text(a, b)} (weight {w:g}): it makes a cycle")
            continue
        parent[find(a)] = find(b)
        chosen.append((a, b))
        total += w
        steps.append(f"Add {_edge_text(a, b)} (weight {w:g})")
    if len(chosen) != graph.number_of_nodes() - 1:
        raise ValueError("The graph is not connected, so it has no spanning tree")
    weights = [graph[a][b]["weight"] for a, b in chosen]
    return Result("Kruskal's algorithm", [f"Minimal spanning tree weight = {total:g}", "Edges: " + ", ".join(_edge_text(a, b) for a, b in chosen)],
                  ["Take the edges from lightest to heaviest, keeping each one that makes no cycle", *steps, f"Add the weights: {' + '.join(f'{w:g}' for w in weights)} = {total:g}"])


def huffman(frequencies):
    """Optimal prefix code, merging the two smallest weights each time."""
    heap = [(weight, n, symbol) for n, (symbol, weight) in enumerate(frequencies.items())]
    heapq.heapify(heap)
    codes = {symbol: "" for symbol in frequencies}
    members = {symbol: [symbol] for symbol in frequencies}
    next_id = len(heap)
    steps = []
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap)
        w2, _, b = heapq.heappop(heap)
        for s in members[a]:
            codes[s] = "0" + codes[s]
        for s in members[b]:
            codes[s] = "1" + codes[s]
        name = f"node{next_id}"
        members[name] = members[a] + members[b]
        heapq.heappush(heap, (w1 + w2, next_id, name))
        steps.append(f"Merge weights {w1:g} and {w2:g} into {w1 + w2:g}")
        next_id += 1
    length = sum(frequencies[s] * len(codes[s]) for s in frequencies)
    answer = [f"{s}: {codes[s]}" for s in frequencies] + [f"Total weighted length = {length:g}"]
    steps = ["Keep merging the two smallest weights, with 0 on one branch and 1 on the other", *steps, "Read each code from the top of the tree down to the symbol",
             "Weighted length = Σ weight × code length: " + " + ".join(f"{frequencies[s]:g}×{len(codes[s])}" for s in frequencies) + f" = {length:g}"]
    return Result("Optimal prefix code (Huffman)", answer, steps)


def huffman_text(message):
    counts = {}
    for ch in message:
        counts[ch] = counts.get(ch, 0) + 1
    label = {" ": "space"}
    return huffman({label.get(ch, ch): n for ch, n in counts.items()})


def merge_sort(values):
    steps = []

    def run(items):
        if len(items) < 2:
            return items
        middle = len(items) // 2
        left, right = run(items[:middle]), run(items[middle:])
        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged += left[i:] + right[j:]
        steps.append(f"Merge {left} and {right} into {merged}")
        return merged

    result = run(list(values))
    return Result("Merge sort", [str(result)], [f"Split {list(values)} in half again and again, then merge the sorted halves back, smallest first", *steps])


def tree_leaves(text):
    """leaves 2:4, 3:1, 4:2, 5:1 means four vertices of degree 2, one of degree 3 and so on."""
    pairs = re.findall(r"(\d+)\s*:\s*(\d+)", text)
    if not pairs:
        raise ValueError("Give degree:count pairs, like 2:4, 3:1")
    inner = sum(int(c) for _, c in pairs)
    degree_sum = sum(int(d) * int(c) for d, c in pairs)
    leaves = degree_sum - 2 * inner + 2
    return Result("Leaves of a tree", [f"Leaves = {leaves}"], [f"The vertices that are not leaves number {inner}, and their degrees add up to {degree_sum}", "A tree with n vertices has n - 1 edges, so its degrees add up to 2(n - 1), with n = inner vertices + L leaves, each leaf of degree 1",
                                                     f"{degree_sum} + L = 2({inner} + L - 1)", f"Solve for L: L = {degree_sum} - 2 × {inner} + 2 = {leaves}"])


def extension_cords(devices, outlets):
    """Cords with m outlets forming an m-ary tree: internal nodes = (leaves - 1)/(m - 1)."""
    cords = sp.ceiling(sp.Rational(devices - 1, outlets - 1))
    return Result("m-ary tree", [f"{cords} extension cords"], [f"Each cord is an inner node with {outlets} branches, and the {devices} devices are the leaves", f"Internal nodes = (leaves - 1)/(m - 1) = ({devices} - 1)/({outlets} - 1) = {show(sp.Rational(devices - 1, outlets - 1))}", f"Round up to whole cords: {cords}"])


def walk_kind(text):
    """walk v1 e1 v2 e2 v3: open or closed, and whether it is a trail, path, circuit or cycle."""
    items = re.split(r"[\s,]+", text.strip())
    vertices, edges = items[0::2], items[1::2]
    closed = vertices[0] == vertices[-1]
    trail = len(edges) == len(set(edges))
    middle = vertices[:-1] if closed else vertices
    path = len(middle) == len(set(middle))
    if closed:
        kind = "a cycle" if trail and path and edges else "a circuit" if trail and edges else "a closed walk"
    else:
        kind = "a path" if trail and path else "a trail" if trail else "an open walk"
    return Result("Nature of a walk", [f"This is {kind}"], [f"Vertices: {' '.join(vertices)}", f"Edges: {' '.join(edges)}", f"Length: {len(edges)}",
                                                           f"It {'ends where it starts' if closed else 'ends away from its start'}, so it is {'closed' if closed else 'open'}",
                                                           f"Edges {'do not repeat' if trail else 'repeat'} and vertices {'do not repeat' if path else 'repeat'}, so this is {kind}"])


def summary(text):
    graph = read_graph(text)
    degrees = dict(graph.degree())
    answer = [f"{graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges", "Degrees: " + ", ".join(f"{v}: {d}" for v, d in sorted(degrees.items(), key=lambda p: str(p[0])))]
    return Result("Graph summary", answer, [f"Count the vertices and edges: v = {graph.number_of_nodes()}, e = {graph.number_of_edges()}", "The degree of a vertex is the number of edges at it",
                                            f"Check: the degrees add up to {sum(degrees.values())}, which is 2e"])


def euler_check(v, e, r):
    return Result("Euler's formula", [f"v - e + r = {v - e + r}" + (", which is 2, so the formula holds" if v - e + r == 2 else ", which is not 2")], [f"Put in the numbers: {v} - {e} + {r} = {v - e + r}", "A connected planar graph must give 2"])


def parts(text):
    return parts_of(text)
