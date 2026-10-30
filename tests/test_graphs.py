"""Real questions from Graph Theory and Trees, with the graphs typed as text."""
import networkx as nx
import pytest
import sympy as sp

from questions import answer_question


def text_of(question):
    return "\n".join(answer_question(question).answer)


def test_k5_is_not_planar_end_sem_5a():
    assert "not planar" in text_of("planar K5")


def test_k33_is_not_planar():
    assert "not planar" in text_of("planar K3,3")


def test_planar_graph_and_euler():
    answer = text_of("planar a-b, b-c, c-d, d-a, a-c")
    assert "is planar" in answer and "Regions: r = e - v + 2 = 3" in answer


def test_regions_assignment4_q8():
    assert "Regions: r = 10" in text_of("regions degrees=2,2,3,3,3,4,5,6,6")


def test_chromatic_k23_assignment4_q7():
    lam = sp.Symbol("λ")
    line = answer_question("chromatic K2,3").answer[0].split("=", 1)[1].replace("·", "*").replace("^", "**")
    polynomial = sp.sympify(line, locals={"λ": lam})
    # same colour on the two: λ(λ-1)^3; different: λ(λ-1)(λ-2)^3
    assert sp.expand(polynomial - (lam * (lam - 1) ** 3 + lam * (lam - 1) * (lam - 2) ** 3)) == 0


def test_chromatic_number_and_colourings():
    answer = text_of("chromatic a-b, b-c, c-a colours=5")
    assert "Chromatic number = 3" in answer and "With 5 colours: 60" in answer


def test_isomorphic_assignment4_q9():
    assert "are isomorphic" in text_of("isomorphic a-b, b-c, c-a; 1-2, 2-3, 3-1")
    assert "not isomorphic" in text_of("isomorphic a-b, b-c, c-a; 1-2, 2-3, 3-4")


def test_spanning_trees_end_sem_1e():
    answer = text_of("spanning trees K4")
    assert "Number of spanning trees = 16" in answer and "Tree 4:" in answer


EDGES = "a-b:4, a-c:2, b-c:1, b-d:5, c-d:8, c-e:10, d-e:2, d-f:6, e-f:3"


def test_prim_assignment5_q7():
    graph = nx.Graph()
    for part in EDGES.split(", "):
        ends, weight = part.split(":")
        u, v = ends.split("-")
        graph.add_edge(u, v, weight=int(weight))
    expected = nx.minimum_spanning_tree(graph).size(weight="weight")
    assert f"weight = {expected:g}" in text_of(f"prim {EDGES}")


def test_kruskal_assignment5_q9():
    assert text_of(f"kruskal {EDGES}").split("\n")[0] == text_of(f"prim {EDGES}").split("\n")[0]


def test_huffman_assignment5_q5():
    frequencies = {"a": 20, "o": 28, "q": 4, "u": 17, "y": 12, "z": 7}
    result = text_of("huffman a:20, o:28, q:4, u:17, y:12, z:7")
    # the merges are 11, 23, 37, 51, 88, and the weighted length is their sum
    assert "Total weighted length = 210" in result
    codes = dict(line.split(": ") for line in result.split("\n")[:6])
    for a in codes.values():
        for b in codes.values():
            assert a == b or not b.startswith(a)
    assert set(codes) == set(frequencies)


def test_huffman_message_assignment5_q6():
    result = text_of("huffman text=\"ROAD IS GOOD\"")
    assert "space" in result and "O:" in result


def test_merge_sort_assignment5_q3():
    assert "[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15]" in text_of("merge sort 12, 11, 2, 3, 1, 5, 6, 7, 10, 9, 4, 15")


def test_tree_leaves_assignment5_q1():
    assert "Leaves = 10" in text_of("leaves 2:4, 3:1, 4:2, 5:1")


def test_extension_cords_assignment5_q2():
    assert "8 extension cords" in text_of("cords devices=25 outlets=4")


@pytest.mark.parametrize("walk, kind", [
    ("v1 e1 v2 e2 v3 e2 v2", "open walk"),
    ("v4 e7 v1 e1 v2 e2 v3 e3 v4 e4 v5", "trail"),
    ("v1 e1 v2 e2 v3 e3 v4 e4 v5", "path"),
    ("v1 e1 v2 e2 v3 e3 v4 e7 v1", "cycle"),
])
def test_walk_kinds_assignment4_q3(walk, kind):
    assert kind in text_of(f"walk {walk}")


def test_drawn_graph_gets_a_clear_message():
    with pytest.raises(ValueError, match="edges"):
        answer_question("planar the graph shown below")
