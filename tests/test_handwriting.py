import numpy as np

from check_handwriting import comparable, edit_distance, expected_text, stroke_layers
from prepare_handwriting import class_index, draw_strokes
from train import CLASSES


def test_class_index_joins_capitals_to_small_letters_but_not_the_bar_shaped_i():
    assert CLASSES[class_index("a")] == "a" and CLASSES[class_index("A")] == "a"
    assert CLASSES[class_index("7")] == "7"
    assert CLASSES[class_index("\\times")] == "x" and CLASSES[class_index("X")] == "x"
    assert CLASSES[class_index("\\sqrt{}")] == "sqrt" and CLASSES[class_index("\\to")] == "arrow"
    assert class_index("I") is None and class_index("\\alpha") is None and class_index("=") is None


def test_draw_strokes_shows_every_stroke_at_a_board_size():
    rng = np.random.default_rng(1)
    bar, dot = np.array([[10.0, 500.0], [10.0, 900.0]]), np.array([[10.0, 300.0]])  # an i: a stem and a one-point stroke
    mask = draw_strokes([bar, dot], rng)
    assert 50 <= max(mask.shape) <= 160
    rows = np.flatnonzero(mask.any(axis=1))
    assert rows[0] < mask.shape[0] * 0.2 and len(rows) < mask.shape[0]  # the dot is there, with a gap below it


def test_expected_text_writes_latex_the_way_the_reader_does():
    assert expected_text("1-e^{-1}=0.61") == "1-e^(-1)=0.61"
    assert expected_text("x^{2}+y_{1}") == "x^2+y_1"
    assert expected_text("\\frac{a}{b+1}") == "a/(b+1)"
    assert expected_text("\\sqrt{25}=5") == "√(25)=5"
    assert expected_text("\\lim_{x\\to0}x") == "lim_(x→0)x"
    assert expected_text("\\int_{0}^{1}xdx") == "∫_0^1xdx"
    assert expected_text("2\\times3") == "2x3"
    assert expected_text("A+B") == "a+b"


def test_expected_text_refuses_what_the_board_cannot_write():
    assert expected_text("\\alpha+1") is None
    assert expected_text("I+1") is None
    assert expected_text("\\sum_{n=0}^{\\infty}") is None
    assert expected_text("\\begin{matrix}1\\end{matrix}") is None
    assert expected_text("f+1") is None  # f is kept for function words
    assert expected_text("sin(x)+1") == "sinx+1"  # a word in the text is fine, and brackets round one symbol do not count


def test_comparable_ignores_only_how_a_reading_is_written():
    assert comparable("2×3 ÷ 4") == "2x3/4"
    assert comparable("(1)/(2)") == "1/2"
    assert comparable("(12)/(2)") == "(12)/2"
    assert comparable("2x+3") != comparable("2x-3")


def test_edit_distance():
    assert edit_distance("x+1", "x+1") == 0
    assert edit_distance("x+1", "x-1") == 1
    assert edit_distance("", "abc") == 3


def test_stroke_layers_gives_one_layer_per_stroke_on_one_board():
    strokes = [[[0, 0], [0, 10]], [[20, 0], [20, 10]], [[40, 5]]]
    mask, layers = stroke_layers(strokes)
    assert len(layers) == 3 and all(layer.shape == mask.shape for layer in layers)
    assert all(layer.any() for layer in layers)
    assert stroke_layers([[[0, 0], [0, 10]], [[100000, 0], [100000, 10]]]) == (None, None)
