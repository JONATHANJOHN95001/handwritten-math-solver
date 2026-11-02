import numpy as np

from readings import can_read, other_readings, read_board, to_latex
from reader import segment

LABELS = ([str(d) for d in range(10)] + ["plus", "minus", "x", "slash"] + list("abcdefghijklmnopqrstuvwyz")
          + ["lparen", "rparen", "arrow", "inf", "integral", "sqrt", "pi", "lbracket", "rbracket", "partial", "fact", "lambda", "mu", "theta"])


class FakeModel:
    """Says each symbol is whatever the list says, in writing order, with the given second guess."""

    def __init__(self, answers):
        self.answers = answers

    def __call__(self, images, training=False):
        rows = []
        for k in range(len(images)):
            probs = np.full(len(LABELS), 1e-4, np.float32)
            first, second = self.answers[k % len(self.answers)]
            probs[LABELS.index(first)] = 0.8
            if second:
                probs[LABELS.index(second)] = 0.15
            rows.append(probs)
        return FakeResult(np.array(rows))


class FakeResult:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def _board():
    mask = np.zeros((200, 400), np.uint8)
    mask[40:160, 40:50] = 255
    mask[40:160, 120:130] = 255
    return mask


def test_the_readers_reading_comes_first_then_the_second_guesses():
    mask = _board()
    model = FakeModel([("2", "z"), ("5", None)])
    symbols, best = read_board(mask, None, model, LABELS)
    assert best == "25"
    assert other_readings(symbols, best, LABELS, mask, None, model) == ["25", "z5"]


def test_a_letter_kept_for_words_can_be_chosen_as_a_second_guess():
    mask = _board()
    model = FakeModel([("5", "s"), ("2", None)])
    symbols, best = read_board(mask, None, model, LABELS)
    assert other_readings(symbols, best, LABELS, mask, None, model) == ["52", "s2"]


def test_the_readings_are_limited_and_have_no_repeats():
    mask = _board()
    model = FakeModel([("2", "z"), ("5", "s")])
    symbols, best = read_board(mask, None, model, LABELS)
    readings = other_readings(symbols, best, LABELS, mask, None, model)
    assert len(readings) == len(set(readings)) <= 4 and readings[0] == "25"


def test_a_reading_from_the_pixels_is_offered_when_the_strokes_group_differently():
    mask = np.zeros((200, 400), np.uint8)
    top, bottom = np.zeros_like(mask), np.zeros_like(mask)
    top[50:150, 40:60] = 255
    bottom[120:150, 60:160] = 255  # they touch: two strokes, but one blob of ink
    mask = top | bottom
    model = FakeModel([("x", None)])
    symbols = segment(mask, [top, bottom])
    assert len(symbols) == 2
    symbols, best = read_board(mask, [top, bottom], model, LABELS)
    readings = other_readings(symbols, best, LABELS, mask, [top, bottom], model)
    assert readings[0] == best and "x" in readings


def test_can_read_and_to_latex():
    assert can_read("2x+5=3") and can_read("x+y=3; x-y=1") and can_read("rank[[1,2],[3,4]]")
    assert not can_read("2x+=3") and not can_read("1/0")
    assert to_latex("2x+5=3") == "2x+5=3"
    assert to_latex("rank[[1,2],[3,4]]") is None
    assert to_latex("x+y=3; x-y=1").count(r"\quad") == 1


def test_a_reading_is_shown_exactly_as_it_was_read():
    assert to_latex("2xt5=3") == "2xt5=3"
    assert to_latex("1.5·4") == r"1.5\cdot 4"
    assert to_latex("5+x=3") == "5+x=3"
    assert to_latex("log_2(8)") == r"\log _{2}\left(8\right)"
    assert to_latex("ln(e)") == r"\ln \left(e\right)"
    assert to_latex("(3)/(4)") == r"\frac{3}{4}"
    assert to_latex("2^(10)") == "2^{10}"
    assert to_latex("√(25)") == r"\sqrt{25}"
    assert to_latex("23c4") == r"\binom{23}{4}"
    assert to_latex("lim_(x→0) (sin(x))/(x)") == r"\lim _{x\to 0}\ \frac{\sin \left(x\right)}{x}"
    assert to_latex("2x+=3") is None
