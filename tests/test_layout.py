import numpy as np

from layout import to_text

LABELS = ([str(d) for d in range(10)] + ["plus", "minus", "x", "slash"] + list("abcdefghijklmnopqrstuvwyz")
          + ["lparen", "rparen", "arrow", "inf", "integral", "sqrt", "pi", "lbracket", "rbracket", "partial", "fact", "lambda", "mu", "theta"])


def sym(x, label, conf=0.9, alt=None, y=0, w=40, h=100):
    probs = np.full(len(LABELS), 1e-4)
    probs[LABELS.index(label)] = conf
    if alt:
        probs[LABELS.index(alt)] = 1 - conf
    return {"box": (x, y, w, h), "kind": None, "probs": probs}


def shape(x, kind, y=40, w=30, h=20):
    return {"box": (x, y, w, h), "kind": kind}


def word(x, letters, y=0, step=35):
    return [sym(x + i * step, c, y=y + (40 if c in "omnscae" else 0), w=30, h=60 if c in "omnscae" else 100)
            for i, c in enumerate(letters)]


def test_plain_sum_reads_x_between_numbers_as_times():
    symbols = [sym(0, "1"), sym(50, "2"), sym(100, "plus", y=30, h=40), sym(150, "7"),
               sym(200, "x", y=40, h=50), sym(250, "4")]
    assert to_text(symbols, LABELS) == "12+7×4"


def test_function_word_beats_lookalike_digits():
    letters = [sym(0, "l", 0.55, "1"), sym(30, "o", 0.55, "0", y=40, h=60), sym(75, "g", 0.5, "9", y=40, h=90)]
    args = [sym(130, "lparen"), sym(170, "1"), sym(210, "0"), sym(250, "0"), sym(290, "rparen")]
    symbols = letters + args
    assert to_text(symbols, LABELS) == "log(100)"
    assert [s["read_as"] for s in letters] == ["l", "o", "g"]


def test_confident_digits_stay_digits():
    symbols = [sym(0, "1", 0.8, "l"), sym(50, "0", 0.8, "o"), sym(100, "9", 0.9, "g")]
    assert to_text(symbols, LABELS) == "109"


def test_lone_letter_is_read_as_its_lookalike_digit():
    symbols = [sym(0, "o", 0.6, "0"), sym(50, "plus", y=30, h=40), sym(100, "5")]
    assert to_text(symbols, LABELS) == "0+5"


def test_arrow_outside_a_limit_is_read_as_minus():
    symbols = [sym(0, "3"), sym(50, "arrow", 0.6, "minus", y=45, w=40, h=12), sym(100, "6")]
    assert to_text(symbols, LABELS) == "3-6"


def test_bracket_without_partner_is_read_again():
    symbols = [sym(0, "lparen", 0.6, "1"), sym(50, "2"), sym(100, "plus", y=30, h=40), sym(150, "3")]
    assert to_text(symbols, LABELS) == "12+3"


def test_power():
    assert to_text([sym(0, "2"), sym(45, "3", y=-20, w=20, h=40)], LABELS) == "2^3"


def test_fraction():
    symbols = [sym(0, "minus", y=95, w=120, h=10), sym(40, "1", y=20, w=20, h=60), sym(40, "2", y=130, w=30, h=60)]
    assert to_text(symbols, LABELS) == "(1)/(2)"


def test_limit_with_x_to_a_underneath():
    lim = [sym(0, "l", w=20), sym(30, "i", y=20, w=15, h=80), sym(55, "m", y=40, w=50, h=60)]
    under = [sym(10, "x", y=120, w=25, h=30), sym(40, "arrow", y=125, w=40, h=15), sym(85, "0", y=115, w=20, h=35)]
    body = [sym(160, "x", y=40, h=60), sym(205, "2", y=10, w=20, h=30)]
    assert to_text(lim + under + body, LABELS) == "lim_(x→0) x^2"


def test_a_dash_under_lim_is_read_as_the_arrow():
    lim = [sym(0, "l", w=20), sym(30, "i", y=20, w=15, h=80), sym(55, "m", y=40, w=50, h=60)]
    under = [sym(10, "x", y=120, w=25, h=30), sym(40, "minus", 0.7, "arrow", y=125, w=40, h=15), sym(85, "3", y=115, w=20, h=35)]
    body = [sym(160, "x", y=40, h=60), sym(205, "2", y=10, w=20, h=30)]
    assert to_text(lim + under + body, LABELS) == "lim_(x→3) x^2"


def test_a_small_arrow_under_lim_is_never_a_power():
    lim = [sym(0, "l", w=20), sym(30, "i", y=20, w=15, h=80), sym(55, "m", y=40, w=50, h=60)]
    under = [sym(5, "x", y=130, w=30, h=35), sym(45, "arrow", y=127, w=40, h=19), sym(95, "inf", y=127, w=44, h=21)]
    body = [sym(160, "x", y=40, h=60)]
    assert to_text(lim + under + body, LABELS) == "lim_(x→∞) x"


def test_log_base_written_low():
    symbols = word(0, "log") + [sym(110, "2", y=90, w=20, h=35), sym(140, "lparen"), sym(180, "8"), sym(220, "rparen")]
    assert to_text(symbols, LABELS) == "log_2(8)"


def test_equation_keeps_x_as_the_unknown():
    symbols = [sym(0, "2"), sym(50, "x", y=40, h=60), sym(110, "plus", y=30, h=40), sym(160, "5"),
               shape(220, "=", y=40, w=40, h=25), sym(280, "7")]
    assert to_text(symbols, LABELS) == "2x+5=7"


def test_two_lines_of_writing_are_two_equations():
    first = [sym(0, "x", y=0, h=60), sym(50, "plus", y=10, h=40), sym(100, "y", y=0, h=60), shape(150, "=", y=15, w=40, h=25), sym(200, "3", y=0, h=60)]
    second = [sym(0, "x", y=130, h=60), sym(50, "minus", y=155, w=40, h=10), sym(100, "y", y=130, h=60), shape(150, "=", y=145, w=40, h=25), sym(200, "1", y=130, h=60)]
    assert to_text(first + second, LABELS) == "x+y=3; x-y=1"


def test_fraction_is_not_split_into_two_lines():
    symbols = [sym(0, "minus", y=95, w=120, h=10), sym(40, "1", y=20, w=20, h=60), sym(40, "2", y=130, w=30, h=60)]
    assert to_text(symbols, LABELS) == "(1)/(2)"


def test_letter_clearly_read_stands_alone_but_a_doubtful_one_does_not():
    assert to_text([sym(0, "2"), sym(50, "y", y=20, h=70), shape(110, "=", y=30, w=40, h=25), sym(170, "6")], LABELS) == "2y=6"
    assert to_text([sym(0, "z", 0.5, "2"), sym(50, "plus", y=30, h=40), sym(100, "3")], LABELS) == "2+3"


def test_subscript_after_a_letter():
    symbols = [sym(0, "x", y=20, h=60), sym(45, "1", y=70, w=12, h=30), sym(80, "plus", y=30, h=40), sym(130, "x", y=20, h=60), sym(175, "2", y=70, w=20, h=30),
               shape(210, "=", y=30, w=40, h=25), sym(270, "3", y=0, h=80)]
    assert to_text(symbols, LABELS) == "x_1+x_2=3"


def test_matrix_between_tall_brackets():
    left, right = sym(0, "lbracket", y=0, w=20, h=200), sym(200, "rbracket", y=0, w=20, h=200)
    cells = [sym(40, "1", y=20, w=20, h=60), sym(130, "2", y=20, w=30, h=60), sym(40, "3", y=120, w=30, h=60), sym(130, "4", y=120, w=30, h=60)]
    word = [sym(-180, "r", y=70, w=30, h=50), sym(-140, "a", y=70, w=30, h=50), sym(-100, "n", y=70, w=30, h=50), sym(-60, "k", y=70, w=30, h=50)]
    assert to_text(word + [left, right] + cells, LABELS) == "rank[[1,2],[3,4]]"


def test_matrix_with_negative_entries():
    left, right = sym(0, "lparen", y=0, w=20, h=200), sym(240, "rparen", y=0, w=20, h=200)
    cells = [sym(40, "minus", y=45, w=25, h=8), sym(72, "1", y=20, w=20, h=60), sym(170, "3", y=20, w=30, h=60),
             sym(40, "minus", y=145, w=25, h=8), sym(72, "2", y=120, w=30, h=60), sym(170, "4", y=120, w=30, h=60)]
    assert to_text([left, right] + cells, LABELS) == "[[-1,3],[-2,4]]"


def test_power_measured_against_its_base_when_the_line_is_taller_than_it():
    symbols = [sym(0, "4", y=0, h=100), sym(50, "x", y=40, h=60), sym(95, "3", y=10, w=20, h=45), sym(130, "plus", y=50, h=40), sym(180, "y", y=40, h=90)]
    assert to_text(symbols, LABELS) == "4x^3+y"


def test_plus_and_minus_at_mid_height_are_not_a_power_of_a_small_letter():
    symbols = [sym(0, "2"), sym(50, "x", y=40, h=60), sym(110, "minus", y=45, h=8), sym(160, "5")]
    assert to_text(symbols, LABELS) == "2x-5"


def test_the_x_of_dx_is_not_a_subscript_of_the_tall_d():
    symbols = [sym(0, "x", y=40, h=60), sym(50, "d", y=0, h=100), sym(95, "x", y=40, h=60)]
    assert to_text(symbols, LABELS) == "xdx"


def test_a_long_1_low_after_a_letter_is_a_subscript_and_a_short_mark_a_comma():
    stroke = sym(45, "1", y=60, w=8, h=38)
    stroke["small_stroke"] = True
    assert to_text([sym(0, "x", y=20, h=60), stroke], LABELS) == "x_1"
    comma = sym(45, "1", y=75, w=6, h=20)
    comma["small_stroke"] = True
    assert to_text([sym(0, "x", y=20, h=60), comma, sym(70, "y", y=20, h=60)], LABELS) == "x,y"


def test_brackets_round_a_line_with_a_power_are_not_a_matrix():
    first = [sym(0, "lparen", y=0, w=25, h=130), sym(35, "x", y=40, h=60), sym(85, "plus", y=45, h=40), sym(135, "2", y=40, h=60),
             sym(185, "rparen", y=15, w=25, h=90), sym(215, "3", y=-30, w=20, h=40)]  # that ) is too short to count as tall
    second = [sym(245, "lparen", y=0, w=25, h=130), sym(280, "x", y=40, h=60), sym(330, "plus", y=45, h=40), sym(380, "5", y=40, h=60),
              sym(430, "rparen", y=0, w=25, h=130)]
    assert to_text(first + second, LABELS) == "(x+2)^3(x+5)"


def test_brackets_round_subscripts_are_not_a_matrix():
    symbols = [sym(0, "lbracket", y=0, w=20, h=200), sym(40, "1", y=20, w=20, h=60), sym(40, "3", y=120, w=30, h=60), sym(130, "4", y=120, w=30, h=60),
               sym(220, "rbracket", y=0, w=20, h=200), sym(260, "5", y=60, w=30, h=60)]
    assert not to_text(symbols, LABELS).startswith("[[")


def test_integral_with_limits():
    sign = sym(0, "integral", y=0, w=30, h=200)
    limits = [sym(35, "1", y=0, w=14, h=40), sym(35, "0", y=165, w=25, h=40)]
    body = [sym(90, "x", y=70, h=60), sym(140, "2", y=45, w=20, h=30), sym(180, "6", 0.5, "d", y=60, w=35, h=70), sym(225, "x", y=70, h=60)]
    assert to_text([sign] + limits + body, LABELS) == "∫_(0)^(1) x^2dx"


def test_derivative_fraction():
    bar = sym(0, "minus", y=95, w=120, h=10)
    top = [sym(20, "d", y=20, w=30, h=60), sym(60, "y", y=30, w=30, h=60)]
    bottom = [sym(20, "d", y=120, w=30, h=60), sym(60, "x", y=130, w=30, h=50)]
    assert to_text([bar] + top + bottom, LABELS) == "(dy)/(dx)"


def test_prime_and_comma_by_position():
    y = sym(0, "y", y=20, w=40, h=70)
    prime = {**sym(45, "1", y=10, w=8, h=25), "small_stroke": True}
    equals = shape(70, "=", y=35, w=40, h=25)
    assert to_text([y, prime, equals, sym(130, "0", y=20, h=70)], LABELS) == "y'=0"
    comma = {**sym(45, "1", y=75, w=8, h=25), "small_stroke": True}
    assert to_text([sym(0, "4", y=0, h=90), comma, sym(70, "5", y=0, h=90)], LABELS) == "4,5"


def test_operator_d_in_a_differential_equation():
    text_parts = [sym(0, "lparen", h=100), sym(40, "d", y=10, w=30, h=80), sym(75, "plus", y=30, w=30, h=40), sym(115, "6", y=10, w=30, h=80), sym(155, "rparen", h=100),
                  sym(195, "y", y=20, w=30, h=70), shape(235, "=", y=35, w=40, h=25), sym(285, "0", y=10, w=30, h=80)]
    assert to_text(text_parts, LABELS) == "(D+6)y=0"


def test_dot_by_height():
    symbols = [sym(0, "1"), shape(45, "dot", y=88, w=10, h=10), sym(60, "5"),
               shape(110, "dot", y=45, w=10, h=10), sym(130, "4")]
    assert to_text(symbols, LABELS) == "1.5·4"


def test_exclamation_mark_needs_the_right_shape():
    for exclaim, read in ((False, "51"), (True, "5!")):
        stem = sym(50, "fact", 0.99, "1")
        stem["exclaim"] = exclaim
        assert to_text([sym(0, "5"), stem], LABELS) == read


def test_log_reads_as_a_word_when_every_letter_is_a_close_lookalike():
    # what the CNN really gave for a clean handwritten log: each letter second to its lookalike digit
    letters = [sym(0, "1", 0.42, "l"), sym(30, "0", 0.88, "o", y=40, h=60), sym(75, "9", 0.81, "g", y=40, h=90)]
    assert to_text(letters + [sym(130, "lparen"), sym(170, "8"), sym(210, "rparen")], LABELS) == "log(8)"


def test_a_stem_with_a_dot_on_top_is_an_i_even_when_the_cnn_says_exclamation_mark():
    letters = [sym(0, "l", 0.5, "1"), sym(30, "fact", 0.99, y=20, h=80), sym(60, "m", y=40, h=60)]
    letters[1].update(exclaim=False, dotted=True)
    assert to_text(letters + [sym(150, "x", y=30, h=60)], LABELS).startswith("lim")


def test_small_digit_after_log_is_its_base_not_a_comma():
    letters = [sym(0, "l", 0.55, "1"), sym(30, "o", 0.55, "0", y=40, h=60), sym(75, "g", 0.5, "9", y=40, h=90)]
    base = sym(130, "2", 0.99, y=80, w=30, h=40)
    base["small_stroke"] = True
    assert to_text(letters + [base, sym(170, "lparen"), sym(210, "8"), sym(250, "rparen")], LABELS) == "log_2(8)"


def test_a_last_digit_sticking_out_past_the_root_bar_stays_under_it():
    symbols = [sym(0, "sqrt", w=300, h=200), sym(100, "4", y=60, w=60), sym(270, "9", y=60, w=80)]
    assert to_text(symbols, LABELS) == "√(49)"


def test_a_symbol_past_the_root_bar_stays_outside():
    symbols = [sym(0, "sqrt", w=200, h=200), sym(100, "4", y=60, w=60), sym(230, "plus", y=80, w=60, h=60), sym(310, "3", y=60, w=60)]
    assert to_text(symbols, LABELS) == "√(4)+3"
