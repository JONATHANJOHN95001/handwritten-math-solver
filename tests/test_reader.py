import cv2
import numpy as np

from reader import ink_mask, normalize, segment, stroke_masks


def test_ink_mask_handles_white_paper_and_transparent_canvas():
    paper = np.full((50, 50, 4), 255, np.uint8)
    clear = np.zeros((50, 50, 4), np.uint8)
    for rgba in (paper, clear):
        cv2.line(rgba, (10, 25), (40, 25), (0, 0, 0, 255), 5)
        mask = ink_mask(rgba)
        assert mask[25, 25] == 255 and mask[5, 5] == 0


def test_normalize_fits_ink_in_a_centred_20px_box():
    mask = np.zeros((200, 300), np.uint8)
    cv2.line(mask, (100, 20), (100, 180), 255, 12)  # a tall '1'
    out = normalize(mask)
    ys, xs = np.nonzero(out > 0.1)
    assert out.shape == (28, 28) and out.max() <= 1
    assert ys.max() - ys.min() + 1 <= 20
    assert abs((ys.min() + ys.max()) / 2 - 13.5) <= 1 and abs((xs.min() + xs.max()) / 2 - 13.5) <= 1


def test_normalize_ignores_pen_width():
    thin, thick = np.zeros((200, 200), np.uint8), np.zeros((200, 200), np.uint8)
    for mask, width in ((thin, 4), (thick, 24)):
        cv2.line(mask, (60, 170), (140, 30), 255, width)  # a '/'
    assert np.abs(normalize(thin) - normalize(thick)).mean() < 0.03


def test_normalize_survives_a_round_blob_that_thinning_wears_away():
    for radius in (3, 8, 20, 40):
        mask = np.zeros((120, 120), np.uint8)
        cv2.circle(mask, (60, 60), radius, 255, -1)
        assert normalize(mask).max() > 0


def test_normalize_survives_a_long_thin_line_that_fades_when_shrunk():
    mask = np.zeros((420, 2000), np.uint8)
    cv2.line(mask, (10, 200), (1990, 200), 255, 3)
    assert normalize(mask).max() > 0


def test_real_equals_signs_lean_and_differ_in_width():
    mask, layers = _pen([[(100, 100), (160, 200)], [(250, 100), (250, 200)],  # tall strokes beside it, as on any real line
                         [(300, 140), (330, 150)], [(295, 170), (345, 176)]], width=5)
    symbols = segment(mask, layers)
    assert [s["kind"] for s in symbols] == [None, None, "="]


def test_something_touching_a_fraction_bar_is_not_part_of_it():
    mask, layers = _pen([[(100, 100), (120, 170)], [(60, 175), (180, 175)], [(100, 180), (120, 250)]], width=5)  # top and bottom both touch the bar
    assert len(segment(mask, layers)) == 3


def test_the_top_bar_of_a_5_has_one_side_only_and_stays_on_it():
    mask, layers = _pen([[(100, 100), (160, 100)], [(105, 112), (102, 150), (140, 150), (150, 190), (100, 200)]], width=5)
    assert len(segment(mask, layers)) == 1


def test_a_digit_touching_the_bar_of_a_root_stays_a_digit():
    root = [(40, 200), (60, 180), (90, 280), (120, 100), (330, 100)]
    digit = [(160, 104), (200, 104), (200, 180), (160, 180), (160, 240), (200, 240)]
    mask, layers = _pen([root, digit], width=5)
    assert len(segment(mask, layers)) == 2


def test_a_limit_touching_a_tall_integral_stays_apart():
    sign = [(120, 60), (100, 70), (90, 150), (80, 300), (60, 310)]
    limit = [(92, 280), (120, 280), (120, 320), (92, 320), (92, 280)]
    mask, layers = _pen([sign, limit], width=5)
    assert len(segment(mask, layers)) == 2


def test_a_tap_with_a_thin_pen_is_a_dot_but_stray_pixels_are_not():
    mask, layers = _pen([[(100, 100), (100, 180)], [(100, 198), (101, 198)]], width=3)
    assert len(segment(mask, layers)) == 1  # a ! keeps its dot
    assert len(segment(mask)) == 2 or segment(mask)[0]["box"][3] < 100  # from pixels alone so small a mark may be a speck


def test_a_dot_a_little_to_one_side_of_a_stem_joins_it_but_a_decimal_point_does_not():
    mask, layers = _pen([[(100, 100), (100, 180)], [(110, 198), (111, 198)]], width=5)
    assert len(segment(mask, layers)) == 1  # a ! with its dot off to the right
    mask, layers = _pen([[(100, 100), (100, 180)], [(114, 176), (115, 176)], [(140, 100), (140, 180)]], width=5)
    assert len(segment(mask, layers)) == 3  # 1.1


def test_a_curly_x_with_a_gap_is_one_symbol_but_tall_brackets_are_two():
    left = [(100, 100), (125, 125), (130, 150), (125, 175), (100, 200)]
    right = [(170, 100), (145, 125), (140, 150), (145, 175), (170, 200)]
    mask, layers = _pen([left, right], width=5)
    assert len(segment(mask, layers)) == 1
    tall_left = [(100, 40), (135, 90), (145, 150), (135, 210), (100, 260)]
    tall_right = [(190, 40), (160, 90), (155, 150), (160, 210), (190, 260)]
    digit = [(30, 110), (30, 190)]
    other = [(250, 110), (250, 190)]
    third = [(290, 110), (290, 190)]
    mask, layers = _pen([digit, tall_left, tall_right, other, third], width=5)
    assert len(segment(mask, layers)) == 5


def test_a_stem_with_an_arm_against_its_side_is_one_symbol():
    mask, layers = _pen([[(100, 100), (100, 220)], [(140, 130), (100, 170), (140, 215)]], width=5)  # a k
    assert len(segment(mask, layers)) == 1


def _poly(mask, points):
    cv2.polylines(mask, [np.array(points, np.int32)], False, 255, 12)


def test_segment_reads_shapes_and_glues_multi_stroke_symbols():
    mask = np.zeros((200, 900), np.uint8)
    cv2.line(mask, (40, 30), (40, 170), 255, 12)                                       # '1'
    cv2.line(mask, (110, 40), (165, 40), 255, 12)                                      # '5': top bar with a gap
    _poly(mask, [(112, 58), (112, 95), (140, 88), (160, 105), (158, 135), (135, 148), (110, 140)])
    cv2.line(mask, (260, 80), (340, 80), 255, 12)                                      # '='
    cv2.line(mask, (260, 120), (340, 120), 255, 12)
    cv2.line(mask, (420, 100), (500, 100), 255, 12)                                    # '÷'
    cv2.circle(mask, (460, 60), 8, 255, -1)
    cv2.circle(mask, (460, 140), 8, 255, -1)
    cv2.line(mask, (570, 80), (570, 150), 255, 12)                                     # 'i': stem and dot
    cv2.circle(mask, (570, 50), 8, 255, -1)
    cv2.circle(mask, (640, 100), 9, 255, -1)                                           # lone dot
    cv2.circle(mask, (720, 100), 1, 255, -1)                                           # stray speck, ignored

    symbols = segment(mask)

    assert [s["kind"] for s in symbols] == [None, None, "=", "÷", None, "dot"]
    lefts = [s["box"][0] for s in symbols]
    assert lefts == sorted(lefts)
    assert symbols[1]["box"][3] > 100  # the '5' kept its top bar
    assert symbols[4]["box"][1] < 50   # the 'i' kept its dot
    assert symbols[0]["image"].shape == (28, 28) and symbols[2]["image"] is None


def test_segment_keeps_stacked_parts_apart():
    mask = np.zeros((300, 400), np.uint8)
    cv2.line(mask, (50, 140), (200, 140), 255, 12)          # fraction bar
    cv2.line(mask, (125, 60), (125, 118), 255, 12)          # '1' just above it (10 px gap)
    _poly(mask, [(100, 170), (150, 170), (110, 240)])        # '7' below it
    _poly(mask, [(260, 40), (260, 100), (290, 60), (320, 100), (320, 40)])  # 'm'-ish letter
    _poly(mask, [(270, 125), (310, 160)])                    # stroke under it with a clear gap
    symbols = segment(mask)
    assert len(symbols) == 5


def test_dot_joins_its_own_letter_not_a_wide_bar_below():
    mask = np.zeros((300, 400), np.uint8)
    cv2.line(mask, (50, 200), (350, 200), 255, 10)          # fraction bar
    cv2.line(mask, (200, 120), (200, 160), 255, 10)         # 'i' stem above it
    cv2.circle(mask, (200, 100), 7, 255, -1)                # its dot
    bar, i = sorted(segment(mask), key=lambda s: -s["box"][2])
    assert bar["box"][1] > 190                              # the bar did not take the dot
    assert i["box"][1] < 100                                # the 'i' kept it


# 2x=3 as written on the board in a live test: the 2's tail runs into a curly x drawn as two arcs,
# and the 3 has a short extra stroke at its foot.
TWO = [(87, 75), (90, 55), (100, 42), (113, 45), (121, 60), (120, 85), (108, 112), (88, 135), (72, 148), (60, 155), (78, 143), (110, 143), (155, 143)]
ARC_RIGHT = [(160, 102), (175, 95), (190, 98), (198, 110), (195, 125), (183, 140), (170, 150), (155, 143)]
ARC_LEFT = [(230, 101), (210, 101), (195, 103), (197, 120), (205, 143), (222, 148), (240, 142)]
BAR_TOP = [(280, 107), (320, 102), (362, 101)]
BAR_BOTTOM = [(288, 140), (330, 131), (365, 131), (392, 141)]
THREE = [(433, 58), (460, 67), (485, 82), (492, 95), (480, 110), (450, 120), (408, 123), (440, 125), (465, 145), (483, 175), (480, 190), (465, 198), (452, 198)]
FOOT = [(425, 184), (450, 188)]


def _pen(strokes, shape=(420, 680), width=7):
    """One mask per pen stroke, and the whole drawing."""
    masks = []
    for points in strokes:
        mask = np.zeros(shape, np.uint8)
        cv2.polylines(mask, [np.array(points, np.int32)], False, 255, width)
        masks.append(mask)
    return np.max(masks, axis=0), masks


def test_pen_strokes_keep_touching_symbols_apart():
    drawing, strokes = _pen([TWO, ARC_RIGHT, ARC_LEFT, BAR_TOP, BAR_BOTTOM, THREE, FOOT])
    assert len(segment(drawing)) == 3  # from the pixels alone the 2 and x are one blob: that was the bug
    two, x, equals, three = segment(drawing, strokes)
    assert two["box"][0] + two["box"][2] < 165 and two["kind"] is None
    assert x["box"][0] >= 150 and x["box"][0] + x["box"][2] > 235  # both arcs of the x stayed together
    assert equals["kind"] == "="
    assert three["box"][1] + three["box"][3] > 195


def test_short_stroke_at_the_foot_of_a_3_is_not_a_minus():
    drawing, _ = _pen([THREE, FOOT])
    assert len(segment(drawing)) == 1


def test_minus_under_a_root_bar_and_a_3_beside_a_minus_stay_apart():
    root = [(20, 150), (40, 200), (70, 40), (400, 40)]
    minus = [(190, 120), (240, 120)]
    drawing, _ = _pen([root, minus])
    assert len(segment(drawing)) == 2
    drawing, _ = _pen([THREE, [(520, 130), (570, 130)]])
    assert len(segment(drawing)) == 2


def test_crossing_strokes_of_one_symbol_stay_one_symbol():
    plus = [[(100, 150), (200, 150)], [(150, 100), (150, 200)]]
    t = [[(300, 100), (300, 220)], [(270, 130), (340, 130)]]
    vee = [[(400, 100), (430, 200)], [(430, 200), (460, 100)]]
    drawing, strokes = _pen(plus + t + vee)
    assert len(segment(drawing, strokes)) == 3


def test_stroke_masks_reads_what_the_canvas_reports():
    # object shapes copied from a real st_canvas: pen paths are absolute until moved, a line is relative to its centre
    pen = {"type": "path", "left": 95.49, "top": 95.5, "stroke": "#111111", "strokeWidth": 7, "scaleX": 1, "scaleY": 1, "angle": 0,
           "path": [["M", 98.993, 99], ["Q", 99, 99, 150, 99], ["L", 200, 99]]}
    line = {"type": "line", "left": 349, "top": 139, "stroke": "#111111", "strokeWidth": 7, "scaleX": 1, "scaleY": 1, "angle": 0,
            "x1": -50, "x2": 50, "y1": -40, "y2": 40}
    eraser = {"type": "path", "left": 114, "top": 43.97, "stroke": "#FFFFFF", "strokeWidth": 30, "scaleX": 1, "scaleY": 1, "angle": 0,
              "path": [["M", 129, 58.97], ["L", 129, 209.03]]}
    board = np.zeros((420, 680), np.uint8)
    cv2.line(board, (99, 99), (200, 99), 255, 7)
    cv2.line(board, (299, 99), (399, 179), 255, 7)

    masks = stroke_masks([pen, line], board)
    assert len(masks) == 2 and masks[0][99, 150] == 255 and masks[0][139, 349] == 0 and masks[1][139, 349] == 255

    board[:, 114:145] = 0
    masks = stroke_masks([pen, line, eraser], board)
    assert len(masks) == 2 and masks[0][99, 129] == 0 and masks[0][99, 150] == 255  # the eraser cut the pen stroke

    moved = {**pen, "left": 195.49, "top": 115.5}  # the Move tool changes left and top, not the path
    board = np.zeros((420, 680), np.uint8)
    cv2.line(board, (199, 119), (300, 119), 255, 7)
    assert stroke_masks([moved], board)[0][119, 250] == 255


def test_stroke_masks_gives_up_when_it_cannot_match_the_board():
    pen = {"type": "path", "left": 95.49, "top": 95.5, "stroke": "#111111", "strokeWidth": 7, "scaleX": 1, "scaleY": 1, "angle": 0,
           "path": [["M", 98.993, 99], ["L", 200, 99]]}
    board = np.zeros((420, 680), np.uint8)
    cv2.line(board, (99, 99), (200, 99), 255, 7)
    assert stroke_masks([], board) is None
    assert stroke_masks([{**pen, "scaleX": 1.5}], board) is None             # stretched with the Move tool
    assert stroke_masks([{"type": "circle", "left": 1, "top": 1}], board) is None
    assert stroke_masks([pen], np.zeros((420, 680), np.uint8)) is None        # strokes that are not what the board shows


def test_a_real_wide_x_written_as_two_arcs_stays_one_symbol():
    """2x+5=3 traced from a real board. Its x is two arcs back to back that touch, and it is wider than it is tall."""
    two = [(75, 74), (90, 50), (110, 25), (120, 12), (130, 10), (140, 15), (152, 30), (150, 50), (140, 65), (125, 85), (115, 100), (105, 107), (85, 118), (120, 115), (140, 117), (160, 124), (183, 130)]
    arc_left = [(187, 70), (200, 68), (215, 63), (240, 62), (252, 68), (262, 78), (265, 88), (260, 95), (245, 103), (225, 112), (210, 120), (192, 122)]
    arc_right = [(302, 70), (292, 66), (283, 67), (270, 75), (256, 83), (250, 92), (255, 100), (280, 118), (300, 130), (320, 138), (332, 138), (338, 134)]
    plus_bar, plus_stem = [(338, 88), (360, 82), (380, 80), (415, 81), (435, 82)], [(368, 42), (372, 60), (372, 80), (375, 110), (378, 140)]
    five_top = [(468, 82), (468, 40), (462, 15), (490, 12), (520, 14), (555, 8)]
    five_body = [(470, 75), (480, 62), (495, 57), (510, 70), (530, 90), (535, 100), (520, 112), (490, 125), (450, 125), (440, 118)]
    equals_top, equals_low = [(586, 62), (595, 58), (625, 58)], [(590, 93), (605, 88), (625, 80), (635, 79)]
    three = [(702, 14), (712, -5), (735, -15), (755, -8), (765, 15), (765, 30), (745, 50), (720, 62), (700, 68), (680, 73), (665, 74), (720, 65), (750, 72), (770, 90), (775, 100), (760, 115), (735, 135), (700, 148), (682, 148), (678, 138)]
    strokes = [two, arc_left, arc_right, plus_bar, plus_stem, five_top, five_body, equals_top, equals_low, three]
    mask, masks = _pen([[(round(x * 0.803), round((y + 40) * 0.803)) for x, y in stroke] for stroke in strokes], width=6)
    symbols = sorted(segment(mask, masks), key=lambda s: s["box"][0])
    assert len(symbols) == 6                                # 2, x, +, 5, the equals sign and 3
    x_box = symbols[1]["box"]
    assert x_box[2] > x_box[3]                              # the x took both arcs, so it is wider than tall
    assert symbols[4]["kind"] == "="


def test_stroke_masks_draw_as_wide_as_the_canvas():
    pen = {"type": "path", "left": 95.5, "top": 45.5, "stroke": "#111111", "strokeWidth": 7, "scaleX": 1, "scaleY": 1, "angle": 0,
           "path": [["M", 99, 49], ["L", 99, 105]]}
    board = np.zeros((420, 680), np.uint8)
    board[46:109, 95:103] = 255  # how a real canvas showed this stroke: 8 px wide and 63 px tall
    mask = stroke_masks([pen], board)[0]
    _, _, width, height, _ = cv2.connectedComponentsWithStats(mask)[2][1]
    assert width <= 8 and height <= 63


def test_only_a_stem_with_a_dot_below_it_can_be_an_exclamation_mark():
    stem = [(100, 40), (100, 130)]
    one, = segment(_pen([stem])[0])
    assert not one["exclaim"]
    drawing, _ = _pen([stem])
    below = drawing.copy()
    cv2.circle(below, (100, 158), 7, 255, -1)
    assert segment(below)[0]["exclaim"]
    above = drawing.copy()
    cv2.circle(above, (100, 15), 7, 255, -1)
    assert not segment(above)[0]["exclaim"]
