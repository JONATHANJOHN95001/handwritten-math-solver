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
