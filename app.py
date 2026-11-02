"""AI Handwritten Math Solver: write maths or type a question. A CNN reads each handwritten symbol, the layout reader puts
them together, the writer picks the right reading, the English reader understands typed questions, and the course methods
work out the answer step by step.

The look (theme in .streamlit/config.toml, the rest in CSS below): ink-blue on paper, white cards, amber kept for the answer.
"""
import html
import json
import pathlib
from urllib.parse import quote

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from questions import answer_question
from reader import ink_mask, stroke_masks
from readings import LABEL_OF, other_readings, read_board, to_latex

HERE = pathlib.Path(__file__).parent
SHOWN = {"plus": "+", "minus": "−", "x": "x", "slash": "÷", "dot": "·", "lparen": "(", "rparen": ")", "arrow": "→", "inf": "∞",
         "integral": "∫", "sqrt": "√", "pi": "π", "lbracket": "[", "rbracket": "]", "partial": "∂", "fact": "!",
         "lambda": "λ", "mu": "μ", "theta": "θ"}
BOARD_WIDTH, BOARD_HEIGHT = 1400, 510  # the whole board and its icons fit one screen at 1536 x 695, a 1080p screen at 125%
FOUND_WIDTH, FOUND_HEIGHT = 700, 240    # what the AI found shows at least this much, so that short writing is not blown up
NONE_OF_THESE = "None of these: I will correct it"
UNSURE = 0.6  # below this confidence the app asks for the symbol to be rewritten
# Shapes that are the same stroke (1, l, i) or nearly (0/o, 5/s, 9/g): where the symbol stands decides
# which it is, so the shape's confidence counts them together.
LOOKALIKES = {"1": ("l", "i"), "l": ("1", "i"), "i": ("1", "l"), "0": ("o",), "o": ("0",),
              "5": ("s",), "s": ("5",), "9": ("g",), "g": ("9",), "2": ("z",), "z": ("2",)}
TOOLS = {  # tool: (canvas drawing mode, starting thickness)
    ":material/stylus: Pen": ("freedraw", 7),
    ":material/edit: Pencil": ("freedraw", 5),
    ":material/horizontal_rule: Line": ("line", 7),
    ":material/ink_eraser: Eraser": ("freedraw", 30),
    ":material/drag_pan: Move": ("transform", 7),
}
PEN, PENCIL, _, ERASER, MOVE = TOOLS
INKS = {"Black": "#111111", "Blue": "#1d4ed8", "Red": "#b91c1c", "Green": "#15803d"}  # all dark enough to read
TIPS = [
    ("Sums and powers:", "12 + 7 × 3, x², 2¹⁰. Between two numbers, × can be an x."),
    ("Fractions:", "draw a bar, the top above it, the bottom below it"),
    ("Logs and limits:", "log₂ 8 with the base low; lim with x→0 under it"),
    ("Equations:", "one line for one equation, a new line for each equation of a system"),
    ("Matrices:", "tall brackets, rows and columns well apart, and rank, det or eigen before them"),
    ("Integrals:", "∫ with its limits at top and bottom, ending in dx"),
    ("Derivatives:", "y' or dy/dx, and (D²+5D+6)y for the D form"),
    ("Dots:", "at mid-height a dot means ×; low on the line it is a decimal point"),
    ("Gaps:", "leave a small gap between symbols. Two touching? Use Move."),
    ("Whole questions:", "paste the wording from your paper in the Type tab"),
]
EXAMPLES = [
    "Find the rank of the matrix A=[[1,2,3],[2,4,6],[1,0,1]].",
    "Test the consistency and solve x+y+z=9; x-2y+3z=8; 2x+y-z=3.",
    "Solve (D^2+5D+6)y=e^(3x).",
    "Solve the integral ∫_0^(π/2) sin^4 x dx.",
    "In a bolt factory there are four machines A, B, C, D manufacturing respectively 20%, 15%, 25%, 40% of the total production. Out of these 5%, 4%, 3%, 2% are defective. If a bolt drawn at random was found defective what is the probability that it was manufactured by A or D?",
    "Solve the recurrence relation a(n+1) = 4a(n), given that a(0) = 3.",
    "prim a-b:4, a-c:2, b-c:1, b-d:5, c-d:8, d-e:2",
]
CSS = """
:root { --ink: #1A365D; --ink-deep: #002045; --amber: #D97706; --muted: #43474E; --line: rgba(100, 116, 139, .18); }
[data-testid="stHeader"] { display: none; }
.block-container { max-width: none; padding: 8px 16px 32px; }
.st-key-topbar { position: relative; z-index: 1; height: 52px !important; min-height: 52px; padding: 0 12px; background: #FFFFFF; flex-wrap: nowrap;
                 border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 1px 3px rgba(26, 54, 93, .05); }
.app-header { display: flex; gap: 10px; align-items: center; }
.app-icon { width: 32px; height: 32px; border-radius: 8px; background: var(--ink); flex: none; }
.app-header h1 { font-family: 'Noto Serif', serif; font-size: 18px; font-weight: 600; color: var(--ink-deep);
                 margin: 0; padding: 0; line-height: 1.2; white-space: nowrap; }
[data-testid="stTabs"] { margin-top: -68px; }  /* the tabs sit in the top bar */
[data-testid="stTabs"] [role="tablist"] { position: relative; z-index: 2; width: fit-content; height: 52px; min-height: 52px; flex: 0 0 52px; margin-left: 350px;
                                          align-items: center; gap: 4px; border: none; box-shadow: none; }
[data-testid="stTab"] { height: 36px; padding: 0 12px; border-radius: 8px; align-items: center; }
[data-testid="stTab"][aria-selected="true"] { background: #EFF4FF; }
[data-testid="stTabPanel"] { padding-top: 0; margin-top: 8px; }
[data-testid="stTabPanel"] > [data-testid="stVerticalBlock"] { gap: 8px; }
.st-key-toolbar, .st-key-board-card { padding: 6px 12px !important; }
.st-key-toolbar { background: #FFFFFF; }
.st-key-toolbar [data-testid="stSliderThumbValue"] { display: none; }
.tool-label { font-size: 14px; color: var(--muted); white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
.tool-label b { color: var(--ink-deep); font-weight: 600; }
.card-title { font-weight: 600; font-size: 15px; color: var(--ink-deep); padding-bottom: 8px; margin-bottom: 4px;
              border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 6px; text-wrap: balance; }
.card-title small { color: #0058BE; background: #EFF4FF; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
.st-key-board-card, .st-key-found-card, .st-key-read-card, .st-key-type-card {
  background: #FFFFFF; box-shadow: 0 1px 3px rgba(26, 54, 93, .05), 0 1px 2px rgba(26, 54, 93, .03); }
.st-key-board-card [data-testid="stElementContainer"]:has(iframe) {  /* graph paper round the board */
  padding: 8px 0; border-radius: 10px; border: 1px solid var(--line); overflow-x: auto;
  background-color: #FBFCFF; background-size: 20px 20px;
  background-image: linear-gradient(to right, rgba(100, 116, 139, .09) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(100, 116, 139, .09) 1px, transparent 1px); }
.st-key-board-card iframe { display: block; width: BOARD_FRAMEpx !important; margin: 0 auto;
                            background: #FFFFFF; border-radius: 6px;
                            box-shadow: 0 0 0 1px rgba(100, 116, 139, .35), 0 4px 14px rgba(26, 54, 93, .08); }
.st-key-answer-card, .st-key-type-answer { background: #FFFBEB !important; border: 2px solid var(--amber) !important; }
.answer-title { font-weight: 700; font-size: 14px; color: #78350F; display: flex; align-items: center; gap: 6px;
                padding-bottom: 6px; border-bottom: 1px solid #FDE68A; }
.answer-value { font-family: 'Noto Serif', serif; font-size: 34px; font-weight: 600; color: #B45309;
                text-align: center; padding: 18px 0 14px; line-height: 1.25; overflow-wrap: anywhere; }
.answer-value.long { font-family: inherit; font-size: 15px; font-weight: 500; text-align: left; line-height: 1.5;
                     white-space: pre-wrap; padding: 10px 2px; color: #78350F; }
.answer-note { text-align: center; font-size: 13px; font-weight: 500; color: #78350F; padding-bottom: 6px; }
.read-as { font-size: 13px; color: var(--muted); padding: 6px 0; overflow-wrap: anywhere; }
.detected { font-family: 'Noto Serif', serif; font-size: 20px; color: var(--ink-deep); padding: 4px 0 2px; overflow-wrap: anywhere; }
.detected b { font-size: 22px; font-weight: 700; margin-left: 4px; }
.st-key-read-card [role="radiogroup"] { width: 100%; gap: 6px; }
.st-key-read-card [role="radiogroup"] label { width: 100%; padding: 8px 12px; border: 1px solid transparent; border-radius: 8px; cursor: pointer; }
.st-key-read-card [role="radiogroup"] label:hover { background: #F7F9FF; }
.st-key-read-card [role="radiogroup"] label:has(input:checked) { background: #EFF4FF; border-color: var(--line); }
.st-key-read-card .katex { font-size: 1.2em; }
.st-key-write-steps, .st-key-type-steps { background: #FFFFFF; }
.step { display: flex; gap: 12px; padding: 10px 12px; margin-bottom: 8px; background: #F7F9FF; border: 1px solid var(--line); border-radius: 8px;
        font-size: 15px; line-height: 1.5; color: #0B1C30; }
.step:last-child { margin-bottom: 0; }
.step-no { flex: none; width: 24px; height: 24px; border-radius: 50%; background: var(--ink); color: #FFFFFF; font-size: 12px; font-weight: 700;
           display: grid; place-items: center; margin-top: 1px; font-variant-numeric: tabular-nums; }
.step-text { font-family: 'Noto Serif', serif; overflow-wrap: anywhere; min-width: 0; text-wrap: pretty; }
.step-text pre { font-family: Consolas, 'Courier New', monospace; font-size: 14px; margin: 4px 0 0; padding: 6px 10px; background: #F7F9FF;
                 border-radius: 6px; overflow-x: auto; }
.read-as b { color: var(--ink-deep); }
.model-card { background: #EFF4FF; border-radius: 12px; padding: 12px; margin-top: 8px; }
.side-title { font-family: 'Noto Serif', serif; font-size: 20px; font-weight: 500; color: var(--ink-deep);
              display: flex; align-items: center; gap: 8px; padding-bottom: 6px; margin-bottom: 8px;
              border-bottom: 1px solid var(--line); }
.tips-card ul { list-style: none; padding: 0; margin: 0; }
.tips-card li { font-size: 13px; line-height: 1.5; color: var(--muted); padding: 3px 0 3px 14px; position: relative; }
.tips-card li::before { content: "•"; color: #0058BE; font-weight: 700; position: absolute; left: 0; }
.tips-card b { color: #0B1C30; font-weight: 600; }
.model-card .side-title { font-family: inherit; font-size: 14px; font-weight: 600; border: none; margin: 0; }
.stat { background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-top: 8px;
        display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: var(--muted); }
.stat b { color: var(--ink-deep); font-weight: 700; font-variant-numeric: tabular-nums; }
.icon::before { content: ""; width: 20px; height: 20px; flex: none; background: center / contain no-repeat; }
"""


def _icon(path, colour):
    """CSS background for a 24x24 Material icon path (st.html strips inline SVG, CSS images get through)."""
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='{colour}'><path d='{path}'/></svg>"
    return f'url("data:image/svg+xml,{quote(svg)}")'


PENCIL_ICON = ("M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zm17.71-10.21a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41"
               " 0l-1.83 1.83 3.75 3.75 1.83-1.83z")
NOTE = "M3 10h11v2H3zm0-4h11v2H3zm0 8h7v2H3zm17.59-2.07-4.25 4.24-2.12-2.12-1.41 1.41L16.34 20 22 14.34z"
CHIP = ("M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7a2 2 0 0 0-2-2h-2V3h-2v2h-2V3H9v2H7a2 2 0 0 0-2 2v2H3v2h2v2H3v2h2"
        "v2a2 2 0 0 0 2 2h2v2h2v-2h2v2h2v-2h2a2 2 0 0 0 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z")
CHECK = "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
CSS += f"""
.app-icon {{ background: var(--ink) {_icon(PENCIL_ICON, '#ffffff')} center / 20px no-repeat; }}
.icon-note::before {{ background-image: {_icon(NOTE, '#1A365D')}; }}
.icon-chip::before {{ background-image: {_icon(CHIP, '#1A365D')}; }}
.icon-check::before {{ background-image: {_icon(CHECK, '#B45309')}; }}
"""


@st.cache_resource(show_spinner="Loading the AI model (about 30 seconds the first time)...")
def load_model():
    import tensorflow as tf  # imported here, not at the top: it takes up to 30 s and the page should show first

    model = tf.keras.models.load_model(HERE / "models" / "symbol_cnn.keras")
    labels = json.loads((HERE / "models" / "labels.json").read_text())
    return model, labels


def add_confidence(symbols, labels):
    """How sure the CNN is of each symbol as read, and its second guess."""
    for s in symbols:
        if s["kind"]:
            s.update(confidence=1.0, runner_up="shape rule")
            continue
        p = dict(zip(labels, s["probs"]))
        label = LABEL_OF.get(s.get("read_as"), s.get("read_as"))
        s["confidence"] = 1.0 if s.get("read_as") in ("[", "]", "'", ",") else (p.get(label, max(p.values())) + sum(p.get(k, 0) for k in LOOKALIKES.get(label, ())))
        others = sorted((k for k in p if k != label), key=p.get, reverse=True)
        s["runner_up"] = SHOWN.get(others[0], others[0])


def _around_writing(symbols, shape):
    """(top, bottom, left, right) of a view of the board centred on the writing."""
    x0, y0 = min(s["box"][0] for s in symbols), min(s["box"][1] for s in symbols)
    x1, y1 = max(s["box"][0] + s["box"][2] for s in symbols), max(s["box"][1] + s["box"][3] for s in symbols)
    width, height = max(x1 - x0 + 60, FOUND_WIDTH), max(y1 - y0 + 60, FOUND_HEIGHT)
    left, top = max((x0 + x1 - width) // 2, 0), max((y0 + y1 - height) // 2, 0)
    return top, min(top + height, shape[0]), left, min(left + width, shape[1])


def annotate(mask, symbols):
    """Black-on-white copy of the writing with a numbered box round each symbol."""
    img = np.full(mask.shape + (3,), 255, np.uint8)
    img[mask > 0] = 0
    for i, s in enumerate(symbols, 1):
        x, y, w, h = s["box"]
        colour = (40, 160, 60) if s["confidence"] >= UNSURE else (230, 120, 0)
        cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h + 4), colour, 2)
        cv2.putText(img, str(i), (x, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    top, bottom, left, right = _around_writing(symbols, mask.shape)
    return img[top:bottom, left:right]


def scroll_to(card):
    """Bring a card into view. The mark in the comment makes the browser run this again for each new request."""
    st.session_state.scrolls = st.session_state.get("scrolls", 0) + 1
    st.html(f"""<script>/* {st.session_state.scrolls} */
      setTimeout(() => document.querySelector('.st-key-{card}')?.scrollIntoView({{
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center'}}), 150);
    </script>""", unsafe_allow_javascript=True)


def card_title(text, note=""):
    st.html(f'<div class="card-title">{text}{f" <small>{note}</small>" if note else ""}</div>')


def show_answer(result):
    """The answer card's contents: short answers large, anything longer as a list of lines."""
    text = "\n".join(result.answer)
    style = "answer-value" if len(result.answer) == 1 and len(text) < 26 else "answer-value long"
    st.html('<div class="answer-title icon icon-check">Answer</div>'
            f'<div class="{style}">{html.escape(text)}</div>'
            f'<div class="answer-note">{html.escape(result.topic)}</div>')


def show_working(result, key):
    """Every step of the solution, numbered. A step with a matrix keeps the matrix in lined-up columns."""
    if not result.steps:
        return
    rows = []
    for number, step in enumerate(result.steps, 1):
        head, _, rest = step.partition("\n")
        block = f"<pre>{html.escape(rest)}</pre>" if rest else ""
        rows.append(f'<div class="step"><span class="step-no">{number}</span><div class="step-text">{html.escape(head)}{block}</div></div>')
    with st.container(border=True, key=key):
        card_title("Step by step")
        st.html("".join(rows))


def reading_label(text):
    """A reading as it shows in the list of choices: a formula if it is one."""
    if text == NONE_OF_THESE:
        return text
    formula = to_latex(text)
    return f"${formula}$" if formula else f"`{text}`"


def board_cursor(tool, width):
    """A ring the size of the pen with a dot at its centre, in amber with a white edge so that it shows on the board and on ink."""
    if tool == MOVE:
        return ""
    size = min(max(width, 24) + 12, 100)
    centre = size / 2
    ring = f"cx='{centre}' cy='{centre}' r='{centre - 3}' fill='none'"
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'><circle {ring} stroke='white' stroke-width='5'/>"
           f"<circle {ring} stroke='#D97706' stroke-width='3'/><circle cx='{centre}' cy='{centre}' r='3.5' fill='#D97706' stroke='white'/></svg>")
    return f'.upper-canvas {{ cursor: url("data:image/svg+xml,{quote(svg)}") {centre} {centre}, crosshair !important; }}'


def set_board_cursor(tool, width):
    """The board is in a frame of its own, and its drawing library keeps setting a thin cursor there, so the style goes inside the frame."""
    st.html(f"""<script>
      window.boardCursorRule = {json.dumps(board_cursor(tool, width))};
      window.boardCursorTimer = window.boardCursorTimer || setInterval(() => {{
        document.querySelectorAll('iframe[title*="drawable_canvas"]').forEach((frame) => {{
          const page = frame.contentDocument;
          if (!page || !page.head) return;
          let style = page.getElementById('board-cursor');
          if (!style) {{ style = page.createElement('style'); style.id = 'board-cursor'; page.head.appendChild(style); }}
          if (style.textContent !== window.boardCursorRule) style.textContent = window.boardCursorRule;
        }});
      }}, 300);
    </script>""", unsafe_allow_javascript=True)


def new_board():
    """A new key gives an empty canvas."""
    st.session_state.board_round += 1
    st.session_state.last_picked = None
    st.session_state.go_to_board = True


def try_answer(text):
    """(result, problem): a Result, or a kind of st message and its text."""
    try:
        return answer_question(text), None
    except ZeroDivisionError as err:
        return None, ("error", str(err))
    except (ValueError, TypeError, NotImplementedError, AttributeError, IndexError, KeyError) as err:
        return None, ("warning", f"I cannot solve this yet: {err}")


st.set_page_config(page_title="AI Handwritten Math Solver", page_icon="✏️", layout="wide")
st.html(f"<style>{CSS.replace('BOARD_FRAME', str(BOARD_WIDTH + 2))}</style>")
st.session_state.setdefault("board_round", 0)
with st.container(key="topbar", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    st.html('<div class="app-header"><div class="app-icon"></div><h1>AI Handwritten Math Solver</h1></div>', width="content")
    with st.popover("How to write", icon=":material/help:"):
        tips = "".join(f"<li><b>{html.escape(head)}</b> {html.escape(body)}</li>" for head, body in TIPS)
        st.html(f'<div class="tips-card"><ul>{tips}</ul></div>')
        metrics_file = HERE / "reports" / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            st.html(f'<div class="model-card"><div class="side-title icon icon-chip">Model</div>'
                    f'<div class="stat">Symbol accuracy (real writing)<b>{metrics["test_accuracy"]:.1%}</b></div>'
                    f'<div class="stat">Whole-sum accuracy (real writing)<b>{metrics["expression_accuracy"]:.1%}</b></div>'
                    '</div>')

write_tab, type_tab = st.tabs([":material/draw: Write on the board", ":material/keyboard: Type or paste a question"])

with type_tab:
    with st.container(border=True, key="type-card"):
        card_title("Your question", "(the wording of your paper, a formula or a command)")
        example = st.selectbox("Try an example", ["", *EXAMPLES], key="example", format_func=lambda e: e[:90] or "Examples")
        question = st.text_area("Your question", value=example, key=f"question-{example}", height=130, label_visibility="collapsed",
                                placeholder="Find the rank of the matrix A=[[1,2],[3,4]].")
    if question.strip():
        result, problem = try_answer(question)
        if result is not None:
            if result.read_as:
                st.html(f'<div class="read-as"><b>I understood this as:</b> {html.escape(result.read_as)}</div>')
            with st.container(border=True, key="type-answer"):
                show_answer(result)
            show_working(result, "type-steps")
        else:
            getattr(st, problem[0])(problem[1])

with write_tab:
    with st.container(border=True, key="toolbar"):
        tool_col, width_label, width_col, ink_label, ink_col, _, done_col = st.columns([3.3, 0.95, 1.5, 0.3, 1, 1.5, 1], vertical_alignment="center")
        tool = tool_col.segmented_control("Tool", list(TOOLS), default=PEN, key="tool", label_visibility="collapsed") or PEN
        mode, start_width = TOOLS[tool]
        width = width_col.slider("Thickness", 3, 40, start_width, key=f"width-{tool}", label_visibility="collapsed")  # each tool keeps its own
        width_label.html(f'<div class="tool-label">Thickness <b>{width}</b></div>')
        ink_label.html('<div class="tool-label">Ink</div>')
        ink = ink_col.selectbox("Ink", list(INKS), key="ink", disabled=tool in (ERASER, MOVE), label_visibility="collapsed")
        done = done_col.button("Done", type="primary", icon=":material/check:", key="done", width="stretch")
    colour = INKS[ink]
    if tool == ERASER:
        colour = "#FFFFFF"  # paints the board colour back over the ink
    elif tool == PENCIL:
        colour = "rgba({}, {}, {}, 0.7)".format(*(int(colour[i:i + 2], 16) for i in (1, 3, 5)))
    with st.container(border=True, key="board-card"):
        canvas = st_canvas(stroke_width=width, stroke_color=colour, background_color="#FFFFFF",
                           width=BOARD_WIDTH, height=BOARD_HEIGHT, drawing_mode=mode, key=f"canvas-{st.session_state.board_round}")
    set_board_cursor(tool, width)
    if st.session_state.pop("go_to_board", False):
        scroll_to("board-card")

    model, labels = load_model()  # after the board is on screen, so you can start writing while it loads
    mask = None if canvas.image_data is None else ink_mask(canvas.image_data.astype(np.uint8))
    if mask is None or not mask.any():
        st.info("Write some maths on the white board above.", icon=":material/draw:")
        if done:
            st.toast("Write something on the board first.", icon=":material/draw:")
        st.stop()

    strokes = stroke_masks((canvas.json_data or {}).get("objects", []), mask)
    symbols, read = read_board(mask, strokes, model, labels)
    if not symbols:
        st.info("Draw a little bigger: those marks are too small to read.", icon=":material/zoom_in:")
        st.stop()
    add_confidence(symbols, labels)
    readings = other_readings(symbols, read, labels, mask, strokes, model)

    found_col, read_col = st.columns([1, 1.15], gap="medium")
    with found_col, st.container(border=True, key="found-card"):
        card_title("What the AI found", f"{len(symbols)} symbol{'s' if len(symbols) != 1 else ''}")
        st.image(annotate(mask, symbols), width="stretch")
    with read_col, st.container(border=True, key="read-card"):
        card_title("This expression is detected")
        st.html(f'<div class="detected">The AI read: <b>{html.escape(read)}</b></div>')
        picked = st.radio("Select the correct one", [*readings, NONE_OF_THESE], index=None, format_func=reading_label,
                          key="choice-" + "|".join(readings))
        if picked == NONE_OF_THESE:
            typed = st.text_input("Correct it here, then press Enter", value=read, key=f"typed-{read}")
            picked = typed if typed != read else None

    if done:
        scroll_to("read-card")

    unsure = [str(i) for i, s in enumerate(symbols, 1) if s["confidence"] < UNSURE]
    if unsure:
        st.warning(f"Not sure about symbol {', '.join(unsure)}: rewrite it more clearly.", icon=":material/help:")

    if picked != st.session_state.get("last_picked"):
        st.session_state.last_picked = picked
        if picked is not None:
            scroll_to("answer-card")
    if picked is None:
        st.info("Select the correct one above and it is solved here, step by step.", icon=":material/touch_app:")
    else:
        result, problem = try_answer(picked)
        with st.container(border=True, key="answer-card"):
            if result is not None:
                show_answer(result)
            else:
                st.html('<div class="answer-title icon icon-check">Answer</div>')
                getattr(st, problem[0])(problem[1])
        if result is not None:
            show_working(result, "write-steps")
        st.button("Write another", icon=":material/restart_alt:", key="again", on_click=new_board)

    with st.expander("How sure is the AI about each symbol?", icon=":material/analytics:"):
        st.dataframe(
            [{"#": i,
              "Read as": s["kind"] if s["kind"] in ("=", "÷") else SHOWN.get(s["kind"], s.get("read_as")),
              "Confidence": round(100 * min(s["confidence"], 1.0)),
              "Second guess": s["runner_up"]} for i, s in enumerate(symbols, 1)],
            hide_index=True, width="stretch",
            column_config={"Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
