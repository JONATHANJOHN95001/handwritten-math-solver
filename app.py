"""AI Handwritten Math Solver: write maths or type a question. A CNN reads each handwritten symbol, the layout reader puts
them together, the English reader understands typed questions, and the course methods work out the answer.

The look (theme in .streamlit/config.toml, the rest in CSS below): ink-blue on paper, white cards, amber kept for the answer.
"""
import html
import json
import pathlib
from urllib.parse import quote

import cv2
import numpy as np
import streamlit as st
import sympy as sp
from streamlit_drawable_canvas import st_canvas

from layout import to_text
from questions import answer_question
from reader import ink_mask, segment, stroke_masks

HERE = pathlib.Path(__file__).parent
SHOWN = {"plus": "+", "minus": "−", "x": "x", "slash": "÷", "dot": "·", "lparen": "(", "rparen": ")", "arrow": "→", "inf": "∞",
         "integral": "∫", "sqrt": "√", "pi": "π", "lbracket": "[", "rbracket": "]", "partial": "∂", "fact": "!",
         "lambda": "λ", "mu": "μ", "theta": "θ"}
LABEL_OF = {"+": "plus", "-": "minus", "×": "x", "÷": "slash", "(": "lparen", ")": "rparen", "→": "arrow", "∞": "inf",
            "fraction bar": "minus", "∫": "integral", "√": "sqrt", "π": "pi", "[": "lbracket", "]": "rbracket", "∂": "partial",
            "!": "fact", "λ": "lambda", "μ": "mu", "θ": "theta"}  # what the reader wrote -> CNN class
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
.block-container { max-width: 1180px; padding-top: 3.8rem; }
.app-header { display: flex; gap: 14px; align-items: center; padding-bottom: 14px;
              border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.app-icon { width: 44px; height: 44px; border-radius: 10px; background: var(--ink); flex: none;
            display: grid; place-items: center; box-shadow: 0 1px 3px rgba(26, 54, 93, .25); }
.app-header h1 { font-family: 'Noto Serif', serif; font-size: 2rem; font-weight: 600; color: var(--ink-deep);
                 margin: 0; padding: 0; line-height: 1.2; letter-spacing: -.015em; }
.app-header p { margin: 4px 0 0; color: var(--muted); font-size: 15px; }
.card-title { font-weight: 600; font-size: 14px; color: var(--ink-deep); padding-bottom: 6px; margin-bottom: 4px;
              border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 6px; }
.card-title small { color: var(--muted); font-weight: 400; }
.st-key-board-card, .st-key-found-card, .st-key-read-card, .st-key-type-card {
  background: #FFFFFF; box-shadow: 0 1px 3px rgba(26, 54, 93, .05), 0 1px 2px rgba(26, 54, 93, .03); }
.st-key-board-card [data-testid="stElementContainer"]:has(iframe) {  /* graph paper round the board */
  padding: 18px 0; border-radius: 10px; border: 1px solid var(--line);
  background-color: #FBFCFF; background-size: 20px 20px;
  background-image: linear-gradient(to right, rgba(100, 116, 139, .09) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(100, 116, 139, .09) 1px, transparent 1px); }
.st-key-board-card iframe { display: block; width: 682px !important; max-width: 100%; margin: 0 auto;
                            background: #FFFFFF; border-radius: 6px;
                            box-shadow: 0 0 0 1px rgba(100, 116, 139, .35), 0 4px 14px rgba(26, 54, 93, .08); }
.st-key-answer-card, .st-key-type-answer { background: #FFFBEB !important; border: 2px solid var(--amber) !important; }
.answer-title { font-weight: 700; font-size: 14px; color: #78350F; display: flex; align-items: center; gap: 6px;
                padding-bottom: 6px; border-bottom: 1px solid #FDE68A; }
.answer-value { font-family: 'Noto Serif', serif; font-size: 34px; font-weight: 600; color: #B45309;
                text-align: center; padding: 18px 0 14px; line-height: 1.25; overflow-wrap: anywhere; }
.answer-value.long { font-family: inherit; font-size: 15px; font-weight: 500; text-align: left; line-height: 1.5;
                     white-space: pre-wrap; padding: 10px 2px; color: #78350F; }
.answer-note { background: rgba(254, 243, 199, .7); border: 1px solid #FDE68A; border-radius: 8px; padding: 4px;
               text-align: center; font-size: 12px; font-weight: 500; color: #78350F; }
.read-as { font-size: 13px; color: var(--muted); padding: 6px 0; overflow-wrap: anywhere; }
.read-as b { color: var(--ink-deep); }
.tips-card, .model-card { border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.tips-card { background: #FFFFFF; }
.model-card { background: #EFF4FF; margin-top: 4px; }
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
.stat b { color: var(--ink-deep); font-weight: 700; }
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
.app-icon {{ background: var(--ink) {_icon(PENCIL_ICON, '#ffffff')} center / 24px no-repeat; }}
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


def read_drawing(mask, strokes, model, labels):
    """Segment the drawing, let the CNN score every symbol the shape rules did not decide, then read the layout."""
    symbols = segment(mask, strokes)
    to_classify = [s for s in symbols if s["kind"] is None]
    if to_classify:
        probs = model(np.stack([s["image"] for s in to_classify])[..., None], training=False).numpy()
        for s, p in zip(to_classify, probs):
            s["probs"] = p
    text = to_text(symbols, labels)
    for s in symbols:
        if s["kind"]:
            s.update(confidence=1.0, runner_up="shape rule")
            continue
        p = dict(zip(labels, s["probs"]))
        label = LABEL_OF.get(s.get("read_as"), s.get("read_as"))
        s["confidence"] = 1.0 if s.get("read_as") in ("[", "]", "'", ",") else (p.get(label, max(p.values())) + sum(p.get(k, 0) for k in LOOKALIKES.get(label, ())))
        others = sorted((k for k in p if k != label), key=p.get, reverse=True)
        s["runner_up"] = SHOWN.get(others[0], others[0])
    return symbols, text


def annotate(mask, symbols):
    """Black-on-white copy of the drawing with a numbered box round each symbol."""
    img = np.full(mask.shape + (3,), 255, np.uint8)
    img[mask > 0] = 0
    for i, s in enumerate(symbols, 1):
        x, y, w, h = s["box"]
        colour = (40, 160, 60) if s["confidence"] >= UNSURE else (230, 120, 0)
        cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h + 4), colour, 2)
        cv2.putText(img, str(i), (x, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


def card_title(text, note=""):
    st.html(f'<div class="card-title">{text}{f" <small>{note}</small>" if note else ""}</div>')


def show_answer(result):
    """The answer card's contents: short answers large, anything longer as a list of lines."""
    text = "\n".join(result.answer)
    style = "answer-value" if len(result.answer) == 1 and len(text) < 26 else "answer-value long"
    st.html('<div class="answer-title icon icon-check">Answer</div>'
            f'<div class="{style}">{html.escape(text)}</div>'
            f'<div class="answer-note">{html.escape(result.topic)}</div>')


def show_working(result):
    if result.steps:
        with st.expander("Working", icon=":material/format_list_numbered:"):
            st.text("\n".join(result.steps))


def try_answer(text):
    """(result, problem): a Result, or a kind of st message and its text."""
    try:
        return answer_question(text), None
    except ZeroDivisionError as err:
        return None, ("error", str(err))
    except (ValueError, TypeError, NotImplementedError, AttributeError, IndexError, KeyError) as err:
        return None, ("warning", f"I cannot solve this yet: {err}")


st.set_page_config(page_title="AI Handwritten Math Solver", page_icon="✏️", layout="wide")
st.html(f"<style>{CSS}</style>")
st.html(f'<div class="app-header"><div class="app-icon"></div><div>'
        '<h1>AI Handwritten Math Solver</h1><p>Write maths by hand or paste a question: sums, equations, matrices, calculus, '
        'statistics, probability, counting, recurrences and graphs.</p></div></div>')

with st.sidebar:
    tips = "".join(f"<li><b>{html.escape(head)}</b> {html.escape(body)}</li>" for head, body in TIPS)
    st.html(f'<div class="tips-card"><div class="side-title icon icon-note">How to write</div><ul>{tips}</ul></div>')
    metrics_file = HERE / "reports" / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        st.html(f'<div class="model-card"><div class="side-title icon icon-chip">Model</div>'
                f'<div class="stat">Symbol accuracy (test set)<b>{metrics["test_accuracy"]:.1%}</b></div>'
                f'<div class="stat">Whole-sum accuracy (test set)<b>{metrics["expression_accuracy"]:.1%}</b></div>'
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
            show_working(result)
        else:
            getattr(st, problem[0])(problem[1])

with write_tab:
    with st.container(border=True, key="board-card"):
        tool_col, width_col, ink_col = st.columns([5.6, 2, 1.5], vertical_alignment="bottom")
        tool = tool_col.segmented_control("Tool", list(TOOLS), default=PEN, key="tool") or PEN
        mode, start_width = TOOLS[tool]
        width = width_col.slider("Thickness", 3, 40, start_width, key=f"width-{tool}")  # each tool keeps its own
        ink = ink_col.selectbox("Ink", list(INKS), key="ink", disabled=tool in (ERASER, MOVE))
        colour = INKS[ink]
        if tool == ERASER:
            colour = "#FFFFFF"  # paints the board colour back over the ink
        elif tool == PENCIL:
            colour = "rgba({}, {}, {}, 0.7)".format(*(int(colour[i:i + 2], 16) for i in (1, 3, 5)))
        canvas = st_canvas(stroke_width=width, stroke_color=colour, background_color="#FFFFFF",
                           width=680, height=420, drawing_mode=mode, key="canvas")

    model, labels = load_model()  # after the board is on screen, so you can start writing while it loads
    mask = None if canvas.image_data is None else ink_mask(canvas.image_data.astype(np.uint8))
    if mask is None or not mask.any():
        st.info("Write some maths on the white board above.", icon=":material/draw:")
        st.stop()

    strokes = stroke_masks((canvas.json_data or {}).get("objects", []), mask)
    symbols, read = read_drawing(mask, strokes, model, labels)
    if not symbols:
        st.info("Draw a little bigger: those marks are too small to read.", icon=":material/zoom_in:")
        st.stop()

    found_col, read_col, answer_col = st.columns([1.15, 1.1, 0.9], gap="medium")
    with found_col, st.container(border=True, key="found-card"):
        card_title("What the AI found", f"{len(symbols)} symbol{'s' if len(symbols) != 1 else ''}")
        st.image(annotate(mask, symbols), width="stretch")
    with read_col, st.container(border=True, key="read-card"):
        card_title("The AI read this", "(fix anything wrong here)")
        text = st.text_input("The AI read this", value=read, key=f"read-{read}", label_visibility="collapsed")
        result, problem = try_answer(text)
        if result is not None and result.parsed is not None:
            st.caption("Understood as")
            st.latex(sp.latex(result.parsed))
    with answer_col, st.container(border=True, key="answer-card"):
        if result is not None:
            show_answer(result)
        else:
            st.html('<div class="answer-title icon icon-check">Answer</div>')
            getattr(st, problem[0])(problem[1])
    if result is not None:
        show_working(result)

    unsure = [str(i) for i, s in enumerate(symbols, 1) if s["confidence"] < UNSURE]
    if unsure:
        st.warning(f"Not sure about symbol {', '.join(unsure)}: rewrite it more clearly.", icon=":material/help:")

    with st.expander("How sure is the AI about each symbol?", icon=":material/analytics:"):
        st.dataframe(
            [{"#": i,
              "Read as": s["kind"] if s["kind"] in ("=", "÷") else SHOWN.get(s["kind"], s.get("read_as")),
              "Confidence": round(100 * min(s["confidence"], 1.0)),
              "Second guess": s["runner_up"]} for i, s in enumerate(symbols, 1)],
            hide_index=True, width="stretch",
            column_config={"Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
