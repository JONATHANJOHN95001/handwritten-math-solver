# AI Handwritten Math Solver

Write maths by hand on a board or paste a question from a paper, and get the answer with every step of its working. A CNN trained for
this project reads each handwritten symbol, a layout reader works out how the symbols sit on the board and method modules
compute the answer with SymPy, SciPy and NetworkX. It runs offline.

## What it can do

**Write on the board:** sums, powers, fractions, `log`, `ln`, `sin`, `cos`, `tan`, limits with `x→a` under `lim`, equations
in any letter, a new line for each equation of a system, matrices in tall brackets with a word such as `rank` before them,
integrals with limits, `y'`, `dy/dx`, factorials and `nCr`.

When the drawing is complete, the app prompts `This expression is detected` and lists what it read with the closest other
readings. Nothing is solved until you select the correct one, or select `None of these` and correct the text. The answer then
appears below with every step, numbered. Every number in the steps is computed, not typed.

**Type or paste a question,** in the wording of a paper or as a formula:

- **Matrices:** rank, echelon form, determinant, inverse, eigenvalues and eigenvectors, diagonalisation, powers,
  consistency of linear systems and Gauss-Seidel.
- **Calculus:** differential equations, partial and total derivatives, Jacobians, extrema, Lagrange multipliers, grad, div,
  curl, scalar potentials and definite integrals.
- **Statistics and probability:** descriptive statistics, correlation, regression, Bayes, binomial, Poisson, normal and
  uniform distributions, sampling, confidence intervals and z, t and chi-square tests.
- **Discrete maths:** counting, pigeonhole, derangements, integer solutions, inclusion-exclusion, rook polynomials,
  generating functions, partitions and recurrences.
- **Graphs and trees, with the edges typed:** chromatic number and polynomial, planarity, isomorphism, spanning trees, Prim,
  Kruskal, Huffman codes, merge sort and tree leaves.

## How well it works

The CNN scores 90.04% per symbol on 53 classes, on 49,513 images of real handwriting kept out of training. `1`, `l` and `i` or
`0` and `o` are nearly the same shapes, so where a symbol stands decides which it is.

`check_handwriting.py` gives the same real test to any model, so the scores below can be compared. Before it was trained on
real handwriting the model's special symbols had only ever seen a script's drawings, and scored near 100% on the script's own
test images while failing on real writing:

| Real handwriting kept out of training | Before | Now |
| --- | ---: | ---: |
| All symbols, 49,513 images | 83.19% | 90.04% |
| Symbols in MathWriting's test expressions, 30,319 images | 81.91% | 91.55% |
| HASYv2 symbols, 2,344 images | 73.12% | 93.81% |
| ∂ | 25.2% | 91.9% |
| μ | 13.5% | 89.1% |
| √ | 48.9% | 95.0% |
| ∫ | 66.5% | 94.7% |

The hardest test is whole expressions as real people wrote them: the full MathWriting test split, each expression redrawn
stroke by stroke as the board does. 1,208 of its 7,644 expressions use only what the board can write. The score counts a read
only when the whole line is exactly right. The reader's rules were tuned on MathWriting's separate validation split, apart from
the rule against false matrices, which came from a first look at these failures:

| Real expressions, 1,208 | Read exactly | Wrong characters |
| --- | ---: | ---: |
| Old model and old reader | 101 (8.4%) | 50.3% |
| Old model, reader rules fixed on real writing | 148 (12.3%) | 36.8% |
| New model and new reader | 244 (20.2%) | 32.0% |

So one line in five comes out perfect and about one character in three is wrong. Short lines do far better (three in four of
the lines under five characters on the validation split) and long ones with subscripts and fractions far worse. 44 of the 53
classes improved. Four fell by more than a point: `z` from 92.9% to 86.8%, `c` from 94.3% to 91.8%, `o` from 45.5% to 42.6% and `!`
from 97.5% to 90.0% (only 40 test images).

Whole expressions built by pasting held-out symbols together, read back exactly (`check_reading.py`). These now use real
handwriting for every symbol, so some kinds score lower than they did on the script's tidy drawings:

| Kind of expression | Read exactly |
| --- | ---: |
| Plain sums | 67.7% |
| Equations | 76.5% |
| Powers | 88.0% |
| Functions (log, ln, sin, cos, tan) | 67.0% |
| Logs with a base | 65.5% |
| Fractions | 85.0% |
| Limits | 55.0% |
| Equations in other letters | 58.0% |
| Two lines of equations | 47.5% |
| Matrices with a word | 31.0% |
| Integrals with limits | 50.0% |
| dy/dx | 46.0% |
| Factorials | 65.0% |
| Combinations (nCr) | 83.0% |

Writing wider or narrower than usual (`check_wide_writing.py`): 13,826 real test symbols from MathWriting, redrawn stretched sideways.
Retraining with half of MathWriting's training symbols drawn 0.8 to 1.8 times as wide helped very wide writing, but a plain stroke
stretched sideways only leans more: stretched `1`s taught the CNN to call real slashes `1`, slash recall fell from 76.0% to 67.3% and
plain sums fell from 67.7% to 58.7%. Whole real expressions did not improve, so that model was not kept. `--vary-width` in
`prepare_handwriting.py` now leaves `1`, `l`, `i` and `/` as written, but that version has not been trained to the end.

| Width of the writing | Model in use | All symbols widened |
| --- | ---: | ---: |
| As written | 90.63% | 90.47% |
| 1.25 times as wide | 90.55% | 90.92% |
| 1.5 times as wide | 89.06% | 90.45% |
| 1.8 times as wide | 86.25% | 88.77% |
| 0.75 times as wide | 88.09% | 87.60% |

Real questions from three university maths courses, in the wording of the papers (`check_courses.py`):

| Set | Tested | Right | Wrong | Refused |
| --- | ---: | ---: | ---: | ---: |
| Assignments and exams the methods were taught on | 49 | 47 | 0 | 2 |
| Previous-year papers kept unread until the methods were finished | 95 | 32 | 6 | 57 |

The second row is the first-try score and it is left as it came out. Most refusals are wordings the English reader does not
know yet, so the same question often works once it is written as a formula or a command.

## How it works

1. **Strokes:** the board's own record of each pen stroke is redrawn stroke by stroke, with erased parts cut out
   (`reader.stroke_masks`). If that record cannot be used, the reader falls back to the pixels of the drawing.
2. **Segmentation:** strokes become symbols (`reader.segment`). Strokes that cross stay one symbol, while a 2 whose tail runs
   into the next x stays two. Shape rules decide `=`, `÷` and dots.
3. **Recognition:** every other symbol is thinned to a skeleton, redrawn at one pen width, scaled into a 20×20 box inside
   28×28 (`reader.normalize`) and scored by the CNN. `!` also needs its shape, a stem with a dot below it.
4. **Layout:** `layout.to_text` finds fraction bars, function words, limits, integrals, matrices, powers, subscripts and
   separate lines of writing, then writes text such as `lim_(x→0) (sin(x))/(x)`.
5. **Prompt:** `readings.py` offers the best reading, the reading from grouping the strokes by their pixels and the closest
   second guesses of the symbols the CNN was least sure about. Each must parse before it is offered.
6. **English:** `english.py` turns a typed word problem into the same command language and the app shows what it understood.
7. **Solving:** `questions.py` sends each command to its method module. `solver.py` parses formulas with its own
   recursive-descent parser (no `eval`). A solve that runs past 60 seconds is stopped with a clear message.
8. **Working:** `working.py` writes the steps of sums, equations, limits, derivatives and integrals, and each method module
   writes its own. A step that cannot be spelled out never stops the answer.

## Data

All real handwriting except the last row, which only joins the training set. Each source keeps its own test split, or holds one
tenth back if it has none, and at most 6,000 images of a class come from MathWriting+.

| Source | What it adds | Training images | Test images | Licence |
| --- | --- | ---: | ---: | --- |
| [MathWriting+](https://zenodo.org/records/15091575), J. Seitz's symbol marks on [Google's MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) | Pen strokes of real maths: every symbol, written inside real expressions | 241,091 | 30,319 | CC BY-NC-SA 4.0 |
| [EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset), NIST | Digits and letters, capitals apart from small letters | 147,411 | 14,300 | NIST |
| [HASYv2](https://zenodo.org/records/259444), M. Thoma | ∫ √ π ∂ ∞ λ μ θ → [ ] and more | 21,093 | 2,344 | ODbL |
| [BHMSDS](https://github.com/wblachowski/bhmsds) | Digits and operators | 22,950 | 2,550 | MIT |
| Drawn by `train.py` | Plain textbook forms of the special symbols, curly x and / | 8,000 | none | |

MathWriting's licence is non-commercial, so a model trained on it is for non-commercial use too.

## Run it

With Python 3.11:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

`Run Math Solver.bat` starts it with a double-click and opens the browser, using a virtual environment at
`%USERPROFILE%\.venvs\handwritten-math-solver`.

## Retrain

Download the sets above, then turn them into one cache of 28×28 images (about 20 minutes) and train on it (about 45 minutes on a
CPU). Only the first 260 MB of `MathWriting+.zip` is needed, which `fetch_mathwriting_plus.py` fetches along with the zip's file list.

```bash
git clone https://github.com/wblachowski/bhmsds ../bhmsds
pip install scikit-learn matplotlib
python fetch_mathwriting_plus.py --out plus
python prepare_handwriting.py --bhmsds ../bhmsds/symbols --emnist <folder with emnist-byclass-*.gz> --hasy HASYv2.tar.bz2 --plus plus/MathWriting+.start.bin --plus-files plus/directory.json --mathwriting mathwriting-2024.tgz --out cache
python train.py --data cache
python check_reading.py --data cache
python check_handwriting.py --data cache --model new=models/symbol_cnn.keras
python check_wide_writing.py --plus plus/MathWriting+.start.bin --plus-files plus/directory.json --model new=models/symbol_cnn.keras
```

Training writes `models/symbol_cnn.keras`, `models/labels.json` and the figures and metrics in `reports/`.

## Tests

```bash
pip install pytest
python -m pytest
```

## Limits

- Most real lines of writing still have a mistake in them, which is why the prompt asks before solving. Keep symbols apart and
  write subscripts clearly below the line: those cause many of the misreadings.
- `f`, `g`, `h`, `i`, `j`, `l`, `o`, `q`, `r`, `s`, `u`, `v` and `w` are kept for function words: alone on a line they read as the nearest
  digit or symbol, because that is the far more common writing. The real-expression test leaves out lines that use them.
- A drawn graph or board cannot be read: type its edges, like `prim a-b:3, b-c:1, c-a:2`.
- Wording the English reader does not know is usually refused, but a sentence such as `Solve the differential equation ...`
  can have its words read as variables. Paste the formula alone.
