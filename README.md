# AI Handwritten Math Solver

Write maths by hand on a board or paste a question from a paper, and get the answer with its working. A CNN trained for
this project reads each handwritten symbol, a layout reader works out how the symbols sit on the board and method modules
compute the answer with SymPy, SciPy and NetworkX. It runs offline.

## What it can do

**Write on the board:** sums, powers, fractions, `log`, `ln`, `sin`, `cos`, `tan`, limits with `x→a` under `lim`, equations
in any letter, a new line for each equation of a system, matrices in tall brackets with a word such as `rank` before them,
integrals with limits, `y'`, `dy/dx`, factorials and `nCr`. The reading stays visible and can be corrected before solving.

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

The CNN scores 90.66% per symbol on 53 classes. `1`, `l` and `i` or `0` and `o` are nearly the same shapes, so where a
symbol stands decides which it is. Whole expressions built from held-out test symbols, read back exactly (`check_reading.py`):

| Kind of expression | Read exactly |
| --- | ---: |
| Plain sums | 75.3% |
| Equations | 80.0% |
| Powers | 91.0% |
| Functions (log, ln, sin, cos, tan) | 87.5% |
| Logs with a base | 76.5% |
| Fractions | 86.5% |
| Limits | 71.5% |
| Equations in other letters | 64.5% |
| Two lines of equations | 66.0% |
| Matrices with a word | 21.0% |
| Integrals with limits | 57.5% |
| dy/dx | 60.0% |
| Factorials | 83.5% |
| Combinations (nCr) | 78.0% |

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
5. **English:** `english.py` turns a typed word problem into the same command language and the app shows what it understood.
6. **Solving:** `questions.py` sends each command to its method module. `solver.py` parses formulas with its own
   recursive-descent parser (no `eval`). A solve that runs past 60 seconds is stopped with a clear message.

## Data

- **MNIST** handwritten digits, 30,000 images, downloaded by Keras.
- **BHMSDS** (Basic Handwritten Math Symbols Dataset, MIT licence), 25,500 images, <https://github.com/wblachowski/bhmsds>.
- **EMNIST letters** (NIST), 75,000 images, <https://www.nist.gov/itl/products-and-services/emnist-dataset>.
- **Drawn by `train.py`,** 45,000 images of brackets, arrows, ∞, ∫, √, π, ∂, !, λ, μ, θ and curly x's.

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

```bash
git clone https://github.com/wblachowski/bhmsds ../bhmsds
pip install scikit-learn matplotlib
python train.py --bhmsds ../bhmsds/symbols --emnist <folder with emnist-letters-*.gz>
```

Training writes `models/symbol_cnn.keras`, `models/labels.json` and the figures and metrics in `reports/`.

## Tests

```bash
pip install pytest
python -m pytest
```

## Limits

- Handwriting has only been checked on scripted strokes and one real board so far, so the scores above are the test-set figures.
- A drawn graph or board cannot be read: type its edges, like `prim a-b:3, b-c:1, c-a:2`.
- Wording the English reader does not know is usually refused, but a sentence such as `Solve the differential equation ...`
  can have its words read as variables. Paste the formula alone.
