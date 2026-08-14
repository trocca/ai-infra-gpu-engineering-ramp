# math-for-ai

A hands-on path to rebuild the mathematics behind modern neural networks. No abstract proofs. Every concept comes as a pair: a plain-language explanation and a PyTorch script you can run line by line — even inside a debugger.

The core idea: **if you can put a breakpoint on a number, that number can't scare you anymore.**

## Who this is for

Anyone coming from the software world (debugging, systems, code) who wants to rebuild the foundations of the math. No prerequisites beyond high-school mathematics. Every symbol is explained in words the first time it appears.

## The map

| Module | Topic | Lessons |
|---|---|---|
| [01-linear-algebra](01-linear-algebra/) | Vectors, matrices, matmul, norms, eigenvalues, SVD | 4 |
| [02-calcolo](02-calcolo/) | Derivatives, gradients, chain rule, Jacobians | 4 |
| [03-probabilita](03-probabilita/) | Random variables, likelihood, entropy, KL | 3 |
| [04-ottimizzazione](04-ottimizzazione/) | Gradient descent, SGD, momentum, Adam | 4 |
| [05-reti-neurali](05-reti-neurali/) | From linear regression to attention, all from scratch | 5 |

All modules are complete: 20 lessons, each with an explanation, a runnable script, exercises, and tests. Estimated time for the whole path: 10–12 weeks at 4–5 hours per week.

The modules build on each other — do them in order. Some examples return throughout the path, in particular a small dataset of 5 houses (square meters, rooms, price) that you will revisit in every module under a different light.

Guided pages with figures for every module are on the site: [AI Math](https://trocca.github.io/ai-infra-gpu-engineering-ramp/docs/ai-math/).

## How to work through a lesson

Every lesson folder contains the same five files. The workflow:

1. Read `EXPLANATION.md`. It contains the intuition, the definition in plain words, and a numeric example worked by hand.
2. Run `python lesson.py`. The script redoes the same steps as the explanation, printing every intermediate value. Even better: open it in a debugger and set breakpoints.
3. Open `exercises.py` and complete the functions marked with `# TODO`. The docstrings explain what each function must do.
4. Run `pytest` from the lesson folder. When all tests pass, the lesson is done.
5. `solutions.py` contains the full solutions. Look at it only after a serious attempt, or to compare your approach with the proposed one.

Some lessons save plots into a `figures/` subfolder. No display needed: the plots are written to PNG files.

## Setup

Python 3.10 or newer is required.

```
python -m venv .venv
.venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

Note on PyTorch: the CPU build is all this path needs. For the lighter install:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Quick check that everything works:

```
python -c "import torch; print(torch.__version__)"
```

## Reference books

The path is anchored to free and legal texts. Each module cites exact chapters and sections.

* Deisenroth, Faisal, Ong, **Mathematics for Machine Learning**: the backbone of the whole path. Free at [mml-book.github.io](https://mml-book.github.io).
* Strang, **Introduction to Linear Algebra** and the **MIT 18.06** video lectures: support for module 01. The lectures are on [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
* Parr, Howard, **The Matrix Calculus You Need For Deep Learning**: support for module 02. Free at [arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528).
* Blitzstein, Hwang, **Introduction to Probability** (Harvard Stat 110): support for module 03. Free at [projects.iq.harvard.edu/stat110](https://projects.iq.harvard.edu/stat110).
* Boyd, Vandenberghe, **Convex Optimization** (early chapters only): support for module 04. Free at [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
* Prince, **Understanding Deep Learning**: main text for module 05. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

Reading the books is not mandatory. They are there for when you want to go deeper on a topic.
