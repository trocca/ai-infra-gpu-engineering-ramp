"""Train an MLP on MNIST with YOUR engine (Day 4).

Acceptance criterion: >= 97.0% test accuracy, deterministic under SEED,
runnable as `make train` (i.e. `python -m src.train_mnist`).

No torch, no torchvision anywhere in this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .nn import Linear, ReLU, Sequential, softmax_cross_entropy
from .optim import Adam
from .tensor import Tensor

SEED = 1234
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "mnist_run.json"


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test).

    x arrays: float64, shape (N, 784), scaled to [0, 1] then standardized
    (subtract mean 0.1307, divide by std 0.3081 — the classic constants).
    y arrays: int64 class indices, shape (N,).

    TODO(Day 4): two acceptable routes, pick one:
      a) Download the 4 raw IDX files from an MNIST mirror (e.g.
         https://storage.googleapis.com/cvdf-datasets/mnist/) into data/
         and parse them with np.frombuffer (IDX format: big-endian magic,
         dims, then bytes). Cache locally; skip download if present.
      b) sklearn.datasets.fetch_openml("mnist_784", version=1) and split
         60k/10k. Slower first call, fewer lines.
    """
    raise NotImplementedError


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int,
                        rng: np.random.Generator):
    """Yield shuffled (xb, yb) minibatches. TODO(Day 4): permutation each
    epoch, drop nothing (last partial batch is fine)."""
    raise NotImplementedError


def evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    """Return test accuracy in [0, 1]. TODO(Day 4): argmax over logits,
    batched so you don't build a 10k-row graph in one go."""
    raise NotImplementedError


def main() -> None:
    rng = np.random.default_rng(SEED)
    x_train, y_train, x_test, y_test = load_mnist()

    model = Sequential(
        Linear(784, 256, rng=rng), ReLU(),
        Linear(256, 128, rng=rng), ReLU(),
        Linear(128, 10, rng=rng),
    )
    opt = Adam(model.parameters(), lr=LR)

    history: list[dict] = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        # TODO(Day 4): the loop —
        #   for xb, yb in iterate_minibatches(...):
        #       logits = model(Tensor(xb))
        #       loss = softmax_cross_entropy(logits, yb)
        #       opt.zero_grad(); loss.backward(); opt.step()
        # Track running mean loss.
        raise NotImplementedError

        epoch_s = time.perf_counter() - t0
        acc = evaluate(model, x_test, y_test)
        history.append({"epoch": epoch, "test_acc": acc, "seconds": epoch_s})
        print(f"epoch {epoch:2d}  test_acc {acc:.4f}  ({epoch_s:.1f}s)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"seed": SEED, "history": history}, indent=2))
    best = max(h["test_acc"] for h in history)
    print(f"best test accuracy: {best:.4f}  (target >= 0.97)")
    assert best >= 0.97, "Acceptance criterion not met yet — keep tuning."


if __name__ == "__main__":
    main()
