"""Lesson 02: matrices and matmul.

Run with: python lesson.py
Matmul written by hand with three loops, checked against torch.matmul,
then a speed comparison, then batch predictions on the house dataset.
"""

import time

import torch

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the house dataset is a 5x2 matrix")
print("=" * 60)

# Rows are houses, columns are features: [square meters, rooms].
X = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)
print(f"X =\n{X}")
print(f"X.shape   = {X.shape}   (5 rows, 2 columns)")
print(f"X[1]      = {X[1]}   (row 1: the second house)")
print(f"X[:, 0]   = {X[:, 0]}   (column 0: all square meters)")
print(f"X[3, 1]   = {X[3, 1]}   (row 3, column 1: rooms of the fourth house)")

print()
print("=" * 60)
print("STEP 2: transpose swaps rows and columns")
print("=" * 60)

print(f"X.T =\n{X.T}")
print(f"X.T.shape = {X.T.shape}   (was 5x2, now 2x5)")

print()
print("=" * 60)
print("STEP 3: matmul by hand, cell by cell")
print("=" * 60)


def matmul_loops(A, B):
    """Matrix multiplication with explicit loops. This IS the definition:
    C[i, j] = dot product of row i of A and column j of B."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "inner dimensions must match"
    C = torch.zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"A =\n{A}")
print(f"B =\n{B}")

C_manual = matmul_loops(A, B)
print("Cell by cell, row of A dot column of B:")
for i in range(2):
    for j in range(2):
        terms = " + ".join(f"{A[i, k].item():.0f}*{B[k, j].item():.0f}" for k in range(2))
        print(f"  C[{i},{j}] = {terms} = {C_manual[i, j].item():.0f}")

C_torch = torch.matmul(A, B)  # same as A @ B
print(f"torch.matmul(A, B) =\n{C_torch}")
assert torch.allclose(C_manual, C_torch), "manual matmul != torch.matmul"
print("OK: manual version and torch.matmul agree")

# Order matters: A @ B and B @ A are different in general.
print(f"B @ A =\n{B @ A}   (different from A @ B: order matters)")

print()
print("=" * 60)
print("STEP 4: speed race, loops vs torch.matmul")
print("=" * 60)

n = 60
R1 = torch.randn(n, n)  # random matrices, just for timing
R2 = torch.randn(n, n)

t0 = time.perf_counter()
slow = matmul_loops(R1, R2)
t_loops = time.perf_counter() - t0

t0 = time.perf_counter()
fast = torch.matmul(R1, R2)
t_torch = time.perf_counter() - t0

assert torch.allclose(slow, fast, atol=1e-4), "results differ"
print(f"{n}x{n} matmul with Python loops: {t_loops * 1000:10.1f} ms")
print(f"{n}x{n} matmul with torch.matmul: {t_torch * 1000:10.3f} ms")
print(f"speedup: about {t_loops / t_torch:,.0f}x. This is why GPUs and BLAS exist.")

print()
print("=" * 60)
print("STEP 5: predictions for ALL houses in one line")
print("=" * 60)

w = torch.tensor([2.0, 10.0])  # weights from lesson 01
b = 20.0

# One matmul replaces the per-house loop from lesson 01.
# X is (5x2), w is (2), result is (5): one prediction per house.
preds_matmul = X @ w + b

# Same thing with an explicit loop over rows, to prove they match.
preds_loop = torch.zeros(5)
for i in range(5):
    preds_loop[i] = torch.dot(X[i], w) + b

print(f"X @ w + b        = {preds_matmul}")
print(f"loop of dots + b = {preds_loop}")
assert torch.allclose(preds_matmul, preds_loop), "batch != loop"
print("OK: one matmul == five dot products")

print()
print("=" * 60)
print("STEP 6: the identity matrix changes nothing")
print("=" * 60)

I = torch.eye(2)  # 1 on the diagonal, 0 elsewhere
print(f"I =\n{I}")
print(f"X @ I =\n{X @ I}")
assert torch.allclose(X @ I, X), "X @ I should equal X"
print("OK: X @ I == X. The identity is the 'multiply by 1' of matrices.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
