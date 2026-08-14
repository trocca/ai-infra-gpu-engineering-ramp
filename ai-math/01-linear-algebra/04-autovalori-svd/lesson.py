"""Lesson 04: eigenvalues, eigenvectors and SVD.

Run with: python lesson.py
Checks the hand computed eigenvectors of EXPLANATION.md, rebuilds A from
its eigendecomposition, then runs SVD on the house dataset and shows how
well it compresses to rank 1.
"""

import torch

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: special directions of A = [[2, 1], [1, 2]]")
print("=" * 60)

A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
print(f"A =\n{A}")

for vec, expected_lam in [([1.0, 1.0], 3.0), ([1.0, -1.0], 1.0)]:
    v = torch.tensor(vec)
    Av = A @ v
    # If v is an eigenvector, Av must be exactly lambda * v.
    print(f"A @ {v.tolist()} = {Av.tolist()} = {expected_lam:.0f} * {v.tolist()}")
    assert torch.allclose(Av, expected_lam * v), "not an eigenvector?"
print("Both checks pass: same direction, only scaled.")

# A non special direction changes direction instead.
v = torch.tensor([1.0, 0.0])
print(f"A @ {v.tolist()} = {(A @ v).tolist()}   (direction changed: NOT an eigenvector)")

print()
print("=" * 60)
print("STEP 2: torch finds them for us, then we rebuild A")
print("=" * 60)

# eigh is for symmetric matrices. Returns eigenvalues in ascending order
# and eigenvectors as COLUMNS of Q.
eigvals, Q = torch.linalg.eigh(A)
print(f"eigenvalues  = {eigvals.tolist()}   (ascending order)")
print(f"eigenvectors (columns of Q) =\n{Q}")
print("Note: [-0.707, 0.707] is just [1, -1] scaled to length 1, sign is arbitrary.")

# Rebuild A = Q @ diag(eigvals) @ Q.T piece by piece.
Lambda = torch.diag(eigvals)
print(f"Lambda = diag(eigenvalues) =\n{Lambda}")
A_rebuilt = Q @ Lambda @ Q.T
print(f"Q @ Lambda @ Q.T =\n{A_rebuilt}")
assert torch.allclose(A_rebuilt, A, atol=1e-5), "eigendecomposition does not rebuild A"
print("OK: A == Q @ Lambda @ Q.T. The matrix IS its directions plus its scales.")

print()
print("=" * 60)
print("STEP 3: SVD of the house dataset (a rectangular matrix)")
print("=" * 60)

X = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)

# full_matrices=False gives the compact shapes: U (5x2), S (2), Vh (2x2).
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
print(f"U.shape = {U.shape}, S.shape = {S.shape}, Vh.shape = {Vh.shape}")
print(f"singular values S = {[round(s, 2) for s in S.tolist()]}   (always descending)")

X_rebuilt = U @ torch.diag(S) @ Vh
assert torch.allclose(X_rebuilt, X, atol=1e-3), "SVD does not rebuild X"
print("OK: X == U @ diag(S) @ Vh, rebuilt exactly from the three pieces.")

ratio = S[0] / S[1]
print(f"S[0] is about {ratio:.0f}x larger than S[1]:")
print("one direction (basically 'house size') carries almost all the information.")

print()
print("=" * 60)
print("STEP 4: rank 1 compression, keep only the top direction")
print("=" * 60)

# Keep only the first singular value and its directions.
# Column 0 of U, S[0], row 0 of Vh. outer product rebuilds a 5x2 matrix.
X_rank1 = S[0] * torch.outer(U[:, 0], Vh[0, :])
print(f"rank 1 approximation of X =\n{X_rank1}")

err = torch.linalg.norm(X - X_rank1)
rel = err / torch.linalg.norm(X)
print(f"reconstruction error (L2 norm) = {err.item():.3f}")
print(f"relative error = {rel.item() * 100:.3f}%")
print("Ten numbers compressed into rank 1, with tiny loss: because in this")
print("dataset rooms grow together with square meters, one direction is enough.")
print()
print("This idea (keep the strongest directions, drop the rest) is PCA,")
print("image compression and low rank adapters (LoRA), all in one.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
