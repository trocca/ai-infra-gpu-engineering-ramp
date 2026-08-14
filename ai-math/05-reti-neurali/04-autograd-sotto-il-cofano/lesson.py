"""Lesson 04: autograd under the hood.

Run with: python lesson.py
The computation graph explored like a stack trace, then backpropagation
done by hand through a real MLP and checked against autograd, gradient
by gradient. After this, backward() has no secrets.
"""

import torch

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the scalar chain, by hand (numbers from SPIEGAZIONE)")
print("=" * 60)

x, t = 2.0, 10.0
w = torch.tensor(3.0, requires_grad=True)
y = w * x
err = y - t
loss = err**2
print(f"forward:  y = {y.item()}, err = {err.item()}, loss = {loss.item()}")

# Backward by hand: local derivatives multiplied top to bottom.
dloss_derr = 2 * err.item()  # local derivative of err^2
dloss_dy = dloss_derr * 1.0  # local derivative of (y - t) wrt y
dloss_dw = dloss_dy * x  # local derivative of (w * x) wrt w
print(f"backward by hand: dloss/derr = {dloss_derr}, dloss/dy = {dloss_dy}, "
      f"dloss/dw = {dloss_dw}")

loss.backward()
print(f"autograd:         w.grad = {w.grad.item()}")
assert abs(w.grad.item() - dloss_dw) < 1e-6
print("OK. Now the same idea, on a real network.")

print()
print("=" * 60)
print("STEP 2: the graph IS a recorded call stack: walk it")
print("=" * 60)

w2 = torch.tensor(3.0, requires_grad=True)
loss2 = ((w2 * x) - t) ** 2
print("walking grad_fn from the loss backwards, like frames in a stack trace:")
node = loss2.grad_fn
depth = 0
while node is not None:
    print(f"  frame {depth}: {type(node).__name__}")
    node = node.next_functions[0][0] if node.next_functions else None
    depth += 1
print("PowBackward (err^2), SubBackward (y - t), MulBackward (w * x),")
print("AccumulateGrad (deposit the result into w.grad). The recording is real.")

print()
print("=" * 60)
print("STEP 3: backprop BY HAND through a one hidden layer MLP")
print("=" * 60)

# Tiny sizes so every tensor fits in your head: 3 inputs, 2 hidden, 1 out.
x_vec = torch.tensor([1.0, -2.0, 0.5])
target = torch.tensor(2.0)
W1 = torch.randn(3, 2, requires_grad=True)
b1 = torch.randn(2, requires_grad=True)
W2 = torch.randn(2, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)

# Forward, keeping every intermediate value (we need them for backward,
# exactly like autograd does internally).
z = x_vec @ W1 + b1  # pre activation, shape (2)
h = torch.relu(z)  # hidden activations, shape (2)
out = h @ W2 + b2  # network output, shape (1)
loss = (out - target) ** 2
print(f"z = {[round(v, 4) for v in z.tolist()]}")
print(f"h = {[round(v, 4) for v in h.tolist()]}   (ReLU killed the negatives)")
print(f"out = {out.item():.4f}, loss = {loss.item():.4f}")

# ---- Backward by hand, layer by layer, chain rule all the way ----
dout = 2 * (out - target)  # dloss/dout, local derivative of the square
db2_hand = dout.clone()  # out = h @ W2 + b2: local derivative wrt b2 is 1
dW2_hand = dout * h  # wrt W2: the other factor of the product, h
dh = dout * W2  # wrt h: the other factor, W2 (gradient flows back)
dz = dh * (z > 0).float()  # through ReLU: pass where z > 0, kill where z <= 0
db1_hand = dz.clone()  # z = x @ W1 + b1: wrt b1 the derivative is 1
dW1_hand = torch.outer(x_vec, dz)  # wrt W1: outer product input x incoming grad

# ---- The verdict: autograd computes the same numbers ----
loss.backward()
checks = [
    ("dloss/db2", db2_hand.detach(), b2.grad),
    ("dloss/dW2", dW2_hand.detach(), W2.grad),
    ("dloss/db1", db1_hand.detach(), b1.grad),
    ("dloss/dW1", dW1_hand.detach(), W1.grad),
]
for name, hand, auto in checks:
    match = torch.allclose(hand, auto, atol=1e-5)
    print(f"{name}: by hand {[round(v, 4) for v in hand.flatten().tolist()]}")
    print(f"{'':>10}  autograd {[round(v, 4) for v in auto.flatten().tolist()]}"
          f"   match = {match}")
    assert match, f"{name} mismatch: the hand derivation is wrong"
print("Every gradient matches to the decimal. backward() is exactly this,")
print("automated for graphs with billions of nodes.")

print()
print("=" * 60)
print("STEP 4: accumulation, the feature behind grad.zero_()")
print("=" * 60)

a = torch.tensor(2.0, requires_grad=True)
(a * 3).backward()
print(f"after first backward:  a.grad = {a.grad.item()}   (d(3a)/da = 3)")
(a * 3).backward()
print(f"after second backward: a.grad = {a.grad.item()}   (3 + 3: it ACCUMULATED)")
assert a.grad.item() == 6.0
a.grad.zero_()
(a * 3).backward()
print(f"after zero_ + backward: a.grad = {a.grad.item()}   (clean again)")
print("Now you know why every training loop calls grad.zero_(): gradients")
print("add up by design, because one tensor can feed many graph branches.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
