"""Lesson 03: the chain rule.

Run with: python lesson.py
The chain of (3x + 1)^2 link by link, then the loss chain of a single
house, every link computed by hand and checked against autograd.
"""

import torch

torch.manual_seed(0)

print("=" * 60)
print("STEP 1: the chain of y = (3x + 1)^2 at x = 2, link by link")
print("=" * 60)

x0 = 2.0
u = 3 * x0 + 1  # inner function
y = u**2  # outer function
print(f"x = {x0}")
print(f"u = 3x + 1 = {u}")
print(f"y = u^2    = {y}")

du_dx = 3.0  # derivative of 3x + 1
dy_du = 2 * u  # derivative of u^2, evaluated at u = 7
dy_dx = dy_du * du_dx
print(f"du/dx = 3")
print(f"dy/du = 2u = {dy_du}")
print(f"dy/dx = dy/du * du/dx = {dy_du} * {du_dx} = {dy_dx}")

# Independent check: y = 9x^2 + 6x + 1, derivative 18x + 6 = 42 at x = 2.
assert abs(dy_dx - (18 * x0 + 6)) < 1e-9
print("Check with the expanded formula 18x + 6 = 42: it matches.")

print()
print("=" * 60)
print("STEP 2: autograd climbs the same chain")
print("=" * 60)

x = torch.tensor(x0, requires_grad=True)
y_t = (3 * x + 1) ** 2
y_t.backward()
print(f"autograd says dy/dx = {x.grad.item()}")
assert abs(x.grad.item() - 42.0) < 1e-5
print("OK: hand chain and autograd agree.")

print()
print("=" * 60)
print("STEP 3: the loss chain of ONE house, by hand")
print("=" * 60)

# One house: 90 sqm (scaled to 0.9), real price 250.
x_house = 0.9
y_house = 250.0
w0 = 200.0

# Forward pass, printing every intermediate value of the chain.
pred = w0 * x_house
err = pred - y_house
loss = err**2
print(f"w    = {w0}")
print(f"pred = w * x     = {w0} * {x_house} = {pred}")
print(f"err  = pred - y  = {pred} - {y_house} = {err}")
print(f"loss = err^2     = {loss}")

# Backward pass BY HAND: one local slope per link, multiplied together.
dloss_derr = 2 * err  # derivative of err^2
derr_dpred = 1.0  # derivative of (pred - y) wrt pred
dpred_dw = x_house  # derivative of (w * x) wrt w
dloss_dw = dloss_derr * derr_dpred * dpred_dw
print(f"dloss/derr  = 2 * err = {dloss_derr}")
print(f"derr/dpred  = 1")
print(f"dpred/dw    = x = {dpred_dw}")
print(f"dloss/dw    = {dloss_derr} * {derr_dpred} * {dpred_dw} = {dloss_dw}")

print()
print("=" * 60)
print("STEP 4: autograd on the same chain, link values included")
print("=" * 60)

w = torch.tensor(w0, requires_grad=True)
pred_t = w * x_house
err_t = pred_t - y_house
loss_t = err_t**2

# retain_grad lets us inspect the gradient of INTERMEDIATE tensors too.
pred_t.retain_grad()
err_t.retain_grad()
loss_t.backward()

print(f"dloss/derr  by autograd = {err_t.grad.item()}   (hand: {dloss_derr})")
print(f"dloss/dpred by autograd = {pred_t.grad.item()}   (hand: {dloss_derr * derr_dpred})")
print(f"dloss/dw    by autograd = {w.grad.item()}   (hand: {dloss_dw})")

assert abs(err_t.grad.item() - dloss_derr) < 1e-4
assert abs(pred_t.grad.item() - dloss_derr * derr_dpred) < 1e-4
assert abs(w.grad.item() - dloss_dw) < 1e-4
print("OK: every link matches. backward() is the chain rule, automated.")

print()
print("=" * 60)
print("STEP 5: a longer chain, three links")
print("=" * 60)

# t = ((2x)^2 + 1)^3 at x = 1. Links: u = 2x, v = u^2 + 1, t = v^3.
x1 = 1.0
u1 = 2 * x1
v1 = u1**2 + 1
t1 = v1**3
dt_dv = 3 * v1**2
dv_du = 2 * u1
du_dx1 = 2.0
hand = dt_dv * dv_du * du_dx1
print(f"u = 2x = {u1}, v = u^2 + 1 = {v1}, t = v^3 = {t1}")
print(f"dt/dx by hand = {dt_dv} * {dv_du} * {du_dx1} = {hand}")

xt = torch.tensor(x1, requires_grad=True)
tt = ((2 * xt) ** 2 + 1) ** 3
tt.backward()
print(f"dt/dx by autograd = {xt.grad.item()}")
assert abs(xt.grad.item() - hand) < 1e-4
print("OK. Three links or three hundred: multiply the local slopes. That is backprop.")
print()
print("Done. Now open exercises.py and complete the TODOs.")
