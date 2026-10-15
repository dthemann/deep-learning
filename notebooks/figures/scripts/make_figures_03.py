"""Figures and worked numbers for the companion guide of 03_tensors_and_autograd.ipynb.

Run:  python3 notebooks/figures/scripts/make_figures_03.py
Every number quoted in notebooks/03_tensors_and_autograd.md is printed by this script.
numpy + matplotlib only (plus sklearn's bundled make_moons generator); no torch.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

import math, time
from sklearn.datasets import make_moons

NB = "03"


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------------------------
# A tiny scalar autograd engine, re-implemented in pure Python (same semantics as the notebook's
# `Value`: closures add to the inputs' grads, intermediate grads are reset on every backward,
# leaves accumulate).
# ---------------------------------------------------------------------------------------------
class Value:
    def __init__(self, data, _children=(), _op="", name=""):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op; self.name = name

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f"**{n}")
        def _backward():
            self.grad += n * self.data ** (n - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Value(other) - self
    def __truediv__(self, other): return self * other ** -1.0
    def __rtruediv__(self, other): return Value(other) * self ** -1.0

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward(): self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward(): self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def _backward(): self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward(): self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def topo(self):
        order, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build(child)
                order.append(v)
        build(self)
        return order

    def backward(self):
        order = self.topo()
        for v in order:
            if v._prev: v.grad = 0.0
        self.grad = 1.0
        for v in reversed(order): v._backward()


def count_nodes(root):
    return len(root.topo())


# ---------------------------------------------------------------------------------------------
# Section 1 numbers (no chart): dtypes, strides, reductions
# ---------------------------------------------------------------------------------------------
def to_bfloat16(x):
    """Round a float32 to bfloat16 (keep the top 16 bits, round to nearest even) and return as float."""
    bits = np.array([x], dtype=np.float32).view(np.uint32)[0]
    lower = bits & 0xFFFF
    rounding = 0x7FFF + ((bits >> 16) & 1)
    bits = (bits + rounding) & 0xFFFF0000
    return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])


def nums_section1():
    banner("Section 1 numbers: dtypes, strides, reductions")
    for name, dt in [("float16", np.float16), ("float32", np.float32), ("float64", np.float64)]:
        fi = np.finfo(dt)
        print(f"{name}: bits={fi.bits} max={fi.max:.3e} eps={fi.eps:.3e} ~digits={-np.log10(float(fi.eps)):.1f}")
    print(f"bfloat16: bits=16 max={3.3895314e38:.3e} eps={2**-7:.3e} ~digits={-np.log10(2**-7):.1f} (7 stored mantissa bits)")
    print(f"0.1 in float32  = {float(np.float32(0.1)):.12f}")
    print(f"0.1 in float16  = {float(np.float16(0.1)):.12f}")
    print(f"0.1 in bfloat16 = {to_bfloat16(0.1):.12f}")
    x = np.arange(6.0).reshape(2, 3)
    print("x = arange(6).reshape(2,3) strides in elements:", tuple(s // x.itemsize for s in x.strides),
          "| contiguous:", x.flags["C_CONTIGUOUS"])
    t = x.T
    print("x.T strides in elements:", tuple(s // t.itemsize for s in t.strides), "| contiguous:", t.flags["C_CONTIGUOUS"],
          "| shares memory:", np.shares_memory(x, t))
    v = x.reshape(2, 3); v[0, 0] = 100.0
    print("after view[0,0] = 100, x[0,0] =", x[0, 0])
    r = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    print("reductions on [[1,5,3],[4,2,6]]: sum()=", r.sum(), "sum(dim=0)=", r.sum(0), "sum(dim=1)=", r.sum(1),
          "max(dim=1) values=", r.max(1), "indices=", r.argmax(1))
    z = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    stable = np.exp(z - z.max(1, keepdims=True)); stable /= stable.sum(1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    print("stable softmax row 0:", np.round(stable[0], 4), "| naive softmax row 0:", naive[0], "(overflow)")
    print("stable softmax row 1:", np.round(stable[1], 4))


# ---------------------------------------------------------------------------------------------
# Figure: broadcasting (3,1) x (1,4) -> (3,4) and per-channel image normalisation numbers
# ---------------------------------------------------------------------------------------------
def fig_broadcasting():
    banner("fig_broadcasting")
    col = np.arange(3.0).reshape(3, 1)
    row = np.arange(4.0).reshape(1, 4)
    out = col * row
    print("col (3,1) =", col.ravel(), "| row (1,4) =", row.ravel())
    print("col * row (3,4) =\n", out)
    # image normalisation example: a (3, 2, 2) image minus its (3, 1, 1) per-channel mean
    img = np.array([[[200., 100.], [150., 50.]],      # R
                    [[20., 60.], [40., 80.]],         # G
                    [[0., 255.], [255., 0.]]])        # B
    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True)
    print("image (3,2,2) per-channel mean (3,1,1):", mean.ravel(), "| std:", np.round(std.ravel(), 3))
    print("normalised R channel:\n", np.round(((img - mean) / std)[0], 3))
    print("(N,1) - (N,) with N=4 -> shape", (np.zeros((4, 1)) - np.zeros(4)).shape, "(the silent bug)")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), gridspec_kw={"width_ratios": [1.2, 3.2, 3.2]})
    vmax = out.max()
    for ax, arr, title in [(axes[0], col, "col: shape (3, 1)"), (axes[1], np.repeat(row, 3, 0), "row: shape (1, 4), stretched"),
                           (axes[2], out, "col * row: shape (3, 4)")]:
        ax.imshow(arr, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="equal")
        for (i, j), val in np.ndenumerate(arr):
            ax.text(j, i, f"{val:g}", ha="center", va="center", fontsize=9,
                    color=SURFACE if val > 0.55 * vmax else INK)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(arr.shape[1])); ax.set_yticks(range(arr.shape[0]))
        ax.grid(False)
    for spine in axes[1].spines.values():
        spine.set_linestyle("--")
    axes[1].text(1.5, 2.95, "rows 1 and 2 are virtual copies (no extra memory)", ha="center", va="top", fontsize=8, color=INK_2)
    fig.suptitle("Broadcasting: sizes are aligned from the right, size-1 dimensions are stretched",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "broadcasting_outer_product")


# ---------------------------------------------------------------------------------------------
# Figure: CPU matrix-product timing grows like n^3
# ---------------------------------------------------------------------------------------------
def fig_matmul_scaling():
    banner("fig_matmul_scaling")
    try:                                   # pin BLAS to one thread so the timings are not scheduler noise
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except Exception:                      # threadpoolctl missing: time the multi-threaded BLAS instead
        import contextlib; limiter = contextlib.nullcontext()
    sizes = [64, 128, 256, 512, 1024, 2048]
    times = []
    with limiter:
        for n in sizes:
            x = rng.standard_normal((n, n)).astype(np.float32)
            _ = x @ x; _ = x @ x                           # warm-up
            reps = 20 if n <= 512 else 3
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); _ = x @ x; ts.append(time.perf_counter() - t0)
            times.append(min(ts))                         # the minimum is the least noisy estimate
    times = np.array(times)
    print("one BLAS thread, float32, best of several repeats")
    print(f"{'n':>6s} {'ms':>10s} {'GFLOP':>8s} {'GFLOP/s':>8s}")
    for n, t in zip(sizes, times):
        flop = 2 * n ** 3
        print(f"{n:6d} {t * 1e3:10.3f} {flop / 1e9:8.3f} {flop / t / 1e9:8.1f}")
    print(f"doubling n from 1024 to 2048 multiplies the time by {times[-1] / times[-2]:.1f} (n^3 predicts 8)")
    print(f"doubling n from 64 to 128 multiplies the time by {times[1] / times[0]:.1f} (small n: overhead-dominated)")
    ref = times[3] * (np.array(sizes) / sizes[3]) ** 3

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sizes, times * 1e3, marker="o", color=SERIES[0], label="measured (numpy float32, one CPU thread)")
    ax.plot(sizes, ref * 1e3, linestyle="--", color=SERIES[1], label="n³ reference through n = 512")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("matrix size n (an n × n times n × n product)"); ax.set_ylabel("time per product (ms)")
    ax.set_title("A dense matrix product costs about 2n³ flops")
    ax.annotate(f"{times[-1] * 1e3:.0f} ms", (sizes[-1], times[-1] * 1e3), xytext=(-42, 6),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.annotate(f"{times[0] * 1e6:.0f} µs", (sizes[0], times[0] * 1e3), xytext=(8, -4),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.legend(loc="upper left")
    save(fig, NB, "matmul_time_vs_size")


# ---------------------------------------------------------------------------------------------
# Figure: forward vs reverse mode, number of sweeps to get a full Jacobian
# ---------------------------------------------------------------------------------------------
def fig_forward_vs_reverse():
    banner("fig_forward_vs_reverse")
    n = np.logspace(0, 6, 200)
    mnist_params = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
    print(f"{'inputs n':>10s} {'forward sweeps':>15s} {'reverse (1 output)':>19s} {'reverse (10 outputs)':>21s}")
    for k in [4, 100, 10_000, mnist_params]:
        print(f"{k:10d} {k:15d} {1:19d} {10:21d}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(n, n, color=SERIES[0], label="forward mode: one sweep per input")
    ax.plot(n, np.ones_like(n), color=SERIES[1], label="reverse mode, 1 output (a loss)")
    ax.plot(n, 10 * np.ones_like(n), color=SERIES[2], linestyle="--", label="reverse mode, 10 outputs")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of inputs n (parameters)"); ax.set_ylabel("sweeps needed for all derivatives")
    ax.set_title("Reverse mode: the whole gradient of one loss in a single sweep")
    ax.plot([mnist_params], [mnist_params], marker="o", color=SERIES[0])
    ax.annotate(f"MNIST MLP: {mnist_params:,} parameters\n= {mnist_params:,} forward-mode sweeps",
                (mnist_params, mnist_params), xytext=(-150, -30), textcoords="offset points", fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.legend(loc="upper left")
    save(fig, NB, "forward_vs_reverse_sweeps")


# ---------------------------------------------------------------------------------------------
# Section 2/3 numbers: worked backward pass on L = (a*b + c) * d, and the messy verification
# ---------------------------------------------------------------------------------------------
def nums_worked_graph():
    banner("Worked backward pass on L = (a*b + c) * d with a=2, b=-3, c=10, d=0.5")
    a, b, c, d = 2.0, -3.0, 10.0, 0.5
    e = a * b; f = e + c; L = f * d
    print(f"forward: e = a*b = {e}, f = e + c = {f}, L = f*d = {L}")
    L_bar = 1.0
    f_bar = L_bar * d; d_bar = L_bar * f
    e_bar = f_bar; c_bar = f_bar
    a_bar = e_bar * b; b_bar = e_bar * a
    print(f"reverse: L_bar = {L_bar}, f_bar = L_bar*d = {f_bar}, d_bar = L_bar*f = {d_bar}, e_bar = c_bar = f_bar = {e_bar}, "
          f"a_bar = e_bar*b = {a_bar}, b_bar = e_bar*a = {b_bar}")
    # forward mode, seed a_dot = 1
    a_dot, b_dot, c_dot, d_dot = 1.0, 0.0, 0.0, 0.0
    e_dot = a_dot * b + a * b_dot; f_dot = e_dot + c_dot; L_dot = f_dot * d + f * d_dot
    print(f"forward mode with a_dot=1: e_dot = {e_dot}, f_dot = {f_dot}, L_dot = {L_dot}  (= dL/da, one sweep per input)")
    Lf = lambda a, b, c, d: (a * b + c) * d
    eps = 1e-6; pt = [a, b, c, d]
    fd = [(Lf(*[v + eps * (i == j) for j, v in enumerate(pt)]) - Lf(*[v - eps * (i == j) for j, v in enumerate(pt)])) / (2 * eps)
          for i in range(4)]
    print("central finite differences:", [round(g, 6) for g in fd])
    # the same with the Value engine
    va, vb, vc, vd = Value(a, name="a"), Value(b, name="b"), Value(c, name="c"), Value(d, name="d")
    ve = va * vb; vf = ve + vc; vL = vf * vd
    vL.backward()
    print("Value engine grads a,b,c,d:", va.grad, vb.grad, vc.grad, vd.grad, "| nodes in graph:", count_nodes(vL))
    order = [v for v in vL.topo()]
    print("topological order (data):", [round(v.data, 2) for v in order])

    banner("Multiple paths and accumulation: y = x*x + x at x = 3")
    x = Value(3.0); y = x * x + x
    y.backward(); g1 = x.grad
    y.backward(); g2 = x.grad
    x.grad = 0.0; y.backward(); g3 = x.grad
    print(f"dy/dx = 2x+1 = 7 | after 1 backward: {g1} | after 2 (no zeroing): {g2} | zeroed then backward: {g3}")

    banner("Messy function: engine vs central differences")
    def messy(x, y, z):
        p = (x * y + z) ** 2 - x / (y - 0.5)
        q = (p * 0.1).exp() + (z ** 2 + 1.0).log() * y
        return q.tanh() * 3.0 + (x - z).relu() + 2.0 / y
    vx, vy, vz = Value(1.3), Value(-0.7), Value(2.1)
    out = messy(vx, vy, vz); out.backward()
    eng = [vx.grad, vy.grad, vz.grad]
    class F(float):
        def exp(self): return F(math.exp(self))
        def log(self): return F(math.log(self))
        def tanh(self): return F(math.tanh(self))
        def relu(self): return F(max(0.0, self))
        def __add__(s, o): return F(float(s) + float(o))
        def __radd__(s, o): return F(float(o) + float(s))
        def __sub__(s, o): return F(float(s) - float(o))
        def __rsub__(s, o): return F(float(o) - float(s))
        def __mul__(s, o): return F(float(s) * float(o))
        def __rmul__(s, o): return F(float(o) * float(s))
        def __truediv__(s, o): return F(float(s) / float(o))
        def __rtruediv__(s, o): return F(float(o) / float(s))
        def __pow__(s, o): return F(float(s) ** float(o))
    pt = [1.3, -0.7, 2.1]; num = []
    for i in range(3):
        plus = [F(v + (eps if j == i else 0.0)) for j, v in enumerate(pt)]
        minus = [F(v - (eps if j == i else 0.0)) for j, v in enumerate(pt)]
        num.append((messy(*plus) - messy(*minus)) / (2 * eps))
    print(f"messy(1.3, -0.7, 2.1) = {out.data:.6f}")
    print("engine  d/dx,d/dy,d/dz:", [f"{g:.6f}" for g in eng])
    print("numeric d/dx,d/dy,d/dz:", [f"{g:.6f}" for g in num])
    print(f"max |engine - numeric| = {max(abs(e - n) for e, n in zip(eng, num)):.1e}  | nodes in graph: {count_nodes(out)}")


# ---------------------------------------------------------------------------------------------
# Figure: the Value engine trains a 2-8-1 tanh MLP on two moons (+ timing against numpy)
# ---------------------------------------------------------------------------------------------
class Neuron:
    def __init__(self, n_in, nonlin, r):
        self.w = [Value(r.uniform(-1, 1)) for _ in range(n_in)]; self.b = Value(0.0); self.nonlin = nonlin   # micrograd init
    def __call__(self, x):
        act = sum((w * xi for w, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
    def parameters(self): return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, nonlin, r): self.neurons = [Neuron(n_in, nonlin, r) for _ in range(n_out)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]; return outs[0] if len(outs) == 1 else outs
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, sizes, r):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], i < len(sizes) - 1, r) for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]


def fig_value_engine_mlp():
    banner("fig_value_engine_mlp")
    r = np.random.default_rng(0)
    X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
    y_pm = 2.0 * y - 1.0
    model = MLP(2, [8, 1], r)
    n_params = len(model.parameters())
    out1 = model([Value(X[0, 0]), Value(X[0, 1])])
    print(f"MLP 2-8-1: {n_params} parameters | one forward pass for one input: {count_nodes(out1)} Value nodes "
          f"(incl. {n_params} parameter leaves and 2 inputs)")
    print("first 5 data points (x1, x2, label +-1):")
    for i in range(5):
        print(f"  {X[i, 0]:7.3f} {X[i, 1]:7.3f} {y_pm[i]:+.0f}")

    def step(lr):
        outs = [model([Value(a), Value(b)]) for a, b in X]
        loss = sum(((o - t) ** 2 for o, t in zip(outs, y_pm)), Value(0.0)) * (1.0 / len(X))
        for p in model.parameters(): p.grad = 0.0
        loss.backward()
        for p in model.parameters(): p.data -= lr * p.grad
        acc = float(np.mean([(o.data > 0) == (t > 0) for o, t in zip(outs, y_pm)]))
        return loss.data, acc, loss

    losses, accs = [], []
    t0 = time.perf_counter()
    STEPS, LR = 120, 0.3
    print(f"training: full-batch MSE on 40 points, lr = {LR}, {STEPS} steps")
    for s in range(1, STEPS + 1):
        l, a, loss_node = step(LR)
        losses.append(l); accs.append(a)
        if s in (1, 5, 10, 20, 40, 60, 80, 100, 120):
            print(f"  step {s:3d} | MSE {l:.4f} | train acc {a:.3f}")
    sec_per_step = (time.perf_counter() - t0) / STEPS
    print(f"full-batch loss graph: {count_nodes(loss_node)} nodes | {sec_per_step * 1e3:.1f} ms per step (engine)")

    # the same network as a batched numpy computation with a hand-written backward (what torch does)
    W1 = np.array([[w.data for w in n.w] for n in model.layers[0].neurons]).T  # (2, 8)
    b1 = np.array([n.b.data for n in model.layers[0].neurons])
    W2 = np.array([[w.data for w in model.layers[1].neurons[0].w]]).T          # (8, 1)
    b2 = np.array([model.layers[1].neurons[0].b.data])
    def numpy_step():
        h = np.tanh(X @ W1 + b1); yhat = (h @ W2 + b2).ravel()
        g = 2 * (yhat - y_pm) / len(X)
        gW2 = h.T @ g[:, None]; gb2 = g.sum(keepdims=True)
        gh = g[:, None] @ W2.T * (1 - h ** 2)
        gW1 = X.T @ gh; gb1 = gh.sum(0)
        return yhat, gW1, gb1, gW2, gb2
    t0 = time.perf_counter()
    for _ in range(200): numpy_step()
    np_per_step = (time.perf_counter() - t0) / 200
    print(f"numpy batched forward+backward: {np_per_step * 1e6:.0f} us per step | engine / numpy = {sec_per_step / np_per_step:.0f}x slower")

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 40),
                         np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 40))
    zz = np.array([model([Value(a), Value(b)]).data for a, b in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].plot(range(1, STEPS + 1), losses, color=SERIES[0])
    axes[0].set_xlabel("full-batch gradient step"); axes[0].set_ylabel("mean squared error")
    axes[0].set_title("Loss of the Value-engine MLP (2-8-1, lr 0.3)")
    axes[0].annotate(f"step {STEPS}: MSE {losses[-1]:.3f}, train acc {accs[-1]:.2f}", (STEPS, losses[-1]),
                     xytext=(-30, 60), textcoords="offset points", ha="right", fontsize=8.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].contourf(xx, yy, zz, levels=[-10, 0, 10], colors=[SERIES[0], SERIES[1]], alpha=0.12)
    axes[1].contour(xx, yy, zz, levels=[0], colors=[INK_2], linewidths=1.0)
    for cls, color, label in [(0, SERIES[0], "class 0 (target -1)"), (1, SERIES[1], "class 1 (target +1)")]:
        m = y == cls
        axes[1].scatter(X[m, 0], X[m, 1], s=28, color=color, edgecolors=SURFACE, linewidths=0.6, label=label)
    axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2"); axes[1].set_title("Decision boundary (40 x 40 scalar passes)")
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, NB, "value_engine_two_moons")


# ---------------------------------------------------------------------------------------------
# Figure: gradients accumulate across backward() calls
# ---------------------------------------------------------------------------------------------
def fig_grad_accumulation():
    banner("fig_grad_accumulation")
    def run(zero):
        x = Value(3.0); y = x * x + x; hist = []
        for _ in range(4):
            if zero: x.grad = 0.0
            y.backward(); hist.append(x.grad)
        return hist
    no_zero, zero = run(False), run(True)
    print("y = x*x + x at x=3, true dy/dx = 7")
    print("x.grad after backward #1..#4 without zeroing:", no_zero)
    print("x.grad after backward #1..#4 with zeroing   :", zero)
    calls = np.arange(1, 5); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(calls - w / 2, no_zero, width=w, color=SERIES[0], edgecolor=SURFACE, label="without zeroing (accumulates)")
    ax.bar(calls + w / 2, zero, width=w, color=SERIES[1], edgecolor=SURFACE, label="x.grad = 0 before each backward")
    ax.axhline(7, color=INK_2, linestyle=":", linewidth=1)
    ax.text(4.45, 7.4, "true derivative 7", fontsize=8.5, color=INK_2, ha="right")
    for c, v in zip(calls, no_zero):
        ax.annotate(f"{v:g}", (c - w / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(calls); ax.set_xticklabels([f"backward #{c}" for c in calls])
    ax.set_ylabel("x.grad"); ax.set_title("Leaf gradients add up: zero them before every backward()")
    ax.legend(loc="upper left")
    save(fig, NB, "gradient_accumulation")


# ---------------------------------------------------------------------------------------------
# Figure: tanh, tanh', tanh'' (what create_graph=True gives you)
# ---------------------------------------------------------------------------------------------
def fig_tanh_derivatives():
    banner("fig_tanh_derivatives")
    x = np.linspace(-4, 4, 401)
    t = np.tanh(x); d1 = 1 - t ** 2; d2 = -2 * t * (1 - t ** 2)
    h = 1e-5
    fd1 = (np.tanh(x + h) - np.tanh(x - h)) / (2 * h)
    fd2 = (np.tanh(x + h) - 2 * np.tanh(x) + np.tanh(x - h)) / h ** 2
    print("max |fd tanh' - (1 - tanh^2)| =", f"{np.abs(fd1 - d1).max():.1e}", "| max |fd tanh'' - formula| =", f"{np.abs(fd2 - d2).max():.1e}")
    print(f"{'x':>5s} {'tanh':>8s} {'tanh_prime':>11s} {'tanh_2nd':>9s}")
    for xv in [-2, -1, 0, 1, 2]:
        tv = math.tanh(xv); print(f"{xv:5d} {tv:8.4f} {1 - tv ** 2:11.4f} {-2 * tv * (1 - tv ** 2):9.4f}")
    xstar = math.atanh(1 / math.sqrt(3)); tv = math.tanh(xstar)
    print(f"tanh'' extremes at x = +-{xstar:.4f}, value -+{abs(-2 * tv * (1 - tv ** 2)):.4f}; tanh' max 1 at x=0")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, t, color=SERIES[0], label="tanh(x)")
    ax.plot(x, d1, color=SERIES[1], label="tanh'(x) = 1 - tanh²(x)")
    ax.plot(x, d2, color=SERIES[2], label="tanh''(x) = -2 tanh(x)(1 - tanh²(x))")
    ax.plot([0], [1], marker="o", color=SERIES[1])
    ax.annotate("slope is largest at 0", (0, 1), xytext=(12, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.plot([-xstar, xstar], [abs(-2 * tv * (1 - tv ** 2)), -abs(-2 * tv * (1 - tv ** 2))], linestyle="none", marker="o", color=SERIES[2])
    ax.set_xlabel("x"); ax.set_ylabel("value"); ax.set_title("tanh and its first two derivatives")
    ax.legend(loc="lower right")
    save(fig, NB, "tanh_derivatives")


# ---------------------------------------------------------------------------------------------
# Figure: softplus, naive vs stable, in float32 (custom autograd.Function example)
# ---------------------------------------------------------------------------------------------
def fig_softplus_stability():
    banner("fig_softplus_stability")
    x = np.linspace(-20, 120, 561).astype(np.float32)
    with np.errstate(over="ignore"):
        naive = np.log(1 + np.exp(x))
    stable = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    first_inf = x[np.isinf(naive)][0]
    print("float32 max =", f"{np.finfo(np.float32).max:.3e}", "| exp(x) overflows for x >", f"{math.log(np.finfo(np.float32).max):.2f}",
          "| first inf in naive at x =", f"{first_inf:.2f}")
    print(f"{'x':>6s} {'naive':>10s} {'stable':>10s} {'sigmoid(x)':>11s}")
    for xv in [-20.0, 0.0, 20.0, 88.0, 89.0, 100.0]:
        xv32 = np.float32(xv)
        with np.errstate(over="ignore"):
            nv = float(np.log(1 + np.exp(xv32)))
        sv = float(np.maximum(xv32, 0) + np.log1p(np.exp(-abs(xv32))))
        print(f"{xv:6.0f} {nv:10.4f} {sv:10.4f} {1 / (1 + math.exp(-xv)):11.6f}")
    h = 1e-6; xs = np.array([-3.0, -0.5, 0.0, 2.0])
    sp = lambda z: np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    print("d softplus/dx by finite differences:", np.round((sp(xs + h) - sp(xs - h)) / (2 * h), 6),
          "| sigmoid:", np.round(1 / (1 + np.exp(-xs)), 6))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ok = ~np.isinf(naive)
    ax.plot(x, stable, color=SERIES[0], label="stable: max(x, 0) + log1p(exp(-|x|))")
    ax.plot(x[ok], naive[ok], color=SERIES[1], linestyle="--", label="naive: log(1 + exp(x))  (float32)")
    ax.axvline(first_inf, color=AXIS, linewidth=1)
    ax.annotate(f"naive = inf for x ≥ {first_inf:.0f}\n(exp overflows float32)", (first_inf, 60), xytext=(8, 0),
                textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlabel("x"); ax.set_ylabel("softplus(x)"); ax.set_title("Softplus: the naive formula overflows, the stable one does not")
    ax.legend(loc="upper left")
    save(fig, NB, "softplus_stable_vs_naive")


# ---------------------------------------------------------------------------------------------
# Section 4 numbers: VJPs / Jacobian, Hessian of a quadratic, swish derivative
# ---------------------------------------------------------------------------------------------
def nums_section4():
    banner("Section 4 numbers: Jacobian by VJPs, Hessian, swish")
    x = np.array([1.0, 2.0, 3.0])
    f = lambda x: np.array([x[0] * x[1], x[1] ** 2 + x[2], x[0] * x[2] * 10])
    J = np.array([[x[1], x[0], 0], [0, 2 * x[1], 1], [10 * x[2], 0, 10 * x[0]]])
    h = 1e-6
    Jfd = np.array([(f(x + h * np.eye(3)[j]) - f(x - h * np.eye(3)[j])) / (2 * h) for j in range(3)]).T
    print("y = f(x) =", f(x)); print("Jacobian (analytic rows):\n", J); print("max |J - finite diff| =", f"{np.abs(J - Jfd).max():.1e}")
    for v in [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 1.0])]:
        print("v =", v, "-> v^T J =", v @ J)
    A = np.array([[2.0, 1, 0], [0, 3, 1], [1, 0, 4]]); x0 = np.array([1.0, -1, 2])
    q = lambda x: 0.5 * x @ A @ x
    g = np.array([(q(x0 + h * np.eye(3)[i]) - q(x0 - h * np.eye(3)[i])) / (2 * h) for i in range(3)])
    print("quadratic f = 0.5 x^T A x at x0 =", x0, ": f =", q(x0), "| gradient (fd):", np.round(g, 6), "| (A + A^T)/2 x0:", 0.5 * (A + A.T) @ x0)
    print("Hessian (A + A^T)/2:\n", 0.5 * (A + A.T))
    s = lambda z: 1 / (1 + np.exp(-z)); sw = lambda z: z * s(z); xs = np.array([-2.0, 0.0, 1.5])
    print("swish'(x) finite diff:", np.round((sw(xs + h) - sw(xs - h)) / (2 * h), 6), "| formula s(1 + x(1 - s)):", np.round(s(xs) * (1 + xs * (1 - s(xs))), 6))
    print("sigmoid saves its output s; d sigmoid = s(1-s): at x=0.5 s =", round(float(s(0.5)), 6), "s(1-s) =", round(float(s(0.5) * (1 - s(0.5))), 6))


# ---------------------------------------------------------------------------------------------
# Figure: linear regression by raw gradient steps on 8 points
# ---------------------------------------------------------------------------------------------
def fig_linear_regression():
    banner("fig_linear_regression")
    r = np.random.default_rng(3)
    x = np.round(r.uniform(-2, 2, 8), 2)
    y = np.round(3 * x - 2 + 0.5 * r.standard_normal(8), 2)
    print("data (x, y) from y = 3x - 2 + noise(std 0.5):")
    for xi, yi in zip(x, y): print(f"  {xi:6.2f} {yi:7.2f}")
    X = np.c_[x, np.ones_like(x)]
    w_ls, b_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"least squares: w = {w_ls:.4f}, b = {b_ls:.4f}")
    w = b = 0.0; lr = 0.1; hist = [(w, b)]; losses = []
    for step in range(1, 201):
        yhat = w * x + b
        loss = np.mean((yhat - y) ** 2)
        gw = np.mean(2 * (yhat - y) * x); gb = np.mean(2 * (yhat - y))
        if step <= 3:
            print(f"step {step}: loss {loss:.4f} | dL/dw = mean(2(yhat-y)x) = {gw:.4f}, dL/db = mean(2(yhat-y)) = {gb:.4f} "
                  f"| w <- {w:.4f} - 0.1*{gw:.4f} = {w - lr * gw:.4f}, b <- {b - lr * gb:.4f}")
        w -= lr * gw; b -= lr * gb
        losses.append(loss); hist.append((w, b))
        if step in (10, 50, 100, 200):
            print(f"step {step}: loss {loss:.4f} | w = {w:.4f}, b = {b:.4f}")
    print(f"final |w - w_ls| = {abs(w - w_ls):.1e}, |b - b_ls| = {abs(b - b_ls):.1e}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    loss_min = np.mean((X @ np.array([w_ls, b_ls]) - y) ** 2)
    excess = np.maximum(np.array(losses) - loss_min, 1e-13)
    print(f"least-squares MSE (the floor) = {loss_min:.4f}; MSE above the floor at step 1, 10, 50: {excess[0]:.3g}, {excess[9]:.3g}, {excess[49]:.3g}")
    axes[0].plot(range(1, 201), excess, color=SERIES[0])
    axes[0].set_yscale("log"); axes[0].set_xlabel("gradient step"); axes[0].set_ylabel(f"MSE minus its minimum ({loss_min:.3f})")
    axes[0].set_title("Loss falls geometrically (lr = 0.1)")
    axes[0].annotate(f"step 10: {excess[9]:.3f} above the floor", (10, excess[9]), xytext=(20, 10), textcoords="offset points", fontsize=8.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    xs = np.linspace(-2.2, 2.2, 50)
    axes[1].scatter(x, y, color=INK_2, s=36, zorder=3, label="8 data points")
    for k, (si, color) in enumerate(zip([0, 1, 3, 10, 200], SERIES)):
        wi, bi = hist[si]
        axes[1].plot(xs, wi * xs + bi, color=color, linewidth=1.6 if si < 200 else 2.2, label=f"after step {si}: w={wi:.2f}, b={bi:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_title("The line after 0, 1, 3, 10 and 200 steps")
    axes[1].legend(fontsize=7.5, loc="upper left")
    save(fig, NB, "linear_regression_gradient_steps")


# ---------------------------------------------------------------------------------------------
# Figure: logistic regression on two moons (the best line gets about 84%)
# ---------------------------------------------------------------------------------------------
def fig_logistic_regression():
    banner("fig_logistic_regression")
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    w = np.zeros(2); b = 0.0; lr = 0.5
    bce = lambda z, y: np.mean(np.logaddexp(0, z) - y * z)   # softplus(z) - y z
    print(f"step 0: BCE = {bce(X @ w + b, y):.4f} (= log 2 for w = b = 0)")
    for step in range(1, 501):
        z = X @ w + b; p = 1 / (1 + np.exp(-z)); g = (p - y) / len(y)
        if step in (125, 250, 375, 500):
            print(f"step {step:3d}: BCE {bce(z, y):.4f} | train acc {((z > 0) == y).mean():.4f}")
        w -= lr * X.T @ g; b -= lr * g.sum()
    z = X @ w + b; acc = ((z > 0) == y).mean()
    print(f"final w = {np.round(w, 4)}, b = {b:.4f}, accuracy {acc:.4f} = {int(((z > 0) == y).sum())}/400")
    print("probability for the first 3 points:", np.round(1 / (1 + np.exp(-z[:3])), 3), "labels:", y[:3])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.linspace(-1.5, 2.5, 50)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=INK_2, linewidth=1.2, label="decision line w·x + b = 0")
    for cls, color in [(0, SERIES[0]), (1, SERIES[1])]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=18, color=color, edgecolors=SURFACE, linewidths=0.4, label=f"class {cls}")
    wrong = (z > 0) != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=60, facecolors="none", edgecolors=INK, linewidths=0.8, label=f"misclassified ({wrong.sum()})")
    ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title(f"Logistic regression on two moons: train accuracy {acc:.1%}")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, NB, "logistic_regression_two_moons")


def nums_mnist_params():
    banner("MNIST MLP parameter counts")
    for sizes in ([784, 256, 128, 10], [784, 256, 128, 64, 10]):
        total = 0; parts = []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            parts.append(f"W({fi}x{fo})={fi * fo:,} + b({fo})={fo:,}"); total += fi * fo + fo
        print("-".join(map(str, sizes)) + ":", " | ".join(parts), f"| total {total:,} | tensors: {2 * (len(sizes) - 1)}")
    print("He init scale sqrt(2/fan_in): 784 ->", f"{math.sqrt(2 / 784):.4f}", "256 ->", f"{math.sqrt(2 / 256):.4f}", "128 ->", f"{math.sqrt(2 / 128):.4f}")
    print("steps per epoch, batch 64: 55,000 train ->", math.ceil(55000 / 64), "| FAST_RUN 10,000 ->", math.ceil(10000 / 64))


# ---------------------------------------------------------------------------------------------
# Figure: same loop, three update rules (preview of torch.optim)
# ---------------------------------------------------------------------------------------------
def fig_optimizers_preview():
    banner("fig_optimizers_preview")
    r = np.random.default_rng(3)
    x = np.round(r.uniform(1, 5, 8), 2); y = np.round(3 * x - 2 + 0.5 * r.standard_normal(8), 2)
    print("data x in [1, 5] (not centred, so slope and intercept are entangled):")
    print("  x =", x); print("  y =", y)
    X = np.c_[x, np.ones_like(x)]
    theta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    loss_min = np.mean((X @ theta_ls - y) ** 2)
    H = 2 * X.T @ X / len(y); ev = np.linalg.eigvalsh(H)
    print(f"least squares: w = {theta_ls[0]:.4f}, b = {theta_ls[1]:.4f}, minimum MSE = {loss_min:.4f}")
    print(f"Hessian eigenvalues {ev[0]:.3f}, {ev[1]:.3f} -> condition number {ev[1] / ev[0]:.0f}; plain SGD is stable only for lr < 2/{ev[1]:.2f} = {2 / ev[1]:.3f}")
    def grad(th):
        res = X @ th - y
        return np.mean(res ** 2), 2 * X.T @ res / len(y)
    STEPS = 200
    curves = {}
    th = np.zeros(2); c = []
    for _ in range(STEPS):
        l, g = grad(th); c.append(l); th -= 0.1 * g
    curves["SGD, lr 0.1"] = c
    th = np.zeros(2); v = np.zeros(2); c = []
    for _ in range(STEPS):
        l, g = grad(th); c.append(l); v = 0.9 * v + g; th -= 0.03 * v
    curves["SGD + momentum 0.9, lr 0.03"] = c
    th = np.zeros(2); m = np.zeros(2); s2 = np.zeros(2); c = []
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, STEPS + 1):
        l, g = grad(th); c.append(l)
        m = b1 * m + (1 - b1) * g; s2 = b2 * s2 + (1 - b2) * g ** 2
        th -= 0.3 * (m / (1 - b1 ** t)) / (np.sqrt(s2 / (1 - b2 ** t)) + eps)
    curves["Adam, lr 0.3"] = c
    th = np.zeros(2); c = []
    for _ in range(STEPS):
        l, g = grad(th); c.append(l); th -= 0.15 * g
    print(f"SGD with lr 0.15 (above the limit) after 10 steps: MSE = {c[9]:.3g}, after 200: {c[-1]:.3g}  -> diverges")
    print(f"{'optimizer':>28s} {'MSE-min @10':>12s} {'@50':>10s} {'@100':>10s} {'@200':>10s}")
    for k, c in curves.items():
        print(f"{k:>28s} {c[9] - loss_min:12.4g} {c[49] - loss_min:10.4g} {c[99] - loss_min:10.4g} {c[199] - loss_min:10.4g}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (k, c), color in zip(curves.items(), SERIES):
        ax.plot(range(1, STEPS + 1), np.maximum(np.array(c) - loss_min, 1e-12), color=color, label=k)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE above its minimum")
    ax.set_title("Same training loop, three one-line optimizer swaps (ill-conditioned fit)")
    ax.legend(loc="upper right")
    save(fig, NB, "optimizers_same_loop")


def main():
    nums_section1()
    fig_broadcasting()
    fig_matmul_scaling()
    fig_forward_vs_reverse()
    nums_worked_graph()
    fig_value_engine_mlp()
    fig_grad_accumulation()
    fig_tanh_derivatives()
    fig_softplus_stability()
    nums_section4()
    fig_linear_regression()
    fig_logistic_regression()
    nums_mnist_params()
    fig_optimizers_preview()


if __name__ == "__main__":
    main()
