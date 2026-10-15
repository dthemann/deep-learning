"""Figures for the companion guide of Notebook 00 · Course Overview and Setup.

Everything here is numpy + matplotlib (plus scikit-learn's bundled ``make_moons`` generator).
The tiny MLP, its Adam optimizer and the binary cross-entropy loss are re-implemented in numpy
so that the notebook's experiments can be reproduced without PyTorch. The weight initialisation
copies PyTorch's ``nn.Linear`` default (uniform in +-1/sqrt(fan_in) for weights and biases).

Run:  python3 notebooks/figures/scripts/make_figures_00.py
Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

import time
from matplotlib.patches import Patch, Rectangle
from sklearn.datasets import make_moons

NB = "00"
SEED = 42            # same seed as the notebook, so the moons data are the very same points


# ----------------------------------------------------------------------------------------------
# A tiny numpy MLP (2 -> hidden -> hidden -> 1, ReLU), BCE-with-logits loss, Adam optimizer
# ----------------------------------------------------------------------------------------------
def sigmoid(z):
    """Numerically stable logistic function (no overflow warnings for large |z|)."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def init_params(hidden=32, depth=2, seed=SEED):
    """PyTorch-style init: weights and biases uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)]."""
    r = np.random.default_rng(seed)
    sizes = [2] + [hidden] * depth + [1]
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / np.sqrt(n_in)
        params.append(r.uniform(-bound, bound, size=(n_out, n_in)))   # W, shape (out, in)
        params.append(r.uniform(-bound, bound, size=(n_out,)))        # b, shape (out,)
    return params


def forward(params, X):
    """Return the logits (N, 1) and a cache of (layer input, pre-activation) per layer."""
    h, cache, n_layers = X, [], len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = h @ W.T + b
        cache.append((h, z))
        h = np.maximum(z, 0.0) if i < n_layers - 1 else z
    return h, cache


def bce_with_logits(z, y):
    """Mean binary cross-entropy computed from logits, the stable way nn.BCEWithLogitsLoss does."""
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def backward(params, cache, logits, y):
    """Gradients of the mean BCE loss with respect to every parameter (plain backpropagation)."""
    N, n_layers = y.shape[0], len(params) // 2
    grads = [None] * len(params)
    g = (sigmoid(logits) - y) / N                   # dJ/dlogit = (p - y) / N
    for i in reversed(range(n_layers)):
        h_in, z = cache[i]
        if i < n_layers - 1:
            g = g * (z > 0)                         # through the ReLU
        grads[2 * i] = g.T @ h_in                   # dJ/dW
        grads[2 * i + 1] = g.sum(axis=0)            # dJ/db
        g = g @ params[2 * i]                       # dJ/d(layer input)
    return grads


class Adam:
    """torch.optim.Adam with default betas (0.9, 0.999) and eps 1e-8, no weight decay."""
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def accuracy(params, X, y):
    logits, _ = forward(params, X)
    return float(np.mean((logits > 0) == (y > 0.5)))


def train_mlp(X, y, hidden=32, lr=1e-2, epochs=1000, seed=SEED, X_test=None, y_test=None):
    """Full-batch Adam training, like the notebook's train_mlp. Returns (params, losses, test_losses)."""
    params = init_params(hidden, 2, seed)
    opt = Adam(params, lr)
    losses, test_losses = [], []
    for _ in range(epochs):
        logits, cache = forward(params, X)
        losses.append(bce_with_logits(logits, y))
        opt.step(params, backward(params, cache, logits, y))
        if X_test is not None:
            test_losses.append(bce_with_logits(forward(params, X_test)[0], y_test))
    return params, losses, test_losses


def count_parameters(params):
    return int(sum(p.size for p in params))


def moons(noise=0.2, seed=SEED, n=1000):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64).reshape(-1, 1)


# cached data and the main training run (shared by several figures)
X_TR, Y_TR = moons(0.2, SEED)
X_TE, Y_TE = moons(0.2, SEED + 1)
_MAIN_RUN = {}


def main_run():
    if not _MAIN_RUN:
        params, losses, test_losses = train_mlp(X_TR, Y_TR, 32, 1e-2, 1000, SEED, X_TE, Y_TE)
        _MAIN_RUN.update(params=params, losses=losses, test_losses=test_losses)
    return _MAIN_RUN


# ----------------------------------------------------------------------------------------------
# Section 1: the timeline
# ----------------------------------------------------------------------------------------------
MILESTONES = [
    (1958, "Perceptron\n(Rosenblatt)"),
    (1986, "Backpropagation\n(Rumelhart et al.)"),
    (1998, "LeNet-5\n(LeCun et al.)"),
    (2012, "AlexNet\n(Krizhevsky et al.)"),
    (2015, "ResNet, BatchNorm,\nAdam"),
    (2017, "Transformer\n(Vaswani et al.)"),
    (2018, "BERT / GPT"),
    (2020, "Diffusion models,\nscaling laws, GPT-3"),
    (2023, "LLM era"),
]


def fig_timeline():
    years = [y for y, _ in MILESTONES]
    gaps = np.diff(years)
    print("\n[fig_timeline] milestone years and gap to the previous milestone (years):")
    for (y, label), gap in zip(MILESTONES, [None] + list(gaps)):
        print(f"  {y}  {label.replace(chr(10), ' '):<36} gap: {'-' if gap is None else gap}")
    print(f"  mean gap before 2012: {np.mean(gaps[:3]):.1f} years | mean gap from 2012 on: {np.mean(gaps[3:]):.1f} years")

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.hlines(0, 1955, 2026, color=AXIS, linewidth=1)
    ax.scatter(years, [0] * len(years), s=60, zorder=3, color=BLUE)
    for i, (year, label) in enumerate(MILESTONES):
        offset = 0.6 if i % 2 == 0 else -0.6
        ax.annotate(f"{year}\n{label}", (year, 0), (year, offset), ha="center",
                    va="bottom" if offset > 0 else "top", fontsize=8, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.annotate(f"{gaps[0]} quiet years", ((1958 + 1986) / 2, -0.12), ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_ylim(-2.2, 2.2); ax.set_xlim(1955, 2027)
    ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel("year"); ax.set_title("A compressed history of deep learning")
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    save(fig, NB, "timeline_milestones")


# ----------------------------------------------------------------------------------------------
# Section 2: the course map and its time budget
# ----------------------------------------------------------------------------------------------
COURSE_MAP = [
    (0, "Course Overview and Setup", 45, 1), (1, "Mathematical Foundations", 120, 1),
    (2, "Neural Networks from Scratch (NumPy)", 120, 1), (3, "Tensors and Autograd", 120, 1),
    (4, "Training Deep Networks", 150, 2), (5, "The PyTorch Workflow and Best Practices", 120, 2),
    (6, "Convolutional Neural Networks", 150, 3), (7, "Transfer Learning and Modern Vision", 120, 3),
    (8, "Sequence Models: RNNs, LSTMs and GRUs", 120, 4), (9, "Attention and Transformers", 150, 4),
    (10, "NLP with Pretrained Transformers", 120, 4), (11, "Autoencoders and VAEs", 120, 5),
    (12, "GANs and Diffusion Models", 150, 5), (13, "Reinforcement Learning", 120, 6),
    (14, "Efficient Training and Scaling", 120, 7), (15, "Deployment and Model Optimization", 120, 7),
    (16, "Interpretability, Robustness and Uncertainty", 120, 7), (17, "Capstone Projects", 90, 7),
]
PART_NAMES = {1: "I Foundations", 2: "II Training", 3: "III Vision", 4: "IV Sequences and language",
              5: "V Generative", 6: "VI Reinforcement learning", 7: "VII Engineering and practice"}
EIGHT_WEEK_PLAN = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16, 17]]


def fig_time_budget():
    minutes = np.array([m for _, _, m, _ in COURSE_MAP])
    total = minutes.sum()
    print("\n[fig_time_budget] course map time budget:")
    print(f"  {len(COURSE_MAP)} notebooks | total minutes {total} | total hours {total / 60:.2f}")
    print(f"  per week on the eight-week plan: {total / 60 / 8:.2f} h of reading (plus about the same for exercises)")
    for part in range(1, 8):
        part_min = minutes[[p == part for _, _, _, p in COURSE_MAP]].sum()
        print(f"  part {PART_NAMES[part]:<28} {part_min:>4} min = {part_min / 60:.2f} h")
    week_hours = []
    for week, ids in enumerate(EIGHT_WEEK_PLAN, start=1):
        m = sum(minutes[i] for i in ids)
        week_hours.append(m / 60)
        print(f"  week {week}: notebooks {', '.join(f'{i:02d}' for i in ids):<12} {m:>4} min = {m / 60:.2f} h")
    print(f"  lightest week {min(week_hours):.2f} h, heaviest week {max(week_hours):.2f} h")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    colors = [SERIES[p - 1] for _, _, _, p in COURSE_MAP]
    labels = [f"{i:02d}" for i, _, _, _ in COURSE_MAP]
    bar(ax1, labels, minutes, color=colors)
    ax1.set_xlabel("notebook"); ax1.set_ylabel("estimated time (minutes)")
    ax1.set_title("Time budget per notebook, coloured by part")
    ax1.legend(handles=[Patch(color=SERIES[p - 1], label=PART_NAMES[p]) for p in range(1, 8)],
               fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax1.set_ylim(0, 230)
    bar(ax2, [f"wk {w}" for w in range(1, 9)], week_hours, color=BLUE, label_fmt="{:.2f}")
    ax2.set_ylabel("hours of guided material"); ax2.set_title("Eight-week plan: load per week")
    ax2.set_ylim(0, 6.5)
    save(fig, NB, "course_time_budget")


# ----------------------------------------------------------------------------------------------
# Section 3: matrix multiplication timing on the CPU (float32 vs float64)
# ----------------------------------------------------------------------------------------------
def time_matmul(n, dtype, repeats=5):
    r = np.random.default_rng(0)
    a = r.standard_normal((n, n)).astype(dtype)
    b = r.standard_normal((n, n)).astype(dtype)
    _ = a @ b                                 # warm-up (not timed)
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    return (time.perf_counter() - t0) / repeats


def fig_matmul_timing():
    sizes = [64, 128, 256, 512, 1024, 2048]
    print("\n[fig_matmul_timing] numpy matmul on this CPU (seconds per n x n @ n x n, mean of 5):")
    print(f"  {'n':>5} {'2n^3 (GFLOP)':>13} {'float32 s':>11} {'float64 s':>11} {'GFLOP/s f32':>12} {'GFLOP/s f64':>12} {'f64/f32':>8}")
    t32, t64 = [], []
    for n in sizes:
        s32, s64 = time_matmul(n, np.float32), time_matmul(n, np.float64)
        t32.append(s32); t64.append(s64)
        flops = 2 * n ** 3
        print(f"  {n:>5} {flops / 1e9:>13.4f} {s32:>11.6f} {s64:>11.6f} {flops / s32 / 1e9:>12.1f} {flops / s64 / 1e9:>12.1f} {s64 / s32:>8.2f}")
    ratio_1024_512 = t32[sizes.index(1024)] / t32[sizes.index(512)]
    print(f"  float32 time ratio n=1024 / n=512: {ratio_1024_512:.2f} (ideal 8.00)")
    t32, t64 = np.array(t32), np.array(t64)
    g32, g64 = [2 * n ** 3 / s / 1e9 for n, s in zip(sizes, t32)], [2 * n ** 3 / s / 1e9 for n, s in zip(sizes, t64)]

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_t.loglog(sizes, t32, "o-", color=BLUE, label="float32")
    ax_t.loglog(sizes, t64, "s-", color=ORANGE, label="float64")
    ref = t32[2] * (np.array(sizes) / sizes[2]) ** 3
    ax_t.loglog(sizes, ref, "--", color=MUTED, linewidth=1, label="slope 3 reference ($\\propto n^3$)")
    ax_t.set_xlabel("matrix side $n$"); ax_t.set_ylabel("seconds per matmul")
    ax_t.set_title("Time of one matrix multiplication (log-log)"); ax_t.legend()
    ax_f.semilogx(sizes, g32, "o-", color=BLUE, label="float32")
    ax_f.semilogx(sizes, g64, "s-", color=ORANGE, label="float64")
    ax_f.set_xlabel("matrix side $n$"); ax_f.set_ylabel("achieved GFLOP/s")
    ax_f.set_title("Achieved throughput on the CPU"); ax_f.legend()
    ax_f.annotate(f"{g32[-1]:.0f} GFLOP/s", (sizes[-1], g32[-1]), xytext=(-70, -14), textcoords="offset points",
                  fontsize=8, color=INK_2)
    save(fig, NB, "matmul_time_and_throughput")


# ----------------------------------------------------------------------------------------------
# Section 4: the make_moons problem, a forward pass by hand, gradient descent, the trained MLP
# ----------------------------------------------------------------------------------------------
def fig_moons_dataset():
    print("\n[fig_moons_dataset] make_moons(n_samples=1000, noise=0.2, random_state=42): first 6 rows")
    print(f"  {'row':>3} {'x1':>8} {'x2':>8} {'label':>5}")
    for i in range(6):
        print(f"  {i:>3} {X_TR[i, 0]:>8.3f} {X_TR[i, 1]:>8.3f} {int(Y_TR[i, 0]):>5}")
    print(f"  class counts: class 0 = {int((Y_TR == 0).sum())}, class 1 = {int((Y_TR == 1).sum())}")
    print(f"  x1 range [{X_TR[:, 0].min():.2f}, {X_TR[:, 0].max():.2f}], x2 range [{X_TR[:, 1].min():.2f}, {X_TR[:, 1].max():.2f}]")
    # a straight-line classifier for comparison: the best of many random lines through the data
    r = np.random.default_rng(1)
    best = 0.0
    for _ in range(4000):
        w = r.standard_normal(2); b = r.uniform(-1.5, 1.5)
        acc = np.mean(((X_TR @ w + b) > 0) == (Y_TR[:, 0] > 0.5))
        best = max(best, acc, 1 - acc)
    print(f"  best straight-line classifier found by random search: train accuracy {best:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    m0, m1 = Y_TR[:, 0] == 0, Y_TR[:, 0] == 1
    ax.scatter(X_TR[m0, 0], X_TR[m0, 1], s=14, color=BLUE, label="class 0", edgecolors=SURFACE, linewidths=0.4)
    ax.scatter(X_TR[m1, 0], X_TR[m1, 1], s=14, color=ORANGE, label="class 1", edgecolors=SURFACE, linewidths=0.4)
    ax.annotate("rows 0 to 5 of the sample table", (X_TR[:6, 0].mean(), X_TR[:6, 1].mean()),
                xytext=(0.1, 1.55), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.scatter(X_TR[:6, 0], X_TR[:6, 1], s=70, facecolors="none", edgecolors=INK, linewidths=0.8)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("The two-moons training set (1000 points, noise 0.2)")
    ax.legend(loc="upper right")
    save(fig, NB, "moons_dataset")


def fig_forward_pass_by_hand():
    """A 2 -> 2 -> 1 network with hand-typed weights on one point, every number printed."""
    x = np.array([1.0, 0.5]); y = 1.0
    W1 = np.array([[1.0, -1.0], [0.5, 1.0]]); b1 = np.array([0.0, -0.5])
    W2 = np.array([[2.0, -1.0]]); b2 = np.array([0.1])
    eta = 0.1
    print("\n[fig_forward_pass_by_hand] one forward/backward/update step of a 2 -> 2 -> 1 ReLU network")
    print(f"  input x = {x.tolist()}, target y = {y:g}, learning rate eta = {eta}")
    print(f"  W1 = {W1.tolist()}, b1 = {b1.tolist()}, W2 = {W2.tolist()}, b2 = {b2.tolist()}")
    z1 = W1 @ x + b1; h = np.maximum(z1, 0)
    logit = float((W2 @ h + b2)[0]); p = float(sigmoid(logit)); loss = -np.log(p) if y == 1 else -np.log(1 - p)
    print(f"  z1 = W1 x + b1 = {z1.tolist()} | h = ReLU(z1) = {h.tolist()}")
    print(f"  logit = W2 h + b2 = {logit:.4f} | p = sigmoid(logit) = {p:.4f} | loss = -log p = {loss:.4f}")
    g_logit = p - y
    gW2 = g_logit * h[None, :]; gb2 = np.array([g_logit])
    g_h = g_logit * W2[0]; g_z1 = g_h * (z1 > 0)
    gW1 = np.outer(g_z1, x); gb1 = g_z1
    print(f"  dloss/dlogit = p - y = {g_logit:.4f}")
    print(f"  dloss/dW2 = (p - y) * h = {np.round(gW2, 4).tolist()} | dloss/db2 = {gb2.round(4).tolist()}")
    print(f"  dloss/dh = (p - y) * W2 = {np.round(g_h, 4).tolist()} | dloss/dz1 (ReLU mask {(z1 > 0).astype(int).tolist()}) = {np.round(g_z1, 4).tolist()}")
    print(f"  dloss/dW1 = outer(dloss/dz1, x) = {np.round(gW1, 4).tolist()} | dloss/db1 = {np.round(gb1, 4).tolist()}")
    W1n, b1n, W2n, b2n = W1 - eta * gW1, b1 - eta * gb1, W2 - eta * gW2, b2 - eta * gb2
    z1n = W1n @ x + b1n; hn = np.maximum(z1n, 0); logit_n = float((W2n @ hn + b2n)[0]); p_n = float(sigmoid(logit_n))
    print(f"  after one gradient step: W2 = {np.round(W2n, 4).tolist()}, b2 = {b2n.round(4).tolist()}, W1 = {np.round(W1n, 4).tolist()}, b1 = {np.round(b1n, 4).tolist()}")
    print(f"  new logit = {logit_n:.4f} | new p = {p_n:.4f} | new loss = {-np.log(p_n):.4f} (was {loss:.4f})")

    # gradient descent on a one-dimensional bowl J(theta) = (theta - 3)^2 for three learning rates
    theta0, steps = 0.0, 8
    lrs = [0.1, 0.8, 1.05]
    labels = ["$\\eta$ = 0.1 (slow)", "$\\eta$ = 0.8 (overshoots, converges)", "$\\eta$ = 1.05 (diverges)"]
    print(f"  gradient descent on J(theta) = (theta - 3)^2 from theta = {theta0}, gradient 2(theta - 3), {steps} steps:")
    paths = []
    for lr in lrs:
        th = [theta0]
        for _ in range(steps):
            th.append(th[-1] - lr * 2 * (th[-1] - 3))
        paths.append(np.array(th))
        print(f"    eta = {lr:<5} theta: " + ", ".join(f"{t:.3f}" for t in th[:6]) + f", ... final J = {(th[-1] - 3) ** 2:.4g}")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    grid = np.linspace(-3, 9, 300)
    for ax, lr, path, label, color in zip(axes, lrs, paths, labels, SERIES):
        ax.plot(grid, (grid - 3) ** 2, color=MUTED, linewidth=1.2)
        pts = path[np.abs(path - 3) <= 6.5]
        ax.plot(pts, (pts - 3) ** 2, "o-", color=color, markersize=6, linewidth=1.2)
        for k in range(min(3, len(pts))):
            ax.annotate(str(k), (pts[k], (pts[k] - 3) ** 2), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
        ax.set_title(label, fontsize=10); ax.set_xlabel("$\\theta$"); ax.set_ylim(-1, 40); ax.set_xlim(-3, 9)
    axes[0].set_ylabel("$J(\\theta) = (\\theta - 3)^2$")
    fig.suptitle("The update $\\theta \\leftarrow \\theta - \\eta \\nabla J$ on a one-dimensional bowl", fontsize=12, fontweight="bold")
    save(fig, NB, "gradient_descent_bowl")


def fig_decision_boundary():
    run = main_run(); params = run["params"]
    tr, te = accuracy(params, X_TR, Y_TR), accuracy(params, X_TE, Y_TE)
    print("\n[fig_decision_boundary] numpy MLP 2 -> 32 -> 32 -> 1, Adam lr 1e-2, 1000 full-batch epochs:")
    print(f"  parameters: {count_parameters(params)} | final train loss {run['losses'][-1]:.4f} | final test loss {run['test_losses'][-1]:.4f}")
    print(f"  train accuracy {tr:.3f} | test accuracy {te:.3f} | gap {tr - te:+.3f}")
    logits, _ = forward(params, X_TR[:6])
    print("  predictions for the 6 sample rows (logit, P(class 1), predicted class, true label):")
    for i, (z, yy) in enumerate(zip(logits[:, 0], Y_TR[:6, 0])):
        print(f"    row {i}: logit {z:+8.3f}  P {sigmoid(z):.3f}  pred {int(z > 0)}  true {int(yy)}")
    far = np.array([[3.0, 2.0], [-2.0, -1.5]])
    z_far, _ = forward(params, far)
    print("  far from the data (over-confidence): " + "; ".join(f"x={pt.tolist()} -> P {sigmoid(z):.4f}" for pt, z in zip(far, z_far[:, 0])))

    res = 300
    x_min, x_max = X_TR[:, 0].min() - 0.5, X_TR[:, 0].max() + 0.5
    y_min, y_max = X_TR[:, 1].min() - 0.5, X_TR[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    zz, _ = forward(params, np.c_[xx.ravel(), yy.ravel()])
    probs = sigmoid(zz).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    # P(class 1) - 0.5 is a signed quantity centred on zero (the boundary), hence the diverging map
    cf = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap=DIV_CMAP, alpha=0.8)
    ax.contour(xx, yy, probs, levels=[0.5], colors=INK, linewidths=1)
    m0, m1 = Y_TR[:, 0] == 0, Y_TR[:, 0] == 1
    ax.scatter(X_TR[m0, 0], X_TR[m0, 1], s=12, color=BLUE, edgecolors=SURFACE, linewidths=0.4, label="class 0")
    ax.scatter(X_TR[m1, 0], X_TR[m1, 1], s=12, color=ORANGE, edgecolors=SURFACE, linewidths=0.4, label="class 1")
    cb = fig.colorbar(cf, ax=ax); cb.set_label("predicted P(class 1)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
    ax.set_title(f"MLP decision boundary: train {tr:.3f} / test {te:.3f}")
    ax.legend(loc="upper right")
    save(fig, NB, "moons_decision_boundary")


def fig_loss_curve():
    run = main_run(); losses, test_losses = run["losses"], run["test_losses"]
    checkpoints = [1, 10, 50, 100, 200, 500, 1000]
    print("\n[fig_loss_curve] loss per epoch of the main run (train / test):")
    for e in checkpoints:
        print(f"  epoch {e:>5}: train loss {losses[e - 1]:.4f} | test loss {test_losses[e - 1]:.4f}")
    first_below = next(i + 1 for i, l in enumerate(losses) if l < 0.2)
    print(f"  first epoch with train loss below 0.2: {first_below} | lowest test loss {min(test_losses):.4f} at epoch {int(np.argmin(test_losses)) + 1}")
    fig, ax = plt.subplots()
    ep = np.arange(1, len(losses) + 1)
    ax.plot(ep, losses, color=BLUE, label="training loss")
    ax.plot(ep, test_losses, color=ORANGE, label="test loss (fresh 1000 points)")
    ax.annotate(f"epoch {first_below}: loss < 0.2", (first_below, losses[first_below - 1]), xytext=(40, 20),
                textcoords="offset points", fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("epoch (one full-batch step each)"); ax.set_ylabel("binary cross-entropy")
    ax.set_title("Loss curve of the tiny MLP on two moons"); ax.legend()
    ax.set_ylim(0, 0.75)
    save(fig, NB, "moons_loss_curve")


def fig_width_and_noise_sweeps():
    widths = [1, 2, 4, 16, 64, 256]
    print("\n[fig_width_and_noise_sweeps] width sweep (lr 1e-2, 1000 epochs, seed 42):")
    w_tr, w_te = [], []
    for w in widths:
        params, losses, _ = train_mlp(X_TR, Y_TR, w, 1e-2, 1000)
        tr, te = accuracy(params, X_TR, Y_TR), accuracy(params, X_TE, Y_TE)
        w_tr.append(tr); w_te.append(te)
        print(f"  width {w:>4} | params {count_parameters(params):>6} | final loss {losses[-1]:.4f} | train acc {tr:.3f} | test acc {te:.3f}")
    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("  noise sweep (width 32, lr 1e-2, 1000 epochs; train seed 42, test seed 43):")
    n_tr, n_te = [], []
    for nz in noises:
        Xa, ya = moons(nz, SEED); Xb, yb = moons(nz, SEED + 1)
        params, _, _ = train_mlp(Xa, ya, 32, 1e-2, 1000)
        tr, te = accuracy(params, Xa, ya), accuracy(params, Xb, yb)
        n_tr.append(tr); n_te.append(te)
        print(f"  noise {nz:.1f} | train acc {tr:.3f} | test acc {te:.3f} | gap {tr - te:+.3f}")

    fig, (ax_w, ax_n) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_w.plot(widths, w_tr, "o-", color=BLUE, label="train accuracy")
    ax_w.plot(widths, w_te, "s-", color=ORANGE, label="test accuracy")
    ax_w.set_xscale("log", base=2); ax_w.set_xticks(widths); ax_w.set_xticklabels([str(w) for w in widths])
    ax_w.set_xlabel("hidden width (units per hidden layer)"); ax_w.set_ylabel("accuracy")
    ax_w.set_title("Exercise 2: accuracy versus model width"); ax_w.set_ylim(0.5, 1.02); ax_w.legend(loc="lower right")
    ax_n.plot(noises, n_tr, "o-", color=BLUE, label="train accuracy")
    ax_n.plot(noises, n_te, "s-", color=ORANGE, label="test accuracy")
    ax_n.set_xlabel("noise level of make_moons"); ax_n.set_ylabel("accuracy")
    ax_n.set_title("Exercise 4: generalisation gap versus noise"); ax_n.set_ylim(0.5, 1.02); ax_n.legend(loc="lower left")
    k = noises.index(0.5)
    ax_n.annotate(f"gap {n_tr[k] - n_te[k]:+.3f}", (noises[k], (n_tr[k] + n_te[k]) / 2), xytext=(-60, 0),
                  textcoords="offset points", fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    save(fig, NB, "width_and_noise_sweeps")


# ----------------------------------------------------------------------------------------------
# Section 5: the experiment log and the learning-rate sweep
# ----------------------------------------------------------------------------------------------
def fig_hyperparameter_grid():
    print("\n[fig_hyperparameter_grid] experiment log: hidden in {4, 32} x lr in {1e-3, 1e-2}, 300 epochs:")
    print(f"  {'hidden':>6} {'lr':>6} {'epochs':>6} {'final_loss':>10} {'train_acc':>9} {'test_acc':>8}")
    fig, ax = plt.subplots()
    k = 0
    for hidden in [4, 32]:
        for lr in [1e-3, 1e-2]:
            params, losses, _ = train_mlp(X_TR, Y_TR, hidden, lr, 300)
            tr, te = accuracy(params, X_TR, Y_TR), accuracy(params, X_TE, Y_TE)
            print(f"  {hidden:>6} {lr:>6g} {len(losses):>6} {losses[-1]:>10.4f} {tr:>9.3f} {te:>8.3f}")
            ax.plot(np.arange(1, 301), losses, color=SERIES[k], label=f"hidden {hidden}, lr {lr:g}",
                    linestyle="-" if hidden == 32 else "--")
            k += 1
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss (binary cross-entropy)")
    ax.set_title("Four logged runs: width and learning rate"); ax.legend(); ax.set_ylim(0, 0.75)
    save(fig, NB, "hyperparameter_grid_loss_curves")


def fig_lr_sweep():
    lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print("\n[fig_lr_sweep] learning-rate sweep (width 32, 1000 epochs):")
    fig, ax = plt.subplots()
    for k, lr in enumerate(lrs):
        params, losses, _ = train_mlp(X_TR, Y_TR, 32, lr, 1000)
        te = accuracy(params, X_TE, Y_TE)
        jag = float(np.mean(np.diff(losses[500:]) > 0))     # share of epochs where the loss went UP
        print(f"  lr {lr:<6g} | loss at epoch 100 {losses[99]:.4f} | final loss {losses[-1]:.4f} | min loss {min(losses):.4f} | "
              f"max loss {max(losses):.3g} | share of up-steps in epochs 500-1000 {jag:.2f} | test acc {te:.3f}")
        ax.plot(np.arange(1, 1001), losses, color=SERIES[k], linewidth=1.4, label=f"lr = {lr:g} (test acc {te:.3f})")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss (log scale)")
    ax.set_title("Exercise 3: learning-rate sweep for the tiny MLP"); ax.legend(fontsize=8)
    save(fig, NB, "lr_sweep_loss_curves")


# ----------------------------------------------------------------------------------------------
# Section 6: shapes and sizes
# ----------------------------------------------------------------------------------------------
def fig_tensor_shapes():
    examples = [("tabular", (64, 784), "float32"), ("images", (64, 3, 32, 32), "float32"),
                ("sequence", (64, 50, 128), "float32"), ("tokens", (64, 50), "int64"),
                ("labels", (64,), "int64"), ("loss", (), "float32")]
    print("\n[fig_tensor_shapes] example tensors of the shape table:")
    print(f"  {'kind':<9} {'shape':<18} {'ndim':>4} {'numel':>8} dtype")
    numels = []
    for name, shape, dtype in examples:
        n = int(np.prod(shape)) if shape else 1
        numels.append(n)
        print(f"  {name:<9} {str(shape):<18} {len(shape):>4} {n:>8,} {dtype}")
    X = rng.standard_normal((64, 784)); W = rng.standard_normal((256, 784)) / np.sqrt(784); b = rng.standard_normal(256) * 0.01
    out = X @ W.T + b
    print(f"  batched linear layer: X {X.shape} @ W^T {W.T.shape} + b {b.shape} -> {out.shape}; "
          f"multiply-adds = 64*784*256 = {64 * 784 * 256:,}")
    print(f"  memory of the image batch in float32: {numels[1] * 4 / 1024:.0f} KiB; of the sequence batch: {numels[2] * 4 / 1024:.0f} KiB")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    bar(ax1, [e[0] for e in examples], numels, color=BLUE, label_fmt="{:,}")
    ax1.set_yscale("log"); ax1.set_ylim(0.5, 3e6)
    ax1.set_ylabel("number of elements (log scale)"); ax1.set_title("Elements per example tensor")
    ax1.set_xlabel("kind of data")
    # shape diagram of X @ W^T + b -> (batch, out)
    ax2.set_xlim(0, 16); ax2.set_ylim(0, 6); ax2.axis("off")
    ax2.set_title("The batched linear layer as matrix shapes")
    def box(x0, y0, w, h, color, text, sub):
        ax2.add_patch(Rectangle((x0, y0), w, h, facecolor=color, edgecolor=SURFACE, alpha=0.9))
        ax2.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9, color=SURFACE, fontweight="bold")
        ax2.text(x0 + w / 2, y0 - 0.35, sub, ha="center", va="top", fontsize=8, color=INK_2)
    box(0.4, 2.3, 4.2, 1.6, BLUE, "X", "(64, 784)\nbatch x features")
    ax2.text(5.0, 3.1, "@", ha="center", va="center", fontsize=14, color=INK)
    box(5.5, 1.0, 2.0, 4.2, ORANGE, "W$^\\top$", "(784, 256)\nin x out")
    ax2.text(8.0, 3.1, "+", ha="center", va="center", fontsize=14, color=INK)
    box(8.4, 2.9, 2.0, 0.4, AQUA, "", "b  (256,)\nbroadcast over rows")
    ax2.text(10.9, 3.1, "=", ha="center", va="center", fontsize=14, color=INK)
    box(11.4, 2.3, 2.0, 1.6, VIOLET, "H", "(64, 256)\nbatch x out")
    ax2.text(8, 5.7, "nn.Linear(784, 256).weight has shape (256, 784); the layer computes X @ weight.T + bias",
             ha="center", va="center", fontsize=8, color=INK_2)
    save(fig, NB, "tensor_shapes_and_linear_layer")


def main():
    fig_timeline()
    fig_time_budget()
    fig_matmul_timing()
    fig_moons_dataset()
    fig_forward_pass_by_hand()
    fig_decision_boundary()
    fig_loss_curve()
    fig_width_and_noise_sweeps()
    fig_hyperparameter_grid()
    fig_lr_sweep()
    fig_tensor_shapes()


if __name__ == "__main__":
    main()
