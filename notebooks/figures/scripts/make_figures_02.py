"""Figures and worked-example numbers for the companion guide of Notebook 02 (Neural Networks from Scratch).

Everything here is NumPy + matplotlib (+ the bundled sklearn datasets make_moons and load_digits; no
downloads). A tiny NumPy MLP library, mirroring the notebook's Section 6, produces real training curves and
decision boundaries. Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

from scipy.special import erf
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split

NB = "02"
SEED = 42
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------------------------
# A tiny NumPy MLP library (mirrors the notebook's Section 6)
# ----------------------------------------------------------------------------------------------
def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_softmax(Z):
    Zs = Z - Z.max(axis=1, keepdims=True)
    return Zs - np.log(np.sum(np.exp(Zs), axis=1, keepdims=True))


def softmax(Z):
    return np.exp(log_softmax(Z))


def cross_entropy(logits, y):
    return float(-np.mean(log_softmax(logits)[np.arange(len(y)), y]))


class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}

    def __call__(self, x):
        return self.forward(x)


class Linear(Layer):
    def __init__(self, fan_in, fan_out, rng, init="he", dtype=np.float32):
        super().__init__()
        std = {"he": np.sqrt(2.0 / fan_in), "xavier": np.sqrt(2.0 / (fan_in + fan_out)), "small": 0.01, "zero": 0.0}[init]
        W = (rng.normal(size=(fan_in, fan_out)) * std).astype(dtype)
        self.params = {"W": W, "b": np.zeros(fan_out, dtype=dtype)}
        self.grads = {"W": np.zeros_like(W), "b": np.zeros(fan_out, dtype=dtype)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        self.grads["W"] = self.x.T @ g
        self.grads["b"] = g.sum(axis=0)
        return g @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class Sigmoid(Layer):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, g):
        return g * self.out * (1.0 - self.out)


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.y, self.N = y, logits.shape[0]
        log_p = log_softmax(logits)
        self.P = np.exp(log_p)
        return float(-np.mean(log_p[np.arange(self.N), y]))

    def backward(self):
        G = self.P.copy()
        G[np.arange(self.N), self.y] -= 1.0
        return G / self.N


class MSE:
    def forward(self, y_hat, y):
        self.diff = y_hat - y
        return float(0.5 * np.mean(self.diff ** 2))

    def backward(self):
        return self.diff / self.diff.size


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def parameters(self):
        for layer in self.layers:
            for name, p in layer.params.items():
                yield layer, name, p

    def n_params(self):
        return sum(p.size for _, _, p in self.parameters())


class SGD:
    def __init__(self, model, lr):
        self.model, self.lr = model, lr

    def step(self):
        for layer, name, p in self.model.parameters():
            p -= self.lr * layer.grads[name]


def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def train_one_epoch(model, loss_fn, opt, X, y, batch_size, rng, shuffle=True):
    total, count = 0.0, 0
    for xb, yb in iterate_minibatches(X, y, batch_size, rng, shuffle):
        loss = loss_fn.forward(model(xb), yb)
        model.backward(loss_fn.backward())
        opt.step()
        total += loss * len(xb); count += len(xb)
    return total / count


def evaluate(model, loss_fn, X, y):
    logits = model(X)
    return loss_fn.forward(logits, y), float(np.mean(logits.argmax(axis=1) == y))


def fit(model, loss_fn, opt, X_tr, y_tr, X_val, y_val, epochs, batch_size, rng, shuffle=True):
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        hist["train_loss"].append(train_one_epoch(model, loss_fn, opt, X_tr, y_tr, batch_size, rng, shuffle))
        vl, va = evaluate(model, loss_fn, X_val, y_val)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
    return hist


def decision_regions(ax, predict, X, y, n_classes, padding=0.5, resolution=250, title=""):
    """Colour the plane by predicted class (palette order), overlay the points."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(X.dtype)
    zz = np.asarray(predict(grid)).reshape(xx.shape)
    colors = SERIES[:n_classes]
    ax.contourf(xx, yy, zz, levels=np.arange(n_classes + 1) - 0.5, colors=colors, alpha=0.18)
    for c in range(n_classes):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=16, color=colors[c], edgecolors="white", linewidths=0.4,
                   label=f"class {c}")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title); ax.grid(False)
    ax.legend(loc="upper right", markerscale=1.4)


# ----------------------------------------------------------------------------------------------
# Section 1: perceptron and XOR
# ----------------------------------------------------------------------------------------------
X_GATE = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
GATES = {"AND": np.array([0, 0, 0, 1.]), "OR": np.array([0, 1, 1, 1.]), "XOR": np.array([0, 1, 1, 0.])}


def step(z):
    return (z > 0).astype(float)


def train_perceptron(X, y, lr=0.1, epochs=50, seed=SEED):
    r = np.random.default_rng(seed)
    w, b = r.normal(0.0, 0.1, size=X.shape[1]), 0.0
    mistakes = []
    for _ in range(epochs):
        n_err = 0
        for x_i, y_i in zip(X, y):
            y_hat = step(x_i @ w + b)
            if y_hat != y_i:
                w += lr * (y_i - y_hat) * x_i; b += lr * (y_i - y_hat); n_err += 1
        mistakes.append(n_err)
        if n_err == 0:
            break
    return w, b, mistakes


def fig_perceptron_gates():
    section("Section 1 | perceptron on AND, OR, XOR (lr 0.1, up to 50 epochs, seed 42)")
    print("truth table (x1, x2 | AND OR XOR):")
    for i in range(4):
        print(f"  {X_GATE[i, 0]:.0f}, {X_GATE[i, 1]:.0f} | {GATES['AND'][i]:.0f}   {GATES['OR'][i]:.0f}   {GATES['XOR'][i]:.0f}")
    print("hand-designed AND neuron w=(1,1), b=-1.5 -> z for the four inputs:", X_GATE @ np.array([1., 1.]) - 1.5,
          "-> output", step(X_GATE @ np.array([1., 1.]) - 1.5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (name, y) in zip(axes, GATES.items()):
        w, b, mistakes = train_perceptron(X_GATE, y)
        acc = np.mean(step(X_GATE @ w + b) == y)
        print(f"{name}: epochs run = {len(mistakes)}, mistakes per epoch = {mistakes[:8]}{'...' if len(mistakes) > 8 else ''}, "
              f"learned w = ({w[0]:.3f}, {w[1]:.3f}), b = {b:.3f}, accuracy = {acc:.2f}")
        decision_regions(ax, lambda G, w=w, b=b: step(G @ w + b).astype(int), X_GATE, y.astype(int), 2,
                         padding=0.6, title=f"{name}: accuracy {acc:.2f} after {len(mistakes)} epochs")
    fig.suptitle("One perceptron learns AND and OR but never XOR", fontsize=12, fontweight="bold")
    save(fig, NB, "perceptron_gates")


def fig_xor_hidden_space():
    section("Section 1 | hand-designed two-layer XOR network")
    W1 = np.array([[1., 1.], [1., 1.]]); b1 = np.array([-0.5, -1.5]); W2 = np.array([1., -1.]); b2 = -0.5
    H = step(X_GATE @ W1 + b1); y_hat = step(H @ W2 + b2)
    print("x1 x2 | h1=OR h2=AND | h1-h2-0.5 | prediction | XOR target")
    for i in range(4):
        print(f" {X_GATE[i, 0]:.0f}  {X_GATE[i, 1]:.0f} |   {H[i, 0]:.0f}     {H[i, 1]:.0f}   |   {H[i] @ W2 + b2:+.1f}    |     {y_hat[i]:.0f}      |   {GATES['XOR'][i]:.0f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    y = GATES["XOR"].astype(int)
    for c in range(2):
        ax.scatter(X_GATE[y == c, 0], X_GATE[y == c, 1], s=90, color=SERIES[c], edgecolors="white", label=f"XOR = {c}")
    xs = np.linspace(-0.4, 1.4, 10)
    ax.plot(xs, 0.5 - xs, "--", color=INK_2, lw=1.2, label="$x_1 + x_2 = 0.5$ (unit $h_1$)")
    ax.plot(xs, 1.5 - xs, ":", color=INK_2, lw=1.4, label="$x_1 + x_2 = 1.5$ (unit $h_2$)")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("input space: XOR needs two lines"); ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    jitter = np.array([0, 0.03, -0.03, 0])
    for c in range(2):
        ax.scatter(H[y == c, 0] + jitter[y == c], H[y == c, 1], s=90, color=SERIES[c], edgecolors="white", label=f"XOR = {c}")
    ax.plot([-0.2, 1.2], [-0.7, 0.7], "--", color=INK_2, lw=1.2, label="$h_1 - h_2 = 0.5$")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_xlabel("$h_1$ = OR"); ax.set_ylabel("$h_2$ = AND")
    ax.set_title("hidden space: one line is enough"); ax.legend(loc="upper left", fontsize=8)
    ax.annotate("(0,1) and (1,0) land here", (1.0, 0.0), xytext=(0.35, 0.45), fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
    save(fig, NB, "xor_hidden_space")


# ----------------------------------------------------------------------------------------------
# Section 2: activations
# ----------------------------------------------------------------------------------------------
def gelu(z):
    return z * 0.5 * (1 + erf(z / np.sqrt(2)))


def d_gelu(z):
    return 0.5 * (1 + erf(z / np.sqrt(2))) + z * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


ACTS = {
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "ReLU": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky ReLU (0.1)": (lambda z: np.where(z > 0, z, 0.1 * z), lambda z: np.where(z > 0, 1.0, 0.1)),
    "GELU": (gelu, d_gelu),
}


def fig_activations():
    section("Section 2 | activation table and linear-collapse check")
    zs = np.array([-4., -1., 0., 1., 4.])
    print("z:             ", zs)
    for name, (f, df) in ACTS.items():
        print(f"{name:17s} phi(z) = {f(zs)}   phi'(z) = {df(zs)}")
    zg = np.linspace(-3, 0, 300001); g = gelu(zg)
    print(f"GELU minimum = {g.min():.4f} at z = {zg[g.argmin()]:.3f}")
    print(f"sigmoid slope at z = 5: {ACTS['sigmoid'][1](np.array(5.0)):.5f}; maximum slope (z = 0): {ACTS['sigmoid'][1](np.array(0.0)):.2f}")
    r = np.random.default_rng(SEED)
    W1, b1 = r.normal(size=(3, 4)), r.normal(size=4); W2, b2 = r.normal(size=(4, 2)), r.normal(size=2)
    X = r.normal(size=(5, 3))
    print("two stacked linear layers == one linear layer with W1 @ W2:", np.allclose((X @ W1 + b1) @ W2 + b2, X @ (W1 @ W2) + (b1 @ W2 + b2)))
    z = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, (f, df)) in enumerate(ACTS.items()):
        axes[0].plot(z, f(z), color=SERIES[i], lw=1.6, label=name)
        axes[1].plot(z, df(z), color=SERIES[i], lw=1.6, label=name)
    axes[0].set_ylim(-1.5, 3); axes[0].set_title("activation functions"); axes[0].set_xlabel("pre-activation $z$"); axes[0].set_ylabel("$\\phi(z)$")
    axes[1].set_ylim(-0.2, 1.3); axes[1].set_title("their derivatives"); axes[1].set_xlabel("pre-activation $z$"); axes[1].set_ylabel("$\\phi'(z)$")
    axes[1].annotate("sigmoid never exceeds 0.25", (0, 0.25), xytext=(1.2, 0.55), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
    axes[0].legend(loc="upper left", fontsize=8); axes[1].legend(loc="upper left", fontsize=8)
    save(fig, NB, "activations")


def fig_dead_relu():
    section("Section 2 | dead ReLU simulation (1000 inputs x 50 features, 100 He-initialised units, seed 42)")
    r = np.random.default_rng(SEED)
    X = r.normal(size=(1000, 50)); W = r.normal(0.0, np.sqrt(2.0 / 50), size=(50, 100))
    shifts = [0.0, -2.0, -4.0, -6.0]; fracs = []
    for s in shifts:
        A = np.maximum(0, X @ W + s)
        fracs.append(np.mean(np.all(A == 0, axis=0)))
        print(f"bias shift {s:+.0f}: dead units = {fracs[-1]:.0%}  (mean active fraction per unit {np.mean(A > 0):.3f})")
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, [f"{s:+.0f}" for s in shifts], [100 * f for f in fracs], color=BLUE, label_fmt="{:.0f}%")
    ax.set_xlabel("bias shift added to every unit"); ax.set_ylabel("dead units (% of the layer)")
    ax.set_title("A large negative bias shift kills ReLU units"); ax.set_ylim(0, 110)
    save(fig, NB, "dead_relu")


# ----------------------------------------------------------------------------------------------
# Sections 3 and 5: a hand-typed 2-3-2 network, forward, backward, gradient check
# ----------------------------------------------------------------------------------------------
TINY = {
    "X": np.array([[1.0, 2.0], [-1.0, 0.5]]),
    "y": np.array([0, 1]),
    "W1": np.array([[0.5, -0.3, 0.8], [0.1, 0.4, -0.6]]), "b1": np.array([0.0, 0.1, -0.2]),
    "W2": np.array([[0.7, -0.5], [0.2, 0.9], [-0.4, 0.3]]), "b2": np.array([0.05, -0.05]),
}


def tiny_forward(p, X):
    Z1 = X @ p["W1"] + p["b1"]; A1 = np.maximum(0, Z1); Z2 = A1 @ p["W2"] + p["b2"]
    return Z2, {"X": X, "Z1": Z1, "A1": A1}


def tiny_backward(p, cache, logits, y, relu_mask=True):
    N = logits.shape[0]; P = softmax(logits)
    G_Z2 = P.copy(); G_Z2[np.arange(N), y] -= 1; G_Z2 /= N
    g = {"W2": cache["A1"].T @ G_Z2, "b2": G_Z2.sum(0)}
    G_A1 = G_Z2 @ p["W2"].T
    G_Z1 = G_A1 * (cache["Z1"] > 0) if relu_mask else G_A1
    g["W1"] = cache["X"].T @ G_Z1; g["b1"] = G_Z1.sum(0)
    return g, {"P": P, "G_Z2": G_Z2, "G_A1": G_A1, "G_Z1": G_Z1}


def tiny_numeric_grads(p, eps=1e-6):
    num = {}
    for name in ["W1", "b1", "W2", "b2"]:
        arr = p[name]; g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            o = arr[idx]
            arr[idx] = o + eps; lp = cross_entropy(tiny_forward(p, p["X"])[0], p["y"])
            arr[idx] = o - eps; lm = cross_entropy(tiny_forward(p, p["X"])[0], p["y"])
            arr[idx] = o; g[idx] = (lp - lm) / (2 * eps)
        num[name] = g
    return num


def worked_forward_pass():
    section("Section 3 | forward pass of the hand-typed 2-3-2 network (2 examples)")
    p = TINY
    print("X (2 x 2):\n", p["X"]); print("y:", p["y"])
    print("W1 (2 x 3):\n", p["W1"]); print("b1:", p["b1"]); print("W2 (3 x 2):\n", p["W2"]); print("b2:", p["b2"])
    logits, c = tiny_forward(p, p["X"])
    print("Z1 = X @ W1 + b1 (2 x 3):\n", c["Z1"]); print("A1 = relu(Z1):\n", c["A1"])
    print("Z2 = A1 @ W2 + b2 = logits (2 x 2):\n", logits)
    P = softmax(logits); print("P = softmax(Z2):\n", P)
    print("per-example loss -log p_y:", -np.log(P[np.arange(2), p["y"]]), " mean loss =", f"{cross_entropy(logits, p['y']):.4f}")
    n = sum(p[k].size for k in ["W1", "b1", "W2", "b2"])
    print(f"parameter count: (2+1)*3 + (3+1)*2 = {n}")
    print("MNIST network 784-256-128-10 per layer:", [(d0 + 1) * d1 for d0, d1 in zip([784, 256, 128], [256, 128, 10])],
          "total", sum((d0 + 1) * d1 for d0, d1 in zip([784, 256, 128], [256, 128, 10])))


def fig_param_count():
    section("Section 3 | parameter count per layer")
    sizes = [784, 256, 128, 10]
    counts = [(a + 1) * b for a, b in zip(sizes[:-1], sizes[1:])]
    total = sum(counts)
    for i, c in enumerate(counts, 1):
        print(f"layer {i} ({sizes[i-1]} -> {sizes[i]}): {c:,} parameters = {c / total:.1%}")
    print(f"total = {total:,}")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bar(ax, [f"layer {i}\n{sizes[i-1]}\u2192{sizes[i]}" for i in range(1, 4)], counts, color=BLUE, label_fmt="{:,.0f}")
    ax.set_ylabel("parameters (weights + biases)"); ax.set_title("784-256-128-10 MLP: the first layer holds 85% of the parameters")
    ax.set_ylim(0, 230_000)
    save(fig, NB, "param_count")


# ----------------------------------------------------------------------------------------------
# Section 4: losses
# ----------------------------------------------------------------------------------------------
def fig_loss_vs_accuracy():
    section("Section 4 | losses on hand-typed data")
    Yh = np.array([[1.5, 2.0], [0.0, -1.0]]); Y = np.array([[1.0, 2.5], [0.5, -1.0]])
    print("MSE example: Y_hat =", Yh.tolist(), " Y =", Y.tolist())
    print("  squared errors:", ((Yh - Y) ** 2).tolist(), " loss = 0.5 * mean =", f"{0.5 * np.mean((Yh - Y) ** 2):.5f}",
          " gradient (Y_hat - Y)/4 =", ((Yh - Y) / 4).tolist())
    z = np.array([[2.0, 1.0, 0.1]]); P = softmax(z)
    print("softmax example: logits", z[0], "-> probabilities", P[0], " sum =", P.sum())
    print("  cross-entropy if true class is 0:", f"{cross_entropy(z, np.array([0])):.4f}", " if true class is 2:", f"{cross_entropy(z, np.array([2])):.4f}")
    big = np.array([[1000.0, 0.0, -1000.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print("big logits [1000, 0, -1000]: naive softmax =", naive[0], " stable softmax =", softmax(big)[0],
          " CE true class 2 =", f"{cross_entropy(big, np.array([2])):.1f}")
    x = np.array([-2.0, -1.2, -0.5, -0.1, 0.3, 0.8, 1.5, 2.2]); y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    print("1-D classifier data x:", x, " y:", y, " (two noisy points: -0.1 -> 1, 0.3 -> 0)")
    w_grid = np.linspace(-1, 4, 501)
    ce, acc = [], []
    for w in w_grid:
        logits = np.stack([np.zeros_like(x), w * x], axis=1)
        ce.append(cross_entropy(logits, y)); acc.append(np.mean(logits.argmax(1) == y))
    ce, acc = np.array(ce), np.array(acc)
    print("w:      ", [0.0, 0.5, 1.0, 2.0, 4.0])
    print("CE(w):  ", [f"{ce[np.argmin(np.abs(w_grid - w))]:.4f}" for w in [0.0, 0.5, 1.0, 2.0, 4.0]])
    print("acc(w): ", [f"{acc[np.argmin(np.abs(w_grid - w))]:.3f}" for w in [0.0, 0.5, 1.0, 2.0, 4.0]])
    print(f"CE minimum {ce.min():.4f} at w = {w_grid[ce.argmin()]:.2f}; accuracy is {acc.max():.3f} for every w > 0")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(w_grid, ce, color=BLUE); axes[0].set_xlabel("weight $w$"); axes[0].set_ylabel("cross-entropy (nats)")
    axes[0].set_title("cross-entropy: smooth, with a slope to follow")
    axes[0].annotate(f"minimum {ce.min():.3f} at w = {w_grid[ce.argmin()]:.2f}", (w_grid[ce.argmin()], ce.min()),
                     xytext=(1.6, 1.0), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
    axes[1].step(w_grid, acc, where="post", color=BLUE); axes[1].set_xlabel("weight $w$"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05); axes[1].set_title("accuracy: flat steps, zero gradient")
    save(fig, NB, "loss_vs_accuracy")


# ----------------------------------------------------------------------------------------------
# Section 5: backward pass and gradient check on the tiny network
# ----------------------------------------------------------------------------------------------
def fig_gradient_check():
    section("Section 5 | backward pass of the hand-typed 2-3-2 network")
    p = {k: v.copy() for k, v in TINY.items()}
    logits, c = tiny_forward(p, p["X"])
    g, inter = tiny_backward(p, c, logits, p["y"])
    Y = np.eye(2)[p["y"]]
    print("P (softmax):\n", inter["P"]); print("one-hot Y:\n", Y)
    print("G_Z2 = (P - Y) / N:\n", inter["G_Z2"])
    print("G_W2 = A1^T @ G_Z2 (3 x 2):\n", g["W2"]); print("G_b2 = column sums of G_Z2:", g["b2"])
    print("G_A1 = G_Z2 @ W2^T (2 x 3):\n", inter["G_A1"]); print("ReLU mask (Z1 > 0):\n", (c["Z1"] > 0).astype(int))
    print("G_Z1 = G_A1 * mask:\n", inter["G_Z1"])
    print("G_W1 = X^T @ G_Z1 (2 x 3):\n", g["W1"]); print("G_b1 = column sums of G_Z1:", g["b1"])
    num = tiny_numeric_grads(p)
    g_bug, _ = tiny_backward(p, c, logits, p["y"], relu_mask=False)
    rel = lambda a, b: np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)
    names = ["W1", "b1", "W2", "b2"]
    err_ok = [rel(g[n], num[n]).max() for n in names]
    err_bug = [rel(g_bug[n], num[n]).max() for n in names]
    print("\ngradient check (central differences, eps = 1e-6), max relative error per parameter:")
    for n, e1, e2 in zip(names, err_ok, err_bug):
        print(f"  {n}: correct backward {e1:.2e} | missing ReLU mask {e2:.2e}")
    print("entry-by-entry for W2 (analytic | numeric | relative error):")
    for idx in np.ndindex(g["W2"].shape):
        print(f"  W2{idx}: {g['W2'][idx]:+.6f} | {num['W2'][idx]:+.6f} | {rel(g['W2'][idx], num['W2'][idx]):.1e}")
    print("entry-by-entry for W1 with the buggy backward (analytic | numeric | relative error):")
    for idx in np.ndindex(g["W1"].shape):
        print(f"  W1{idx}: {g_bug['W1'][idx]:+.6f} | {num['W1'][idx]:+.6f} | {rel(g_bug['W1'][idx], num['W1'][idx]):.1e}")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xpos = np.arange(len(names)); wdt = 0.36
    ax.bar(xpos - wdt / 2, np.maximum(err_ok, 1e-13), wdt, color=BLUE, label="correct backward pass")
    ax.bar(xpos + wdt / 2, np.maximum(err_bug, 1e-13), wdt, color=ORANGE, label="bug: ReLU mask forgotten")
    ax.axhline(1e-6, color=INK_2, lw=0.9, ls="--"); ax.text(3.45, 1.6e-6, "pass threshold $10^{-6}$", fontsize=8, color=INK_2, ha="right")
    ax.set_yscale("log"); ax.set_ylim(1e-12, 10); ax.set_xticks(xpos); ax.set_xticklabels(["$W_1$", "$b_1$", "$W_2$", "$b_2$"])
    ax.set_ylabel("max relative error vs finite differences"); ax.set_title("Gradient check: the planted bug fails only upstream of the mask")
    ax.legend(loc="upper right")
    save(fig, NB, "gradient_check")


# ----------------------------------------------------------------------------------------------
# Section 6: two moons and spiral
# ----------------------------------------------------------------------------------------------
def fig_two_moons():
    section("Section 6 | two moons, 2-32-32-2 ReLU MLP, SGD lr 0.1, batch 32, 300 epochs")
    X, y = make_moons(n_samples=600, noise=0.2, random_state=SEED); X = X.astype(np.float32)
    print("first 5 rows (x1, x2, label):")
    for i in range(5):
        print(f"  {X[i, 0]:+.3f}  {X[i, 1]:+.3f}  {y[i]}")
    Xtr, ytr, Xval, yval = X[:500], y[:500], X[500:], y[500:]
    r = np.random.default_rng(SEED)
    model = Sequential(Linear(2, 32, r), ReLU(), Linear(32, 32, r), ReLU(), Linear(32, 2, r))
    hist = fit(model, SoftmaxCrossEntropy(), SGD(model, lr=0.1), Xtr, ytr, Xval, yval, 300, 32, r)
    for e in [1, 10, 50, 100, 300]:
        print(f"  epoch {e:3d}: train loss {hist['train_loss'][e-1]:.4f} | val loss {hist['val_loss'][e-1]:.4f} | val acc {hist['val_acc'][e-1]:.3f}")
    print(f"parameters: {model.n_params()}")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ep = np.arange(1, 301)
    axes[0].plot(ep, hist["train_loss"], color=BLUE, lw=1.6, label="train loss")
    axes[0].plot(ep, hist["val_loss"], color=ORANGE, lw=1.6, label="validation loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title("training curves"); axes[0].legend()
    decision_regions(axes[1], lambda G: model(G).argmax(1), X, y, 2, padding=0.4,
                     title=f"decision boundary (val acc {hist['val_acc'][-1]:.2f})")
    save(fig, NB, "two_moons")


def make_spiral(n_per_class=200, n_classes=3, noise=0.2, seed=SEED):
    r = np.random.default_rng(seed); X, y = [], []
    for c in range(n_classes):
        radius = np.linspace(0.05, 1.0, n_per_class)
        angle = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + noise * r.normal(size=n_per_class)
        X.append(np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1)); y.append(np.full(n_per_class, c))
    X, y = np.concatenate(X).astype(np.float32), np.concatenate(y)
    order = r.permutation(len(X))
    return X[order], y[order]


def fig_spiral():
    section("Section 6 | three-class spiral, 2-64-64-3, ReLU vs sigmoid hidden units, SGD lr 0.1, batch 32, 600 epochs")
    X, y = make_spiral(); Xtr, ytr, Xval, yval = X[:480], y[:480], X[480:], y[480:]
    print("first 5 rows (x1, x2, class):")
    for i in range(5):
        print(f"  {X[i, 0]:+.3f}  {X[i, 1]:+.3f}  {y[i]}")
    results = {}
    for name, Act in [("ReLU", ReLU), ("sigmoid", Sigmoid)]:
        r = np.random.default_rng(SEED)
        model = Sequential(Linear(2, 64, r), Act(), Linear(64, 64, r), Act(), Linear(64, 3, r))
        hist = fit(model, SoftmaxCrossEntropy(), SGD(model, lr=0.1), Xtr, ytr, Xval, yval, 600, 32, r)
        results[name] = (model, hist)
        for e in [50, 100, 200, 400, 600]:
            print(f"  {name:8s} epoch {e:3d}: train loss {hist['train_loss'][e-1]:.4f} | val acc {hist['val_acc'][e-1]:.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ep = np.arange(1, 601)
    for i, (name, (m, h)) in enumerate(results.items()):
        axes[0].plot(ep, h["train_loss"], color=SERIES[i], lw=1.6, label=f"{name} hidden units")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training cross-entropy"); axes[0].set_title("same initial weights, two activations"); axes[0].legend()
    for ax, (name, (m, h)) in zip(axes[1:], results.items()):
        decision_regions(ax, lambda G, m=m: m(G).argmax(1), X, y, 3, padding=0.15, title=f"{name}: val acc {h['val_acc'][-1]:.2f}")
    save(fig, NB, "spiral_relu_vs_sigmoid")


# ----------------------------------------------------------------------------------------------
# Section 7: digits (sklearn load_digits as the offline stand-in for MNIST)
# ----------------------------------------------------------------------------------------------
def load_digits_split():
    d = load_digits(); X = (d.data / 16.0).astype(np.float32); y = d.target.astype(np.int64)
    Xtr_full, Xte, ytr_full, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    mean, std = Xtr_full.mean(), Xtr_full.std()
    Xtr_full = (Xtr_full - mean) / std; Xte = (Xte - mean) / std
    n_val = len(Xtr_full) // 5
    return (Xtr_full[:-n_val], ytr_full[:-n_val], Xtr_full[-n_val:], ytr_full[-n_val:], Xte, yte, mean, std)


def fig_digits_training():
    section("Section 7 | sklearn digits (8x8, offline stand-in for MNIST), 64-256-128-10 ReLU MLP, SGD lr 0.1, batch 64, 10 epochs")
    Xtr, ytr, Xval, yval, Xte, yte, mean, std = load_digits_split()
    print(f"train {Xtr.shape}, val {Xval.shape}, test {Xte.shape}; pixel mean {mean:.4f}, std {std:.4f} (computed on the training split)")
    print("first 8 training labels:", ytr[:8])
    print("first training image, standardised, row 0 (8 values):", Xtr[0, :8])
    r = np.random.default_rng(SEED)
    model = Sequential(Linear(64, 256, r), ReLU(), Linear(256, 128, r), ReLU(), Linear(128, 10, r))
    loss_fn = SoftmaxCrossEntropy()
    print(f"parameters: {model.n_params():,}")
    hist = fit(model, loss_fn, SGD(model, lr=0.1), Xtr, ytr, Xval, yval, 10, 64, r)
    for e in range(10):
        print(f"  epoch {e+1:2d}: train loss {hist['train_loss'][e]:.4f} | val loss {hist['val_loss'][e]:.4f} | val acc {hist['val_acc'][e]:.4f}")
    tl, ta = evaluate(model, loss_fn, Xte, yte)
    print(f"test loss {tl:.4f} | test accuracy {ta:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, 11)
    axes[0].plot(ep, hist["train_loss"], color=BLUE, marker="o", ms=6, lw=1.6, label="train loss")
    axes[0].plot(ep, hist["val_loss"], color=ORANGE, marker="o", ms=6, lw=1.6, label="validation loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title("digits: loss per epoch"); axes[0].legend()
    axes[1].plot(ep, hist["val_acc"], color=BLUE, marker="o", ms=6, lw=1.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy"); axes[1].set_ylim(0.9, 1.0)
    axes[1].set_title(f"digits: validation accuracy (test acc {ta:.3f})")
    save(fig, NB, "digits_training")
    return model, Xte, yte


def fig_confusion_matrix(model, Xte, yte):
    section("Section 7 | confusion matrix on the digits test split")
    pred = model(Xte).argmax(1)
    cm = np.zeros((10, 10), dtype=int); np.add.at(cm, (yte, pred), 1)
    per_class = cm.diagonal() / cm.sum(1)
    print("confusion matrix rows = true digit, columns = predicted digit:\n", cm)
    print("per-class accuracy:", " ".join(f"{d}:{a:.3f}" for d, a in enumerate(per_class)))
    off = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    print("largest confusions (count, true -> predicted):", off[:4])
    print(f"misclassified: {int((pred != yte).sum())} of {len(yte)}")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap=SEQ_CMAP)
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False)
    ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit"); ax.set_title("Confusion matrix (off-diagonal counts shown)")
    plt.colorbar(im, ax=ax, label="number of test images")
    save(fig, NB, "digits_confusion_matrix")


def fig_hyperparameters():
    section("Section 7 | learning rate, batch size, initialization (digits, 3 epochs each, same seed)")
    Xtr, ytr, Xval, yval, Xte, yte, _, _ = load_digits_split()

    def run(lr=0.1, batch_size=64, init="he", epochs=3):
        r = np.random.default_rng(SEED)
        m = Sequential(Linear(64, 256, r, init=init), ReLU(), Linear(256, 128, r, init=init), ReLU(), Linear(128, 10, r, init=init))
        lf, opt = SoftmaxCrossEntropy(), SGD(m, lr=lr)
        for _ in range(epochs):
            tl = train_one_epoch(m, lf, opt, Xtr, ytr, batch_size, r)
        return tl, evaluate(m, lf, Xval, yval)[1]

    groups = {"learning rate": [("0.01", dict(lr=0.01)), ("0.1", dict(lr=0.1)), ("1.0", dict(lr=1.0))],
              "batch size": [("8", dict(batch_size=8)), ("64", dict(batch_size=64)), ("512", dict(batch_size=512))],
              "initialization": [("zero", dict(init="zero")), ("small (0.01)", dict(init="small")), ("Xavier", dict(init="xavier")), ("He", dict(init="he"))]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (gname, settings) in zip(axes, groups.items()):
        labels, accs = [], []
        for label, kw in settings:
            tl, va = run(**kw)
            print(f"  {gname} = {label:12s}: final train loss {tl:.4f} | val acc {va:.4f}")
            labels.append(label); accs.append(va)
        bar(ax, labels, accs, color=BLUE, label_fmt="{:.3f}")
        ax.set_title(gname); ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("validation accuracy after 3 epochs")
    fig.suptitle("Three knobs, one network: digits after 3 epochs", fontsize=12, fontweight="bold")
    save(fig, NB, "hyperparameter_study")


# ----------------------------------------------------------------------------------------------
# Section 8: universal approximation and depth
# ----------------------------------------------------------------------------------------------
def fit_regression(x, y, width, epochs, lr, seed=SEED):
    r = np.random.default_rng(seed)
    model = Sequential(Linear(1, width, r), ReLU(), Linear(width, 1, r))
    hidden = model.layers[0]
    kinks = r.uniform(x.min(), x.max(), size=width).astype(np.float32)
    hidden.params["b"][:] = -hidden.params["W"][0] * kinks
    loss_fn, opt = MSE(), SGD(model, lr=lr)
    for _ in range(epochs):
        for xb, yb in iterate_minibatches(x, y, 32, r):
            loss_fn.forward(model(xb), yb); model.backward(loss_fn.backward()); opt.step()
    return model, loss_fn.forward(model(x), y)


def fig_universal_approximation():
    EPOCHS = 1500
    section(f"Section 8 | one-hidden-layer ReLU fits of sin(x) and step(x), 256 points on [-pi, pi], {EPOCHS} epochs, lr 0.2/width")
    x = np.linspace(-np.pi, np.pi, 256, dtype=np.float32).reshape(-1, 1)
    targets = {"sin(x)": np.sin(x), "step(x)": (x > 0).astype(np.float32)}
    widths = [2, 4, 16, 128]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for row, (tname, y) in enumerate(targets.items()):
        for ax, w in zip(axes[row], widths):
            model, mse = fit_regression(x, y, w, EPOCHS, lr=0.2 / w)
            pred = model(x).ravel()
            extra = ""
            if tname.startswith("step"):
                ramp = x.ravel()[(pred > 0.1) & (pred < 0.9)]
                extra = f", ramp width (0.1 to 0.9) = {ramp.max() - ramp.min():.3f}"
            print(f"  {tname} width {w:3d}: MSE {mse:.4f}{extra}")
            ax.plot(x, y, color=INK_2, ls="--", lw=1.2, label="target")
            ax.plot(x, pred, color=BLUE, lw=1.6, label="ReLU MLP")
            ax.set_title(f"{tname}, width {w}, MSE {mse:.4f}", fontsize=10); ax.set_xlabel("x")
            if row == 0 and w == 2:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("More hidden units, more kinks, finer polyline", fontsize=12, fontweight="bold")
    save(fig, NB, "universal_approximation")


def fig_depth_vs_width():
    section("Section 8 | depth versus width: the sawtooth built by composing a 2-unit tent")
    tent = lambda x: 2 * np.maximum(0, x) - 4 * np.maximum(0, x - 0.5)   # Linear(1,2) -> ReLU -> Linear(2,1)
    x = np.linspace(0, 1, 4097)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for L, ax in zip([1, 2, 3, 4], axes):
        f = x.copy()
        for _ in range(L):
            f = tent(f)
        slope = np.diff(f) / np.diff(x)
        kinks = int(np.sum(np.abs(np.diff(slope)) > 1e-6))
        print(f"  depth {L}: hidden units used = {2 * L}, kinks = {kinks}, teeth = {2 ** (L - 1)}, shallow network would need >= {kinks} units")
        ax.plot(x, f, color=BLUE, lw=1.4); ax.set_title(f"depth {L}: {2 * L} units, {kinks} kinks", fontsize=10); ax.set_xlabel("x")
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Composition: 2L ReLU units make a sawtooth with 2^L - 1 kinks", fontsize=12, fontweight="bold")
    save(fig, NB, "depth_vs_width")


def main():
    fig_perceptron_gates()
    fig_xor_hidden_space()
    fig_activations()
    fig_dead_relu()
    worked_forward_pass()
    fig_param_count()
    fig_loss_vs_accuracy()
    fig_gradient_check()
    fig_two_moons()
    fig_spiral()
    model, Xte, yte = fig_digits_training()
    fig_confusion_matrix(model, Xte, yte)
    fig_hyperparameters()
    fig_universal_approximation()
    fig_depth_vs_width()


if __name__ == "__main__":
    main()
