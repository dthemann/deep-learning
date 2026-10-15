"""Figures for the companion guide of notebook 04 (Training Deep Networks).

Every figure function builds its data with numpy, prints the numbers quoted in the guide
(clearly labelled) and saves a PNG under notebooks/figures/04/.  No torch, no downloads.
Run:  python3 notebooks/figures/scripts/make_figures_04.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import math
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

NB = "04"


def banner(title):
    print(f"\n### {title}")


# ----------------------------------------------------------------------------------------------
# Shared helpers: the optimizers of the notebook, written for numpy vectors
# ----------------------------------------------------------------------------------------------
class NpOptimizer:
    """Mirror of the notebook's optimizer classes for a list of numpy arrays."""
    def __init__(self, kind, lr, beta=0.9, rho=0.9, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.kind, self.lr, self.beta, self.rho = kind, lr, beta, rho
        self.b1, self.b2, self.eps, self.wd = b1, b2, eps, wd
        self.t, self.state = 0, {}

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            s = self.state.setdefault(i, {})
            k = self.kind
            if k == "SGD":
                p -= self.lr * g
            elif k in ("Momentum", "Nesterov"):
                v = s.setdefault("v", np.zeros_like(p))
                v *= self.beta; v += g
                p -= self.lr * (g + self.beta * v) if k == "Nesterov" else self.lr * v
            elif k == "AdaGrad":
                G = s.setdefault("G", np.zeros_like(p))
                G += g * g
                p -= self.lr * g / (np.sqrt(G) + self.eps)
            elif k == "RMSProp":
                sq = s.setdefault("s", np.zeros_like(p))
                sq *= self.rho; sq += (1 - self.rho) * g * g
                p -= self.lr * g / (np.sqrt(sq) + self.eps)
            elif k in ("Adam", "AdamW"):
                if k == "AdamW" and self.wd:
                    p *= 1 - self.lr * self.wd
                m = s.setdefault("m", np.zeros_like(p))
                sq = s.setdefault("s", np.zeros_like(p))
                m *= self.b1; m += (1 - self.b1) * g
                sq *= self.b2; sq += (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1 ** self.t)
                s_hat = sq / (1 - self.b2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


def run_2d(grad_fn, kind, lr, start, steps):
    theta = np.array(start, dtype=float)
    opt = NpOptimizer(kind, lr)
    path = [theta.copy()]
    for _ in range(steps):
        opt.step([theta], [grad_fn(theta)])
        if not np.all(np.isfinite(theta)) or np.abs(theta).max() > 1e6:
            break
        path.append(theta.copy())
    return np.array(path)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyMLP:
    """One-hidden-layer ReLU network with hand-written backprop (softmax cross-entropy)."""
    def __init__(self, d_in, d_h, d_out, rng, w_std=None):
        s1 = math.sqrt(2.0 / d_in) if w_std is None else w_std
        s2 = math.sqrt(1.0 / d_h) if w_std is None else w_std
        self.params = [rng.normal(0, s1, (d_in, d_h)), np.zeros(d_h), rng.normal(0, s2, (d_h, d_out)), np.zeros(d_out)]

    def forward(self, X):
        W1, b1, W2, b2 = self.params
        z1 = X @ W1 + b1
        h = np.maximum(z1, 0)
        return h @ W2 + b2, (X, z1, h)

    def backward(self, cache, dlogits):
        X, z1, h = cache
        W1, b1, W2, b2 = self.params
        dW2 = h.T @ dlogits; db2 = dlogits.sum(0)
        dh = dlogits @ W2.T
        dz1 = dh * (z1 > 0)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        return [dW1, db1, dW2, db2]


def ce_loss_and_grad(logits, y, double_softmax=False):
    """Mean cross-entropy and d loss / d logits.  double_softmax reproduces the notebook's bug."""
    n = len(y)
    onehot = np.zeros_like(logits); onehot[np.arange(n), y] = 1.0
    p1 = softmax(logits)
    if not double_softmax:
        loss = -np.log(p1[np.arange(n), y] + 1e-12).mean()
        return loss, (p1 - onehot) / n
    p2 = softmax(p1)
    loss = -np.log(p2[np.arange(n), y]).mean()
    dp1 = (p2 - onehot) / n
    dlogits = p1 * (dp1 - (dp1 * p1).sum(axis=1, keepdims=True))
    return loss, dlogits


# ----------------------------------------------------------------------------------------------
# Section 1: mini-batch gradient noise
# ----------------------------------------------------------------------------------------------
def fig_sgd_noise():
    banner("Section 1: mini-batch gradient noise (1-D linear regression y = 2x + noise)")
    r = np.random.default_rng(1)
    N = 256
    x = r.normal(0, 1, N)
    y = 2.0 * x + 0.5 * r.normal(0, 1, N)
    theta0 = 0.0
    g_i = (theta0 * x - y) * x                         # per-example gradient of 0.5 (theta x - y)^2
    print("sample data (first 6 examples): x, y, per-example gradient g_i at theta = 0")
    for i in range(6):
        print(f"  {i + 1}: x = {x[i]:6.3f}  y = {y[i]:6.3f}  g_i = {g_i[i]:7.3f}")
    print(f"full-batch gradient (mean of all {N}): {g_i.mean():.3f}   (true gradient of the population: -2.0)")
    print(f"mini-batch gradient from examples 1-4: {g_i[:4].mean():.3f}")
    print(f"std of a single-example gradient: {g_i.std():.3f}")
    sizes = [1, 4, 16, 64, 128]
    stds = []
    for b in sizes:
        est = [g_i[r.choice(N, b, replace=False)].mean() for _ in range(4000)]
        stds.append(np.std(est))
    print("std of the mini-batch gradient estimate vs batch size (batches drawn without replacement from the 256):")
    for b, s in zip(sizes, stds):
        print(f"  b = {b:3d}: std = {s:.3f}   (single-example std / sqrt(b) = {g_i.std() / math.sqrt(b):.3f})")

    # SGD runs with three batch sizes at the same learning rate, 6 epochs
    lr, epochs = 0.05, 6
    runs = {}
    for b in (4, 16, 64):
        rr = np.random.default_rng(2)
        th, losses = 0.0, []
        for ep in range(epochs):
            perm = rr.permutation(N)
            for i in range(0, N, b):
                idx = perm[i:i + b]
                losses.append(0.5 * np.mean((th * x[idx] - y[idx]) ** 2))
                th -= lr * np.mean((th * x[idx] - y[idx]) * x[idx])
        runs[b] = (np.array(losses), th)
        last = np.array(losses[-(N // b):])
        print(f"SGD lr={lr}, batch {b:2d}: {N // b:3d} steps/epoch, {len(losses)} steps, final theta = {th:.3f} (optimum 2.0), "
              f"last-epoch mini-batch losses: mean {last.mean():.3f}, std {last.std():.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(sizes, stds, "o-", color=SERIES[0], label="measured std of the estimate")
    ax1.loglog(sizes, g_i.std() / np.sqrt(sizes), "--", color=SERIES[1], label="single-example std / sqrt(b)")
    ax1.set_xlabel("batch size b"); ax1.set_ylabel("std of the mini-batch gradient")
    ax1.set_title("Gradient noise shrinks like 1/sqrt(b)"); ax1.legend()
    for k, b in enumerate((4, 16, 64)):
        losses, _ = runs[b]
        ax2.plot(np.arange(len(losses)) / (N // b), losses, lw=0.8, alpha=0.9, color=SERIES[k], label=f"batch {b}")
    ax2.set_ylim(0, 3); ax2.set_xlabel("epoch"); ax2.set_ylabel("mini-batch loss")
    ax2.set_title("Same learning rate, three batch sizes"); ax2.legend()
    save(fig, NB, "sgd_batch_noise")


# ----------------------------------------------------------------------------------------------
# Section 2: optimizers on two test functions, Adam by hand
# ----------------------------------------------------------------------------------------------
KAPPA = 20.0


def quad(x, y): return 0.5 * (x ** 2 + KAPPA * y ** 2)
def grad_quad(p): return np.array([p[0], KAPPA * p[1]])
def rosen(x, y): return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
def grad_rosen(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])


def fig_adam_by_hand():
    banner("Section 2: Adam's first three steps by hand (constant gradient g = 1)")
    b1, b2 = 0.9, 0.999
    m = s = 0.0
    print(" t |   m_t   |   s_t    | m_hat | s_hat  | uncorrected m/sqrt(s) | corrected m_hat/sqrt(s_hat)")
    for t in range(1, 4):
        m = b1 * m + (1 - b1) * 1.0
        s = b2 * s + (1 - b2) * 1.0
        m_hat, s_hat = m / (1 - b1 ** t), s / (1 - b2 ** t)
        print(f" {t} | {m:.4f} | {s:.6f} | {m_hat:.3f} | {s_hat:.4f} | {m / math.sqrt(s):21.3f} | {m_hat / math.sqrt(s_hat):26.3f}")
    print(f"first uncorrected step would be {0.1 / math.sqrt(0.001):.2f} x lr instead of 1 x lr")
    # SGD / RMSProp first steps on the quadratic for the sample-data table
    p = np.array([-9.0, 2.0])
    print("quadratic f = 0.5 (x^2 + 20 y^2), start (-9, 2): gradient", grad_quad(p), " f =", quad(*p))
    print("stability limit of plain GD: 2/kappa =", 2 / KAPPA, "; multipliers at lr 0.09: x", 1 - 0.09, " y", round(1 - 0.09 * KAPPA, 2))
    th = p.copy()
    for t in range(1, 4):
        g = grad_quad(th); th = th - 0.09 * g
        print(f"  SGD lr=0.09 step {t}: gradient ({g[0]:7.3f}, {g[1]:7.3f}) -> theta ({th[0]:7.3f}, {th[1]:7.3f})  f = {quad(*th):8.3f}")


def fig_optimizer_paths_quadratic():
    banner("Section 2: trajectories on the badly conditioned quadratic (kappa = 20, 100 steps)")
    configs = [("SGD", 0.09), ("Momentum", 0.02), ("Nesterov", 0.02), ("AdaGrad", 1.0), ("RMSProp", 0.2), ("Adam", 0.3)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs, ys = np.linspace(-10, 4, 300), np.linspace(-3, 3, 300)
    XX, YY = np.meshgrid(xs, ys)
    ax.contour(XX, YY, quad(XX, YY), levels=np.linspace(0, 9, 19) ** 2, colors=[AXIS], linewidths=0.6)
    for k, (kind, lr) in enumerate(configs):
        path = run_2d(grad_quad, kind, lr, (-9.0, 2.0), 100)
        f_end = quad(*path[-1])
        r = np.hypot(path[:, 0], path[:, 1])
        first = int(np.argmax(r < 1.0)) if (r < 1.0).any() else None
        print(f"  {kind:9s} lr={lr:<5}: first step to ({path[1, 0]:6.3f}, {path[1, 1]:6.3f}); |theta| < 1 after {first} steps; "
              f"final ({path[-1, 0]:7.4f}, {path[-1, 1]:7.4f}), f = {f_end:.1e}")
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.2, color=SERIES[k], label=f"{kind} lr={lr}")
        ax.plot(path[-1, 0], path[-1, 1], "o", ms=6, color=SERIES[k])
    ax.plot(0, 0, "*", ms=12, color=INK, label="minimum")
    ax.set_xlim(-10, 4); ax.set_ylim(-3, 3); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Six optimizers on f = 0.5 (x^2 + 20 y^2), 100 steps each")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, NB, "optimizer_paths_quadratic")


def fig_optimizer_paths_rosenbrock():
    banner("Section 2: trajectories on the Rosenbrock function (500 steps)")
    configs = [("SGD", 1e-3), ("Momentum", 1e-3), ("Nesterov", 1e-3), ("AdaGrad", 0.3), ("RMSProp", 0.01), ("Adam", 0.02)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs, ys = np.linspace(-2, 2, 300), np.linspace(-1, 3, 300)
    XX, YY = np.meshgrid(xs, ys)
    ax.contour(XX, YY, rosen(XX, YY), levels=np.logspace(-0.5, 3.5, 20), colors=[AXIS], linewidths=0.6)
    for k, (kind, lr) in enumerate(configs):
        path = run_2d(grad_rosen, kind, lr, (-1.2, 1.0), 500)
        d = math.hypot(path[-1, 0] - 1, path[-1, 1] - 1)
        print(f"  {kind:9s} lr={lr:<6}: final ({path[-1, 0]:6.3f}, {path[-1, 1]:6.3f}), distance to (1, 1) = {d:.3f}, f = {rosen(*path[-1]):.3g}")
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.2, color=SERIES[k], label=f"{kind} lr={lr}")
        ax.plot(path[-1, 0], path[-1, 1], "o", ms=6, color=SERIES[k])
    ax.plot(1, 1, "*", ms=12, color=INK, label="minimum (1, 1)")
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 3); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Six optimizers on the Rosenbrock valley, 500 steps each")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, NB, "optimizer_paths_rosenbrock")
    for lr in (2e-3, 1e-2):
        path = run_2d(grad_rosen, "SGD", lr, (-1.2, 1.0), 500)
        print(f"  SGD lr={lr}: {'diverged after ' + str(len(path) - 1) + ' steps' if len(path) < 501 else 'stayed finite for 500 steps'}")


# ----------------------------------------------------------------------------------------------
# Section 3: learning-rate schedules and a range test
# ----------------------------------------------------------------------------------------------
def schedules(T=1000, lr0=0.1):
    t = np.arange(T)
    out = {"constant": np.full(T, lr0),
           "step (x0.1 every 300)": lr0 * 0.1 ** (t // 300),
           "exponential (0.997^t)": lr0 * 0.997 ** t,
           "cosine": lr0 * 0.5 * (1 + np.cos(np.pi * t / T))}
    w = 100
    warm = np.where(t < w, (t + 1) / w, 0.5 * (1 + np.cos(np.pi * np.minimum(1.0, (t - w) / (T - w)))))
    out["warmup 10% + cosine"] = lr0 * warm
    init, fin, up = lr0 / 25, lr0 / 25 / 1e4, int(0.3 * T)
    out["one-cycle"] = np.where(t < up, init + (lr0 - init) * 0.5 * (1 - np.cos(np.pi * t / up)),
                                fin + (lr0 - fin) * 0.5 * (1 + np.cos(np.pi * (t - up) / (T - up))))
    return out


def fig_lr_schedules():
    banner("Section 3: learning-rate schedules over 1000 steps, base lr 0.1")
    sch = schedules()
    checkpoints = [0, 100, 300, 500, 999]
    print("learning rate at steps", checkpoints)
    for name, lr in sch.items():
        print(f"  {name:22s}: " + "  ".join(f"{lr[c]:.5f}" for c in checkpoints))
    fig, ax = plt.subplots()
    for k, (name, lr) in enumerate(sch.items()):
        ax.plot(lr, color=SERIES[k], label=name)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("learning rate")
    ax.set_title("Six learning-rate schedules (base lr = 0.1, 1000 steps)")
    ax.legend(fontsize=8)
    save(fig, NB, "lr_schedules")


def fig_lr_range_test():
    banner("Section 3: a learning-rate range test on a noisy 20-D quadratic (momentum SGD)")
    r = np.random.default_rng(3)
    d = 20
    curv = np.linspace(1.0, 50.0, d)                    # eigenvalues of the Hessian
    theta = r.normal(0, 3, d)
    lr_min, lr_max, steps, beta = 1e-4, 10.0, 200, 0.98
    gamma = (lr_max / lr_min) ** (1 / steps)
    lr, v, avg, best = lr_min, np.zeros(d), 0.0, float("inf")
    lrs, losses = [], []
    for i in range(steps):
        g = curv * theta + r.normal(0, 2.0, d)          # noisy gradient
        loss = 0.5 * np.sum(curv * theta ** 2) + 10.0  # +10: an irreducible noise floor
        v = 0.9 * v + g
        theta = theta - lr * v
        avg = beta * avg + (1 - beta) * loss
        smoothed = avg / (1 - beta ** (i + 1))
        if not np.isfinite(smoothed) or smoothed > 4 * best:
            break
        lrs.append(lr); losses.append(smoothed); best = min(best, smoothed)
        lr *= gamma
    lrs, losses = np.array(lrs), np.array(losses)
    i_min = int(np.argmin(losses))
    print(f"stopped after {len(lrs)} steps at lr = {lrs[-1]:.3g} (smoothed loss exceeded 4 x best); start loss {losses[0]:.1f}")
    print(f"lowest smoothed loss {losses[i_min]:.2f} at lr = {lrs[i_min]:.3g}; suggested lr = minimum/5 = {lrs[i_min] / 5:.3g}")
    print(f"for reference: the stability limit of momentum SGD on this quadratic is 2(1+beta)/lambda_max = {2 * 1.9 / curv.max():.3g}")
    fig, ax = plt.subplots()
    ax.semilogx(lrs, losses, color=SERIES[0])
    ax.axvline(lrs[i_min], color=SERIES[1], ls="--", lw=1.2, label=f"loss minimum at lr = {lrs[i_min]:.2g}")
    ax.axvline(lrs[i_min] / 5, color=SERIES[2], ls="--", lw=1.2, label=f"suggested lr = {lrs[i_min] / 5:.2g}")
    ax.set_xlabel("learning rate (log scale)"); ax.set_ylabel("smoothed training loss")
    ax.set_title("Learning-rate range test: flat, falling, exploding"); ax.legend()
    save(fig, NB, "lr_range_test")


# ----------------------------------------------------------------------------------------------
# Section 4: variance propagation through 10 layers
# ----------------------------------------------------------------------------------------------
def fig_variance_propagation():
    banner("Section 4: activation std through 10 layers of width 256 (input dim 784, 512 examples)")
    r = np.random.default_rng(4)
    depth, width, in_dim, n = 10, 256, 784, 512
    inits = {"zero": lambda fi, fo: 0.0, "small (std 0.01)": lambda fi, fo: 0.01, "large (std 0.2)": lambda fi, fo: 0.2,
             "Xavier": lambda fi, fo: math.sqrt(2.0 / (fi + fo)), "He": lambda fi, fo: math.sqrt(2.0 / fi)}
    print("per-layer factor n_in * sigma_W^2 for a 256 -> 256 layer, and the std multiplier sqrt of it:")
    for name, f in inits.items():
        fac = 256 * f(256, 256) ** 2
        print(f"  {name:17s}: sigma_W = {f(256, 256):.4f}, n_in sigma_W^2 = {fac:.4f}, std multiplier per linear layer = {math.sqrt(fac):.3f}")
    X = r.normal(0, 1, (n, in_dim))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    results = {}
    for col, (act_name, act) in enumerate([("tanh", np.tanh), ("ReLU", lambda z: np.maximum(z, 0))]):
        for k, (name, f) in enumerate(inits.items()):
            rr = np.random.default_rng(5)
            h, stds = X, []
            dims = [in_dim] + [width] * depth
            for L in range(depth):
                W = rr.normal(0, 1, (dims[L], dims[L + 1])) * f(dims[L], dims[L + 1])
                h = act(h @ W)
                stds.append(h.std())
            results[(act_name, name)] = stds
            axes[col].semilogy(range(1, depth + 1), np.maximum(stds, 1e-12), "o-", ms=6, color=SERIES[k], label=name)
        axes[col].set_title(f"{act_name} network: activation std per layer")
        axes[col].set_xlabel("layer"); axes[col].set_ylim(1e-12, 1e4)
    axes[0].set_ylabel("std of activations (log)"); axes[1].legend(fontsize=8, loc="lower left")
    save(fig, NB, "variance_propagation")
    print("activation std after layers 1, 5, 10:")
    for (act_name, name), stds in results.items():
        print(f"  {act_name:4s} {name:17s}: {stds[0]:.3e}  {stds[4]:.3e}  {stds[9]:.3e}")
    print(f"hand prediction for 'small' with tanh ~ identity: 0.28 * 0.16^9 = {math.sqrt(784e-4) * math.sqrt(0.0256) ** 9:.2e}")


# ----------------------------------------------------------------------------------------------
# Section 5: BatchNorm on a 4 x 3 batch and the noise of batch statistics
# ----------------------------------------------------------------------------------------------
def fig_batchnorm_example():
    banner("Section 5: BatchNorm on a hand-typed 4 x 3 batch")
    X = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 110.0], [3.0, 30.0, 90.0], [6.0, 40.0, 100.0]])
    eps = 1e-5
    mu, var = X.mean(0), X.var(0)
    xhat = (X - mu) / np.sqrt(var + eps)
    print("batch X (rows = examples, columns = features):"); print(X)
    print("batch mean per feature:", mu); print("biased batch variance per feature:", var)
    print("normalized x_hat (gamma = 1, beta = 0):"); print(np.round(xhat, 4))
    print("check: column means", np.round(xhat.mean(0), 6), " column stds", np.round(xhat.std(0), 4))
    n = X.shape[0]
    run_mean = 0.1 * mu; run_var = 0.9 * 1.0 + 0.1 * var * n / (n - 1)
    print("running stats after this one batch (momentum 0.1, from mean 0 / var 1): mean", np.round(run_mean, 3), " var", np.round(run_var, 3))
    print("unbiased variance stored (var * 4/3):", np.round(var * n / (n - 1), 3))
    gamma, beta = np.array([2.0, 1.0, 0.5]), np.array([0.0, 1.0, -1.0])
    print("with gamma = [2, 1, 0.5], beta = [0, 1, -1] the first row becomes:", np.round(gamma * xhat[0] + beta, 4))
    # noise of batch statistics
    r = np.random.default_rng(6)
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    mean_err, var_err = [], []
    for B in sizes:
        samples = r.normal(0, 1, (4000, B))
        mean_err.append(samples.mean(1).std())
        var_err.append(samples.var(1).std())
    print("std of the batch mean / batch variance estimates of a N(0,1) feature vs batch size:")
    for B, a, b in zip(sizes, mean_err, var_err):
        print(f"  B = {B:3d}: mean estimate std {a:.3f} (1/sqrt(B) = {1 / math.sqrt(B):.3f}), variance estimate std {b:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for j in range(3):
        axes[0].plot(np.full(n, j), X[:, j], "o", ms=7, color=SERIES[j], label=f"feature {j + 1}")
        axes[0].plot([j - 0.25, j + 0.25], [mu[j], mu[j]], "-", color=SERIES[j], lw=1.5)
        axes[1].plot(np.full(n, j), xhat[:, j], "o", ms=7, color=SERIES[j])
        axes[1].plot([j - 0.25, j + 0.25], [0, 0], "-", color=INK_2, lw=1.0)
    axes[0].set_xticks([0, 1, 2]); axes[0].set_xticklabels(["f1", "f2", "f3"]); axes[0].set_ylabel("raw value")
    axes[0].set_title("Before: three very different scales"); axes[0].legend(fontsize=8)
    axes[1].set_xticks([0, 1, 2]); axes[1].set_xticklabels(["f1", "f2", "f3"]); axes[1].set_ylabel("normalized value")
    axes[1].set_title("After BatchNorm: mean 0, std 1 per feature"); axes[1].set_ylim(-2, 2)
    axes[2].loglog(sizes, mean_err, "o-", ms=6, color=SERIES[0], label="std of batch mean")
    axes[2].loglog(sizes, var_err, "s-", ms=6, color=SERIES[1], label="std of batch variance")
    axes[2].set_xlabel("batch size B"); axes[2].set_ylabel("error of the batch statistic")
    axes[2].set_title("Why tiny batches hurt BatchNorm"); axes[2].legend(fontsize=8)
    save(fig, NB, "batchnorm_example")


# ----------------------------------------------------------------------------------------------
# Section 6: overfitting with polynomials, weight decay, dropout mask, label smoothing
# ----------------------------------------------------------------------------------------------
def fig_overfit_polynomial():
    banner("Section 6: overfitting seen with polynomial fits (12 noisy points of sin(2 pi x))")
    r = np.random.default_rng(7)
    n_train = 12
    x_tr = np.sort(r.uniform(0, 1, n_train)); y_tr = np.sin(2 * np.pi * x_tr) + 0.3 * r.normal(0, 1, n_train)
    x_te = np.linspace(0.02, 0.98, 200); y_te = np.sin(2 * np.pi * x_te) + 0.3 * r.normal(0, 1, len(x_te))
    print("sample data (first 6 training points): x, y")
    for i in range(6):
        print(f"  {i + 1}: x = {x_tr[i]:.3f}  y = {y_tr[i]:6.3f}")

    def design(x, deg): return np.vander(2 * x - 1, deg + 1, increasing=True)

    def fit(x, y, deg, lam=0.0):
        A = design(x, deg)
        pen = lam * np.eye(deg + 1); pen[0, 0] = 0.0           # do not penalize the intercept
        return np.linalg.solve(A.T @ A + pen, A.T @ y)

    degs = list(range(0, 10))
    tr_err, te_err = [], []
    for dg in degs:
        w = fit(x_tr, y_tr, dg)
        tr_err.append(np.mean((design(x_tr, dg) @ w - y_tr) ** 2))
        te_err.append(np.mean((design(x_te, dg) @ w - y_te) ** 2))
    print("train / test MSE by polynomial degree:")
    for dg, a, b in zip(degs, tr_err, te_err):
        print(f"  degree {dg}: train {a:.4f}  test {b:.4f}")
    print(f"best test MSE at degree {degs[int(np.argmin(te_err))]}; noise floor sigma^2 = 0.09")
    lams = np.logspace(-6, 1, 40)
    te_ridge = [np.mean((design(x_te, 9) @ fit(x_tr, y_tr, 9, lam) - y_te) ** 2) for lam in lams]
    tr_ridge = [np.mean((design(x_tr, 9) @ fit(x_tr, y_tr, 9, lam) - y_tr) ** 2) for lam in lams]
    i_best = int(np.argmin(te_ridge))
    w9 = fit(x_tr, y_tr, 9); w9r = fit(x_tr, y_tr, 9, lams[i_best])
    print(f"degree 9 with L2 penalty: best lambda = {lams[i_best]:.2e}, test MSE {te_ridge[i_best]:.4f} (vs {te_err[9]:.4f} without); "
          f"weight norm {np.linalg.norm(w9):.1f} -> {np.linalg.norm(w9r):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    xx = np.linspace(0, 1, 300)
    axes[0].plot(x_tr, y_tr, "o", ms=7, color=INK, label="training points", zorder=5)
    for k, dg in enumerate((1, 3, 9)):
        w = fit(x_tr, y_tr, dg)
        axes[0].plot(xx, design(xx, dg) @ w, color=SERIES[k], label=f"degree {dg}")
    axes[0].set_ylim(-2.2, 2.2); axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("Under-, well- and over-fitted")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(degs, tr_err, "o-", ms=6, color=SERIES[0], label="training MSE")
    axes[1].semilogy(degs, te_err, "s-", ms=6, color=SERIES[1], label="test MSE")
    axes[1].axhline(0.09, color=MUTED, ls=":", lw=1); axes[1].annotate("noise floor 0.09", (0.2, 0.1), fontsize=8, color=MUTED)
    axes[1].set_xlabel("polynomial degree (capacity)"); axes[1].set_ylabel("mean squared error (log)")
    axes[1].set_title("The U-shaped test error"); axes[1].legend(fontsize=8)
    axes[2].semilogx(lams, tr_ridge, color=SERIES[0], label="training MSE")
    axes[2].semilogx(lams, te_ridge, color=SERIES[1], label="test MSE")
    axes[2].plot(lams[i_best], te_ridge[i_best], "o", ms=7, color=SERIES[1])
    axes[2].annotate(f"best lambda = {lams[i_best]:.1e}", (lams[i_best], te_ridge[i_best]), xytext=(8, 12),
                     textcoords="offset points", fontsize=8)
    axes[2].set_xlabel("L2 penalty lambda (degree 9)"); axes[2].set_ylabel("mean squared error")
    axes[2].set_title("Weight decay tames the degree-9 fit"); axes[2].legend(fontsize=8)
    save(fig, NB, "overfitting_polynomial")


def fig_dropout_and_label_smoothing():
    banner("Section 6: a dropout mask and label-smoothing targets")
    r = np.random.default_rng(8)
    h = np.round(r.uniform(0.5, 2.0, 12), 2)
    mask = (r.random(12) >= 0.5).astype(float)
    out = h * mask / 0.5
    print("activations h:", h)
    print("mask (1 = keep), p = 0.5:", mask.astype(int))
    print("output h * mask / (1 - p):", out)
    print(f"input mean {h.mean():.3f}, output mean {out.mean():.3f}, units kept {int(mask.sum())} of 12, survivors scaled by {1 / 0.5:.0f}x")
    big = np.full(200_000, 2.0); big_mask = (r.random(200_000) >= 0.5)
    big_out = big * big_mask / 0.5
    print(f"over 200,000 units of value 2.0: fraction zeroed {1 - big_mask.mean():.3f}, output mean {big_out.mean():.3f} (expectation preserved)")
    K, alpha, y = 10, 0.1, 3
    onehot = np.zeros(K); onehot[y] = 1
    smooth = (1 - alpha) * onehot + alpha / K
    print("label smoothing, K = 10, alpha = 0.1, true class 3:")
    print("  one-hot target:", onehot); print("  smoothed target:", np.round(smooth, 3), " (sums to", smooth.sum(), ")")
    p_ideal = smooth
    loss_ideal = -(smooth * np.log(p_ideal)).sum()
    p_conf = np.full(K, 1e-4); p_conf[y] = 1 - 9e-4
    loss_conf = -(smooth * np.log(p_conf)).sum()
    print(f"  smoothed loss when predicting the smoothed target itself: {loss_ideal:.4f}; when predicting 0.9991 on the true class: {loss_conf:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    M = np.vstack([h, mask * h.max(), out / 2])
    ax1.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=h.max())
    for i in range(12):
        ax1.text(i, 0, f"{h[i]:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax1.text(i, 1, "keep" if mask[i] else "drop", ha="center", va="center", fontsize=7, color=INK if mask[i] else SURFACE)
        ax1.text(i, 2, f"{out[i]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax1.set_yticks([0, 1, 2]); ax1.set_yticklabels(["input h", "mask (p = 0.5)", "output h·mask/(1-p)"]); ax1.set_xticks([])
    ax1.grid(False); ax1.set_title(f"Inverted dropout (p = 0.5): {int(mask.sum())} of 12 units kept, survivors doubled")
    xk = np.arange(K)
    ax2.bar(xk - 0.2, onehot, width=0.4, color=SERIES[0], label="one-hot target")
    ax2.bar(xk + 0.2, smooth, width=0.4, color=SERIES[1], label="smoothed, alpha = 0.1")
    ax2.annotate("0.91", (3.2, 0.91), xytext=(10, 0), textcoords="offset points", fontsize=8, va="center")
    ax2.annotate("0.01 each", (7, 0.01), xytext=(0, 10), textcoords="offset points", fontsize=8, ha="center")
    ax2.set_xticks(xk); ax2.set_xlabel("class"); ax2.set_ylabel("target probability")
    ax2.set_title("Label smoothing softens the target"); ax2.legend(fontsize=8)
    save(fig, NB, "dropout_label_smoothing")


# ----------------------------------------------------------------------------------------------
# Section 7: loss-curve shapes and the overfit-one-batch check
# ----------------------------------------------------------------------------------------------
def fig_loss_curve_shapes():
    banner("Section 7: six loss-curve shapes, all simulated")
    r = np.random.default_rng(9)
    N, b = 256, 16
    x = r.normal(0, 1, N); y = 2.0 * x + 0.5 * r.normal(0, 1, N)

    def sgd_curve(lr, epochs=6, seed=10):
        rr = np.random.default_rng(seed); th, losses = 0.0, []
        for _ in range(epochs):
            perm = rr.permutation(N)
            for i in range(0, N, b):
                idx = perm[i:i + b]
                losses.append(0.5 * np.mean((th * x[idx] - y[idx]) ** 2))
                th -= lr * np.mean((th * x[idx] - y[idx]) * x[idx])
                if not np.isfinite(th) or abs(th) > 1e6:
                    return np.array(losses + [np.nan] * (epochs * (N // b) - len(losses)))
        return np.array(losses)

    low, high, healthy = sgd_curve(0.001), sgd_curve(3.0), sgd_curve(0.05)
    n_finite = int(np.sum(np.isfinite(high)))
    print(f"lr too low (0.001): loss after {len(low)} steps {low[-1]:.3f} (start {low[0]:.3f})")
    print(f"lr too high (3.0): first losses {np.round(high[:5], 1)}; largest finite loss {np.nanmax(high):.3g}; non-finite after {n_finite} steps")
    print(f"healthy (0.05): loss after {len(healthy)} steps {healthy[-1]:.3f}; below 0.3 after {int(np.argmax(healthy < 0.3))} steps")
    # plateau: a chain of 10 scalar weights, loss 0.5 (prod w - 1)^2, small init
    n_w = 10
    w = np.full(n_w, 0.5); plateau = []
    for t in range(800):
        P = np.prod(w); plateau.append(0.5 * (P - 1) ** 2)
        g = (P - 1) * P / w
        w -= 0.1 * g
    plateau = np.array(plateau)
    print(f"bad-init plateau ({n_w} weights at 0.5, product {0.5 ** n_w:.5f}, lr 0.1): loss stays within 1% of {plateau[0]:.3f} for "
          f"{int(np.argmax(plateau < 0.99 * plateau[0]))} steps, drops below 0.1 at step {int(np.argmax(plateau < 0.1))}, final {plateau[-1]:.1e}")
    # overfitting and underfitting: tiny MLP on make_moons
    from sklearn.datasets import make_moons
    Xtr, ytr = make_moons(n_samples=40, noise=0.3, random_state=0)
    Xte, yte = make_moons(n_samples=1000, noise=0.3, random_state=1)
    net = TinyMLP(2, 64, 2, np.random.default_rng(11))
    opt = NpOptimizer("Adam", 0.01)
    tr_l, te_l = [], []
    for ep in range(1500):
        logits, cache = net.forward(Xtr)
        loss, dl = ce_loss_and_grad(logits, ytr)
        grads = net.backward(cache, dl)
        opt.step(net.params, grads)
        tr_l.append(loss); te_l.append(ce_loss_and_grad(net.forward(Xte)[0], yte)[0])
    tr_l, te_l = np.array(tr_l), np.array(te_l)
    i_best = int(np.argmin(te_l))
    print(f"overfitting (40 moons points, 64 hidden units, 1500 epochs): train loss {tr_l[-1]:.4f}, test loss {te_l[-1]:.3f}; "
          f"best test loss {te_l[i_best]:.3f} at epoch {i_best + 1}")
    print(f"underfitting / too short: the first 40 epochs of the same run end at train {tr_l[39]:.3f}, test {te_l[39]:.3f}, both still falling")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    panels = [(axes[0, 0], "Learning rate too low", low, "barely moves"),
              (axes[0, 1], "Learning rate too high", np.minimum(high, 5.0), "spikes off the chart, then inf / NaN"),
              (axes[0, 2], "Healthy", healthy, "fast drop, slow noisy decline"),
              (axes[1, 0], "Bad initialization", plateau, "flat plateau, then a sudden drop")]
    for ax, title, losses, note in panels:
        ax.plot(losses, lw=1.0, color=SERIES[0])
        ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.text(0.03, 0.9, note, transform=ax.transAxes, fontsize=8, color=INK_2)
    axes[0, 0].set_ylim(0, 3); axes[0, 1].set_ylim(0, 5.2); axes[0, 2].set_ylim(0, 3); axes[1, 0].set_ylim(0, 0.6)
    ax = axes[1, 1]
    ax.plot(tr_l, color=SERIES[0], label="train"); ax.plot(te_l, color=SERIES[1], label="validation")
    ax.axvline(i_best + 1, color=MUTED, ls=":", lw=1); ax.set_ylim(0, 1.2)
    ax.set_title("Overfitting"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.text(0.03, 0.9, f"validation turns up after epoch {i_best + 1}", transform=ax.transAxes, fontsize=8, color=INK_2)
    ax = axes[1, 2]
    ax.plot(tr_l[:40], color=SERIES[0], label="train"); ax.plot(te_l[:40], color=SERIES[1], label="validation")
    ax.set_ylim(0, 1.2); ax.set_title("Underfitting / stopped too early"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.text(0.03, 0.9, "both still high and falling", transform=ax.transAxes, fontsize=8, color=INK_2)
    save(fig, NB, "loss_curve_shapes")


def fig_overfit_one_batch():
    banner("Section 7: the overfit-one-batch check (16 random examples, 10 classes)")
    r = np.random.default_rng(12)
    Xb = r.normal(0, 1, (16, 20)); yb = r.integers(0, 10, 16)
    results = {}
    for name, bug in [("correct", False), ("double softmax", True)]:
        net = TinyMLP(20, 64, 10, np.random.default_rng(13))
        opt = NpOptimizer("Adam", 0.01)
        losses = []
        for _ in range(300):
            logits, cache = net.forward(Xb)
            loss, dl = ce_loss_and_grad(logits, yb, double_softmax=bug)
            opt.step(net.params, net.backward(cache, dl))
            losses.append(loss)
        results[name] = np.array(losses)
        print(f"  {name:14s}: loss at step 1 {losses[0]:.3f}, step 100 {losses[99]:.4f}, step 300 {losses[-1]:.4f}")
    floor = -math.log(math.e / (math.e + 9))
    print(f"theoretical floor of the double-softmax loss: e/(e+9) = {math.e / (math.e + 9):.4f} -> -ln = {floor:.4f}")
    fig, ax = plt.subplots()
    ax.semilogy(results["correct"], color=SERIES[0], label="correct model: loss -> 0")
    ax.semilogy(results["double softmax"], color=SERIES[1], label="softmax before cross-entropy: stuck")
    ax.axhline(floor, color=MUTED, ls=":", lw=1)
    ax.annotate(f"floor {floor:.2f}", (150, floor), xytext=(0, 6), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("step"); ax.set_ylabel("loss on the single batch (log)"); ax.set_title("Overfit-one-batch sanity check")
    ax.legend()
    save(fig, NB, "overfit_one_batch")


# ----------------------------------------------------------------------------------------------
# Section 8: grid versus random search, log-uniform sampling
# ----------------------------------------------------------------------------------------------
def fig_grid_vs_random():
    banner("Section 8: grid vs random search (9 trials) and sampling scale")
    r = np.random.default_rng(14)
    grid = np.array([(i, j) for i in np.linspace(0.15, 0.85, 3) for j in np.linspace(0.15, 0.85, 3)])
    rand = r.uniform(0.02, 0.98, size=(9, 2))
    print("grid points (important, unimportant):", [(round(float(a), 2), round(float(b), 2)) for a, b in grid])
    print("random points:", [(round(float(a), 2), round(float(b), 2)) for a, b in rand])
    print(f"distinct values of the important hyperparameter: grid {len(np.unique(np.round(grid[:, 0], 3)))}, random {len(np.unique(np.round(rand[:, 0], 3)))}")
    # a toy objective that depends almost only on the important hyperparameter
    def score(p): return np.exp(-((p[:, 0] - 0.62) / 0.12) ** 2) * (1 - 0.05 * p[:, 1])
    print(f"toy objective (peak at important = 0.62): best grid score {score(grid).max():.3f}, best random score {score(rand).max():.3f}")
    u = r.uniform(1e-4, 1e-1, 2000); lu = 10 ** r.uniform(-4, -1, 2000)
    print(f"uniform draws on [1e-4, 1e-1]: {np.mean(u > 1e-2) * 100:.1f}% above 1e-2, {np.mean(u < 1e-3) * 100:.1f}% below 1e-3")
    print(f"log-uniform draws: {np.mean(lu > 1e-2) * 100:.1f}% above 1e-2, {np.mean(lu < 1e-3) * 100:.1f}% below 1e-3")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, pts, name in [(axes[0], grid, "Grid search"), (axes[1], rand, "Random search")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=60, color=SERIES[0], zorder=3)
        ax.scatter(pts[:, 0], np.full(len(pts), -0.06), marker="|", s=200, color=SERIES[1], zorder=3)
        n_distinct = len(np.unique(np.round(pts[:, 0], 3)))
        ax.set_title(f"{name}: {n_distinct} distinct values tried")
        ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1)
        ax.set_xlabel("important hyperparameter (ticks = projection)"); ax.set_ylabel("unimportant hyperparameter")
    bins = np.logspace(-4, -1, 25)
    axes[2].hist(u, bins=bins, color=SERIES[0], alpha=0.85, label="uniform on [1e-4, 1e-1]")
    axes[2].hist(lu, bins=bins, color=SERIES[1], alpha=0.75, label="log-uniform (10^U(-4, -1))")
    axes[2].set_xscale("log"); axes[2].set_xlabel("sampled learning rate (log axis)"); axes[2].set_ylabel("count of 2000 draws")
    axes[2].set_title("Sample the exponent, not the value"); axes[2].legend(fontsize=8)
    save(fig, NB, "grid_vs_random_search")


def main():
    fig_sgd_noise()
    fig_adam_by_hand()
    fig_optimizer_paths_quadratic()
    fig_optimizer_paths_rosenbrock()
    fig_lr_schedules()
    fig_lr_range_test()
    fig_variance_propagation()
    fig_batchnorm_example()
    fig_overfit_polynomial()
    fig_dropout_and_label_smoothing()
    fig_loss_curve_shapes()
    fig_overfit_one_batch()
    fig_grid_vs_random()


if __name__ == "__main__":
    main()
