"""Figures for the companion guide of Notebook 16 (interpretability, robustness, uncertainty).

Everything is numpy + matplotlib (plus the bundled scikit-learn `load_digits` for a tiny real
image dataset). Every number quoted in the guide is printed by the function that draws the figure.
Run:  python3 notebooks/figures/scripts/make_figures_16.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import math
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

NB = "16"


# ----------------------------------------------------------------------------------------------
# small numerical helpers (no scipy, no torch)
# ----------------------------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def norm_ppf(p):
    """Inverse standard-normal CDF by bisection (vectorised, accurate to 1e-10)."""
    p = np.asarray(p, dtype=float)
    lo, hi = np.full(p.shape, -10.0), np.full(p.shape, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = norm_cdf(mid) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def auroc(score_pos, score_neg):
    """P(random positive scores higher than random negative), ties count one half (Mann-Whitney)."""
    s = np.concatenate([score_pos, score_neg])
    ranks = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    i = 0
    while i < len(s):                       # average ranks for ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = len(score_pos), len(score_neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve(score_pos, score_neg):
    thresholds = np.sort(np.concatenate([score_pos, score_neg]))[::-1]
    tpr = [(score_pos >= t).mean() for t in thresholds]
    fpr = [(score_neg >= t).mean() for t in thresholds]
    return np.array([0.0] + fpr + [1.0]), np.array([0.0] + tpr + [1.0])


def adam_step(params, grads, state, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in params:
        m, v = state[k]
        m = beta1 * m + (1 - beta1) * grads[k]
        v = beta2 * v + (1 - beta2) * grads[k] ** 2
        state[k] = (m, v)
        params[k] -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)


def train_logreg(X, y, n_classes, steps=400, lr=0.05, weight_decay=1e-4, seed=0):
    """Multinomial logistic regression with Adam, full batch. Returns (W, b)."""
    r = np.random.default_rng(seed)
    W = 0.01 * r.standard_normal((X.shape[1], n_classes)); b = np.zeros(n_classes)
    params = {"W": W, "b": b}; state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    Y = np.eye(n_classes)[y]
    for t in range(1, steps + 1):
        P = softmax(X @ params["W"] + params["b"])
        G = (P - Y) / len(X)
        grads = {"W": X.T @ G + weight_decay * params["W"], "b": G.sum(0)}
        adam_step(params, grads, state, lr, t)
    return params["W"], params["b"]


def logreg_acc(W, b, X, y):
    return ((X @ W + b).argmax(1) == y).mean()


# ----------------------------------------------------------------------------------------------
# 1. Why trust matters: shortcut learning on a toy husky/wolf task
# ----------------------------------------------------------------------------------------------
def fig_shortcut_learning():
    r = np.random.default_rng(1)

    def make(n, p_snow_given_wolf, p_snow_given_husky):
        y = r.integers(0, 2, n)                                         # 1 = wolf
        muzzle = 1.0 * y + r.normal(0.0, 1.0, n)                        # intended feature, weakly informative
        snow = np.where(y == 1, r.random(n) < p_snow_given_wolf, r.random(n) < p_snow_given_husky).astype(float)
        return np.stack([muzzle, snow], 1), y

    X_tr, y_tr = make(2000, 0.9, 0.1)
    X_iid, y_iid = make(2000, 0.9, 0.1)
    X_shift, y_shift = make(2000, 0.5, 0.5)                             # snow no longer tells wolf from husky
    print("\n[fig_shortcut_learning] sample rows (muzzle score, snow, label 1=wolf):")
    for i in range(6):
        print(f"   {X_tr[i, 0]:6.2f}  {int(X_tr[i, 1])}  {y_tr[i]}")
    print(f"   P(snow | wolf) train = {X_tr[y_tr == 1, 1].mean():.3f}, P(snow | husky) train = {X_tr[y_tr == 0, 1].mean():.3f}")
    W_both, b_both = train_logreg(X_tr, y_tr, 2)
    W_muz, b_muz = train_logreg(X_tr[:, :1], y_tr, 2)
    res = {}
    for name, (W, b, cols) in {"muzzle + snow": (W_both, b_both, slice(None)), "muzzle only": (W_muz, b_muz, slice(0, 1))}.items():
        res[name] = (logreg_acc(W, b, X_tr[:, cols], y_tr), logreg_acc(W, b, X_iid[:, cols], y_iid), logreg_acc(W, b, X_shift[:, cols], y_shift))
        print(f"   model '{name}': train acc {res[name][0]:.3f} | same-distribution test {res[name][1]:.3f} | snow-decorrelated test {res[name][2]:.3f}")
    print(f"   weights of the two-feature model: muzzle {W_both[0, 1] - W_both[0, 0]:+.2f}, snow {W_both[1, 1] - W_both[1, 0]:+.2f} (log-odds for 'wolf')")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["same distribution\n(snow correlated)", "deployment\n(snow decorrelated)"]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [res["muzzle + snow"][1], res["muzzle + snow"][2]], w, color=SERIES[0], label="model with shortcut (muzzle + snow)", edgecolor=SURFACE)
    ax.bar(x + w / 2, [res["muzzle only"][1], res["muzzle only"][2]], w, color=SERIES[1], label="model on intended feature (muzzle)", edgecolor=SURFACE)
    for xi, v in zip(x - w / 2, [res["muzzle + snow"][1], res["muzzle + snow"][2]]):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=9, color=INK_2, xytext=(0, 2), textcoords="offset points")
    for xi, v in zip(x + w / 2, [res["muzzle only"][1], res["muzzle only"][2]]):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=9, color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy"); ax.set_title("Shortcut learning: the test set hides the failure")
    ax.legend(loc="upper right")
    save(fig, NB, "shortcut_learning")


# ----------------------------------------------------------------------------------------------
# 2. Attribution: Integrated Gradients as a path integral (2-feature logistic model)
# ----------------------------------------------------------------------------------------------
def fig_ig_path_integral():
    w = np.array([1.5, -2.0]); b = 0.5
    x = np.array([0.8, 0.6]); x0 = np.zeros(2)
    F = lambda v: sigmoid(w @ v + b)                       # class probability
    gradF = lambda v: F(v) * (1 - F(v)) * w
    print("\n[fig_ig_path_integral] model: F(x) = sigmoid(1.5 x1 - 2.0 x2 + 0.5); x = (0.8, 0.6); baseline = (0, 0)")
    print(f"   F(x) = {F(x):.4f}, F(baseline) = {F(x0):.4f}, difference = {F(x) - F(x0):.4f}")
    print(f"   plain gradient at x: {gradF(x)[0]:+.4f}, {gradF(x)[1]:+.4f}; gradient x input: {gradF(x)[0] * x[0]:+.4f}, {gradF(x)[1] * x[1]:+.4f}, sum {np.sum(gradF(x) * x):+.4f}")
    steps = 5
    alphas = (np.arange(steps) + 0.5) / steps
    print(f"   midpoint Riemann sum with {steps} steps:")
    print("   alpha   point on path         F(point)   dF/dx1    dF/dx2")
    grads = []
    for a in alphas:
        pt = x0 + a * (x - x0); g = gradF(pt); grads.append(g)
        print(f"   {a:4.1f}   ({pt[0]:.2f}, {pt[1]:.2f})          {F(pt):.4f}   {g[0]:+.4f}   {g[1]:+.4f}")
    grads = np.array(grads)
    mean_g = grads.mean(0)
    ig = (x - x0) * mean_g
    print(f"   mean gradient: {mean_g[0]:+.4f}, {mean_g[1]:+.4f}")
    print(f"   IG = (x - baseline) * mean gradient = {ig[0]:+.4f}, {ig[1]:+.4f}; sum = {ig.sum():+.4f} vs F(x) - F(baseline) = {F(x) - F(x0):.4f}")
    sums = {}
    for m in (5, 10, 20, 50, 200):
        al = (np.arange(m) + 0.5) / m
        gm = np.array([gradF(x0 + a * (x - x0)) for a in al]).mean(0)
        sums[m] = float(((x - x0) * gm).sum())
        print(f"   steps {m:4d}: sum of attributions = {sums[m]:.6f} (error {abs(sums[m] - (F(x) - F(x0))):.2e})")

    # left: g'(alpha) and the midpoint rectangles; right: convergence of the sum
    a_fine = np.linspace(0, 1, 400)
    gprime = np.array([(x - x0) @ gradF(x0 + a * (x - x0)) for a in a_fine])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.plot(a_fine, gprime, color=SERIES[0], label="g'(alpha) = (x - x') . grad F along the path")
    for a, g in zip(alphas, grads):
        h = (x - x0) @ g
        ax.bar(a, h, width=1 / steps, color=SERIES[1], alpha=0.35, edgecolor=SERIES[1], label="midpoint rectangles (5 steps)" if a == alphas[0] else None)
    ax.set_xlabel("alpha (position on the straight path from baseline to x)"); ax.set_ylabel("g'(alpha)")
    ax.set_title("The integral behind Integrated Gradients"); ax.legend(loc="upper right")
    ax = axes[1]
    ms = sorted(sums)
    ax.plot(ms, [sums[m] for m in ms], "o-", color=SERIES[0], label="sum of attributions")
    ax.axhline(F(x) - F(x0), color=SERIES[1], ls="--", label=f"F(x) - F(baseline) = {F(x) - F(x0):.4f}")
    ax.set_xscale("log"); ax.set_xlabel("number of Riemann steps"); ax.set_ylabel("sum of attributions")
    ax.set_title("Completeness check"); ax.legend(loc="lower right")
    ax.annotate(f"{sums[5]:.4f}", (5, sums[5]), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK_2)
    save(fig, NB, "ig_path_integral")


# ----------------------------------------------------------------------------------------------
# 3. Feature visualisation and probes: a numpy MLP on 8x8 digits
# ----------------------------------------------------------------------------------------------
def load_digits_split():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0; y = d.target
    r = np.random.default_rng(3)
    perm = r.permutation(len(X))
    tr, te = perm[:1200], perm[1200:]
    return X[tr], y[tr], X[te], y[te]


def mlp_init(seed, sizes=(64, 32, 32, 10)):
    r = np.random.default_rng(seed)
    P = {}
    for i in range(len(sizes) - 1):
        P[f"W{i + 1}"] = r.standard_normal((sizes[i], sizes[i + 1])) * math.sqrt(2.0 / sizes[i])
        P[f"b{i + 1}"] = np.zeros(sizes[i + 1])
    return P


def mlp_forward(P, X):
    z1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(z1, 0)
    z2 = h1 @ P["W2"] + P["b2"]; h2 = np.maximum(z2, 0)
    logits = h2 @ P["W3"] + P["b3"]
    return z1, h1, z2, h2, logits


def mlp_train(X, y, steps=400, lr=5e-3, seed=0):
    P = mlp_init(seed)
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in P.items()}
    Y = np.eye(10)[y]
    for t in range(1, steps + 1):
        z1, h1, z2, h2, logits = mlp_forward(P, X)
        G = (softmax(logits) - Y) / len(X)
        g = {"W3": h2.T @ G, "b3": G.sum(0)}
        d2 = (G @ P["W3"].T) * (z2 > 0)
        g["W2"] = h1.T @ d2; g["b2"] = d2.sum(0)
        d1 = (d2 @ P["W2"].T) * (z1 > 0)
        g["W1"] = X.T @ d1; g["b1"] = d1.sum(0)
        adam_step(P, g, state, lr, t)
    return P


def mlp_input_grad(P, X, cls):
    """d logit_cls / d X for a batch X (one class index per row)."""
    z1, h1, z2, h2, logits = mlp_forward(P, X)
    d3 = np.zeros_like(logits); d3[np.arange(len(X)), cls] = 1.0
    d2 = (d3 @ P["W3"].T) * (z2 > 0)
    d1 = (d2 @ P["W2"].T) * (z1 > 0)
    return d1 @ P["W1"].T, logits[np.arange(len(X)), cls]


_DIGITS = {}


def trained_digit_mlp():
    if not _DIGITS:
        Xtr, ytr, Xte, yte = load_digits_split()
        P = mlp_train(Xtr, ytr)
        _DIGITS.update(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, P=P)
    return _DIGITS


def fig_linear_probes_depth():
    D = trained_digit_mlp(); Xtr, ytr, Xte, yte, P = D["Xtr"], D["ytr"], D["Xte"], D["yte"], D["P"]
    acc_model = (mlp_forward(P, Xte)[-1].argmax(1) == yte).mean()
    P_rand = mlp_init(99)
    print(f"\n[fig_linear_probes_depth] digits 8x8: {len(Xtr)} train / {len(Xte)} test; MLP 64-32-32-10 test accuracy {acc_model:.3f}")
    results = {}
    for net_name, Pn in (("trained network", P), ("random network", P_rand)):
        feats_tr = {"pixels": Xtr, "hidden 1": mlp_forward(Pn, Xtr)[1], "hidden 2": mlp_forward(Pn, Xtr)[3]}
        feats_te = {"pixels": Xte, "hidden 1": mlp_forward(Pn, Xte)[1], "hidden 2": mlp_forward(Pn, Xte)[3]}
        accs = []
        for layer in ("pixels", "hidden 1", "hidden 2"):
            mu, sd = feats_tr[layer].mean(0), feats_tr[layer].std(0) + 1e-6
            W, b = train_logreg((feats_tr[layer] - mu) / sd, ytr, 10, steps=300, lr=0.02)
            a = logreg_acc(W, b, (feats_te[layer] - mu) / sd, yte); accs.append(a)
            print(f"   probe on {layer:9s} of the {net_name}: {feats_tr[layer].shape[1]:3d} features -> test accuracy {a:.3f}")
        results[net_name] = accs
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, results["trained network"], w, color=SERIES[0], label="probe on the trained network", edgecolor=SURFACE)
    ax.bar(x + w / 2, results["random network"], w, color=SERIES[1], label="probe on an untrained (random) network", edgecolor=SURFACE)
    ax.axhline(acc_model, color=INK_2, ls="--", lw=1, label=f"the trained MLP itself ({acc_model:.3f})")
    for xi, v in zip(x - w / 2, results["trained network"]):
        ax.annotate(f"{v:.3f}", (xi, v), ha="center", va="bottom", fontsize=8, color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(["pixels (64)", "hidden layer 1 (32)", "hidden layer 2 (32)"])
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("linear probe test accuracy"); ax.set_title("Linear decodability of the label by depth")
    ax.legend(loc="lower left")
    save(fig, NB, "linear_probes_depth")


def fig_activation_maximization():
    D = trained_digit_mlp(); P, Xte, yte = D["P"], D["Xte"], D["yte"]
    classes = np.arange(10)
    real_logits = np.array([mlp_forward(P, Xte[yte == c])[-1][:, c].mean() for c in classes])

    def total_variation(img):            # img: (B, 8, 8); returns per-image TV and its gradient
        dh = img[:, 1:, :] - img[:, :-1, :]; dw = img[:, :, 1:] - img[:, :, :-1]
        tv = np.abs(dh).sum((1, 2)) + np.abs(dw).sum((1, 2))
        g = np.zeros_like(img)
        sh, sw = np.sign(dh), np.sign(dw)
        g[:, 1:, :] += sh; g[:, :-1, :] -= sh; g[:, :, 1:] += sw; g[:, :, :-1] -= sw
        return tv, g

    def optimise(tv_weight, steps=300, lr=0.1, seed=5):
        r = np.random.default_rng(seed)
        wv = 0.1 * r.standard_normal((10, 64))
        params = {"w": wv}; state = {"w": (np.zeros_like(wv), np.zeros_like(wv))}
        for t in range(1, steps + 1):
            img = sigmoid(params["w"])
            g_obj, _ = mlp_input_grad(P, img, classes)
            _, g_tv = total_variation(img.reshape(10, 8, 8))
            g_img = -g_obj + tv_weight * g_tv.reshape(10, 64) + 0.1 * 2 * img / 64
            adam_step(params, {"w": g_img * img * (1 - img)}, state, lr, t)
        img = sigmoid(params["w"])
        _, logit = mlp_input_grad(P, img, classes)
        tv, _ = total_variation(img.reshape(10, 8, 8))
        return img, logit, tv

    img_free, logit_free, tv_free = optimise(0.0)
    img_reg, logit_reg, tv_reg = optimise(0.05)
    print("\n[fig_activation_maximization] class logit reached by gradient ascent on the input (8x8 digits MLP):")
    print("   class   real test mean   no regulariser   with TV penalty   TV(no reg)  TV(reg)")
    for c in classes:
        print(f"   {c:5d}   {real_logits[c]:14.2f}   {logit_free[c]:14.2f}   {logit_reg[c]:15.2f}   {tv_free[c]:9.1f}  {tv_reg[c]:7.1f}")
    print(f"   means: real {real_logits.mean():.2f}, unregularised {logit_free.mean():.2f}, regularised {logit_reg.mean():.2f}")
    fig, axes = plt.subplots(3, 10, figsize=(10, 3.6))
    for c in classes:
        axes[0, c].imshow(Xte[yte == c][0].reshape(8, 8), cmap=SEQ_CMAP, vmin=0, vmax=1)
        axes[1, c].imshow(img_free[c].reshape(8, 8), cmap=SEQ_CMAP, vmin=0, vmax=1)
        axes[2, c].imshow(img_reg[c].reshape(8, 8), cmap=SEQ_CMAP, vmin=0, vmax=1)
        axes[0, c].set_title(f"class {c}", fontsize=8)
    for ax, lab in zip(axes[:, 0], ["real digit", "ascent, no\nregulariser", "ascent with\nTV penalty"]):
        ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center", labelpad=28)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Activation maximization: what each class logit wants to see", fontsize=12, fontweight="bold")
    save(fig, NB, "activation_maximization")


# ----------------------------------------------------------------------------------------------
# 4. Adversarial examples: FGSM and PGD on a 2-D linear classifier; the linearity argument;
#    certified radius of randomized smoothing
# ----------------------------------------------------------------------------------------------
def fig_fgsm_pgd_2d():
    w = np.array([2.0, -1.0]); b = -0.2
    s = lambda v: w @ v + b
    x = np.array([0.30, 0.50]); eps = 0.10
    eta = eps * np.sign(w)
    x_adv = np.clip(x + eta, 0, 1)
    print("\n[fig_fgsm_pgd_2d] score s(x) = 2.0 x1 - 1.0 x2 - 0.2, class = sign(s); x = (0.30, 0.50), eps = 0.10")
    print(f"   s(x) = {s(x):+.3f} -> class 0 (negative side)")
    print(f"   FGSM perturbation eps * sign(w) = ({eta[0]:+.2f}, {eta[1]:+.2f}); x_adv = ({x_adv[0]:.2f}, {x_adv[1]:.2f}); s(x_adv) = {s(x_adv):+.3f} -> class 1")
    print(f"   change in score = {s(x_adv) - s(x):+.3f} = eps * ||w||_1 = {eps * np.abs(w).sum():.3f}")
    print(f"   smallest L2 perturbation that reaches the boundary: |s(x)| / ||w||_2 = {abs(s(x)) / np.linalg.norm(w):.4f}; the FGSM step has L2 norm {np.linalg.norm(eta):.4f}")
    # PGD: 4 steps of alpha = 2.5 eps / 4 from a random start, projected onto the box
    r = np.random.default_rng(7)
    alpha = 2.5 * eps / 4
    pt = np.clip(x + r.uniform(-eps, eps, 2), 0, 1); path = [pt.copy()]
    for _ in range(4):
        pt = pt + alpha * np.sign(w)                        # gradient of the loss has the sign of w here
        pt = np.clip(np.minimum(np.maximum(pt, x - eps), x + eps), 0, 1)
        path.append(pt.copy())
    path = np.array(path)
    print(f"   PGD (4 steps, alpha = {alpha:.3f}): start ({path[0, 0]:.3f}, {path[0, 1]:.3f}) -> end ({path[-1, 0]:.3f}, {path[-1, 1]:.3f}), s(end) = {s(path[-1]):+.3f}")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    xx = np.linspace(0, 0.7, 2)
    ax.fill_between(np.linspace(0, 0.7, 50), (2.0 * np.linspace(0, 0.7, 50) - 0.2), 1.0, color=SEQ_STEPS[0], alpha=0.6, label="class 0 region (s < 0)")
    ax.fill_between(np.linspace(0, 0.7, 50), 0.0, (2.0 * np.linspace(0, 0.7, 50) - 0.2), color="#fde5d9", alpha=0.8, label="class 1 region (s > 0)")
    ax.plot(xx, 2.0 * xx - 0.2, color=INK_2, lw=1.2, label="decision boundary s(x) = 0")
    box = plt.Rectangle((x[0] - eps, x[1] - eps), 2 * eps, 2 * eps, fill=False, ls="--", color=MUTED, lw=1)
    ax.add_patch(box)
    ax.plot(*x, "o", color=SERIES[0], ms=9, label="clean x")
    ax.annotate("", xy=x_adv, xytext=x, arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=2))
    ax.plot(*x_adv, "s", color=SERIES[1], ms=9, label="FGSM: x + eps sign(w)")
    ax.plot(path[:, 0], path[:, 1], "-", color=SERIES[2], lw=1.2)
    ax.plot(path[:, 0], path[:, 1], "^", color=SERIES[2], ms=7, label="PGD path (random start, 4 steps)")
    ax.annotate(f"s = {s(x):+.2f}", x, xytext=(-62, 6), textcoords="offset points", fontsize=9, color=SERIES[0])
    ax.annotate(f"s = {s(x_adv):+.2f}", x_adv, xytext=(8, -4), textcoords="offset points", fontsize=9, color=SERIES[1])
    ax.text(x[0] - eps, x[1] + eps + 0.012, "L-inf ball, eps = 0.1", fontsize=8, color=MUTED)
    ax.set_xlim(0.1, 0.7); ax.set_ylim(0.2, 0.8); ax.set_aspect("equal")
    ax.set_xlabel("feature x1"); ax.set_ylabel("feature x2"); ax.set_title("FGSM and PGD on a linear classifier")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "fgsm_pgd_2d")


def fig_linearity_argument():
    eps, mean_abs_w = 0.1, 0.01
    dims = np.array([2, 10, 100, 784, 3072, 150528])
    print(f"\n[fig_linearity_argument] eps = {eps}, average |w| = {mean_abs_w} per input dimension; score change eps * ||w||_1 = eps * n * mean|w|:")
    change = eps * dims * mean_abs_w
    r = np.random.default_rng(11)
    for n, c in zip(dims, change):
        w = r.standard_normal(n); w *= mean_abs_w / np.abs(w).mean()        # random weights with exactly that average size
        print(f"   n = {n:6d}: eps * ||w||_1 = {eps * np.abs(w).sum():8.3f} (formula {c:8.3f}); random-sign perturbation would give about {eps * np.abs(w @ np.sign(r.standard_normal(n))):.3f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(dims, change, "o-", color=SERIES[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of input dimensions n"); ax.set_ylabel("maximal change in the score (eps = 0.1, mean |w| = 0.01)")
    ax.set_title("The linearity argument: tiny per-pixel changes add up")
    ax.annotate(f"MNIST, n = 784\nchange = {change[3]:.2f}", (784, change[3]), xytext=(-95, 10), textcoords="offset points", fontsize=9, color=INK_2)
    ax.annotate(f"224x224 RGB, n = 150 528\nchange = {change[5]:.0f}", (150528, change[5]), xytext=(-150, -8), textcoords="offset points", fontsize=9, color=INK_2)
    save(fig, NB, "linearity_argument")


def fig_smoothing_radius():
    pA = np.linspace(0.5, 0.9995, 400)
    sigmas = [0.12, 0.25, 0.5, 1.0]
    print("\n[fig_smoothing_radius] certified L2 radius R = sigma * Phi^-1(p_A):")
    table_p = [0.6, 0.8, 0.9, 0.99, 0.99312]
    print("   p_A      Phi^-1(p_A)   " + "  ".join(f"sigma={s:<5}" for s in sigmas))
    for p in table_p:
        q = float(norm_ppf(p))
        print(f"   {p:7.5f}   {q:10.3f}   " + "  ".join(f"{s * q:10.3f}" for s in sigmas))
    cap = 0.001 ** (1 / 1000)
    print(f"   all n = 1000 samples agree, alpha = 0.001: Clopper-Pearson lower bound = alpha^(1/n) = {cap:.5f}, Phi^-1 = {float(norm_ppf(cap)):.3f} -> R = {float(norm_ppf(cap)):.2f} sigma")
    cap200 = 0.001 ** (1 / 200)
    print(f"   all n = 200 samples agree: lower bound {cap200:.5f}, Phi^-1 = {float(norm_ppf(cap200)):.3f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    q = norm_ppf(pA)
    for i, s in enumerate(sigmas):
        ax.plot(pA, s * q, color=SERIES[i], label=f"sigma = {s}")
    ax.axvline(cap, color=MUTED, ls=":", lw=1)
    ax.text(cap - 0.005, 2.3, "cap with n = 1000\nnoisy copies", fontsize=8, color=MUTED, ha="right")
    ax.set_xlabel("lower bound on p_A (probability of the top class under noise)"); ax.set_ylabel("certified L2 radius")
    ax.set_title("Randomized smoothing: certified radius vs vote margin"); ax.set_ylim(0, 3.2); ax.legend(loc="upper left")
    save(fig, NB, "smoothing_radius")


# ----------------------------------------------------------------------------------------------
# 5. OOD detection: MSP vs energy on simulated logits, with AUROC
# ----------------------------------------------------------------------------------------------
def fig_ood_scores_roc():
    r = np.random.default_rng(21)
    K, n = 10, 2000
    lg_id = r.normal(0.0, 1.0, (n, K)); lg_id[np.arange(n), r.integers(0, K, n)] += r.normal(7.0, 2.0, n)
    lg_ood = r.normal(0.0, 1.5, (n, K)); lg_ood[np.arange(n), r.integers(0, K, n)] += r.normal(2.5, 1.5, n)
    msp = lambda z: softmax(z).max(1)
    energy = lambda z: logsumexp(z)                              # negative energy at T = 1
    ent = lambda z: entropy(softmax(z))
    print("\n[fig_ood_scores_roc] simulated 10-class logits, 2000 in-distribution (ID) and 2000 OOD inputs")
    print("   sample rows: max logit | logsumexp (= -energy) | MSP | entropy")
    for name, z in (("ID ", lg_id[:3]), ("OOD", lg_ood[:3])):
        for row in z:
            print(f"   {name}  {row.max():6.2f}   {logsumexp(row):6.2f}   {softmax(row).max():.3f}   {entropy(softmax(row)):.3f}")
    a_msp, a_en, a_ent = auroc(msp(lg_id), msp(lg_ood)), auroc(energy(lg_id), energy(lg_ood)), auroc(-ent(lg_id), -ent(lg_ood))
    print(f"   fraction of OOD inputs with MSP > 0.9: {(msp(lg_ood) > 0.9).mean():.3f}; of ID inputs: {(msp(lg_id) > 0.9).mean():.3f}")
    print(f"   AUROC: MSP {a_msp:.3f} | negative energy {a_en:.3f} | negative entropy {a_ent:.3f}")
    print(f"   mean MSP: ID {msp(lg_id).mean():.3f}, OOD {msp(lg_ood).mean():.3f}; mean -energy: ID {energy(lg_id).mean():.2f}, OOD {energy(lg_ood).mean():.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(msp(lg_id), bins=30, range=(0, 1), color=SERIES[0], alpha=0.7, label="in-distribution")
    axes[0].hist(msp(lg_ood), bins=30, range=(0, 1), color=SERIES[1], alpha=0.7, label="OOD")
    axes[0].set_xlabel("maximum softmax probability"); axes[0].set_ylabel("count"); axes[0].set_title("MSP: saturates near 1"); axes[0].legend(loc="upper left")
    axes[1].hist(energy(lg_id), bins=30, color=SERIES[0], alpha=0.7, label="in-distribution")
    axes[1].hist(energy(lg_ood), bins=30, color=SERIES[1], alpha=0.7, label="OOD")
    axes[1].set_xlabel("negative energy = logsumexp(logits)"); axes[1].set_ylabel("count"); axes[1].set_title("Energy: keeps the logit magnitude"); axes[1].legend(loc="upper right")
    for i, (name, sp, sn, a) in enumerate((("MSP", msp(lg_id), msp(lg_ood), a_msp), ("negative energy", energy(lg_id), energy(lg_ood), a_en))):
        fpr, tpr = roc_curve(sp, sn)
        axes[2].plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUROC {a:.3f})")
    axes[2].plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
    axes[2].set_xlabel("false positive rate (OOD accepted as ID)"); axes[2].set_ylabel("true positive rate (ID accepted)"); axes[2].set_title("ROC curves"); axes[2].legend(loc="lower right")
    save(fig, NB, "ood_scores_roc")


# ----------------------------------------------------------------------------------------------
# 6. Calibration: reliability diagram and ECE before/after temperature scaling; entropy vs MI
# ----------------------------------------------------------------------------------------------
def calibration_stats(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, acc_b, conf_b, frac_b = 0.0, [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        f = m.mean()
        if f > 0:
            a, c = correct[m].mean(), conf[m].mean(); ece += f * abs(a - c)
        else:
            a = c = np.nan
        acc_b.append(a); conf_b.append(c); frac_b.append(f)
    return ece, np.array(acc_b), np.array(conf_b), np.array(frac_b), edges


def nll(probs, labels):
    return -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()


def fig_reliability_temperature():
    # tiny hand example first
    conf8 = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.65, 0.55, 0.50]); right8 = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    print("\n[fig_reliability_temperature] hand example: 8 predictions, 4 bins of width 0.25")
    print("   confidence:", conf8.tolist()); print("   correct   :", right8.tolist())
    edges4 = np.linspace(0, 1, 5); ece8 = 0.0
    for lo, hi in zip(edges4[:-1], edges4[1:]):
        m = (conf8 > lo) & (conf8 <= hi)
        if m.any():
            a, c = right8[m].mean(), conf8[m].mean(); ece8 += m.mean() * abs(a - c)
            print(f"   bin ({lo:.2f}, {hi:.2f}]: {m.sum()} predictions, accuracy {a:.3f}, mean confidence {c:.3f}, |gap| {abs(a - c):.3f}, weight {m.mean():.3f}")
    print(f"   ECE = {ece8:.4f}; overall accuracy {right8.mean():.3f}, mean confidence {conf8.mean():.3f}")

    # simulation: calibrated latent logits, model reports them 2.5x too sharp
    r = np.random.default_rng(31)
    K = 10
    def simulate(n):
        z = r.normal(0.0, 1.0, (n, K)); z[np.arange(n), r.integers(0, K, n)] += 2.2
        p_true = softmax(z)
        y = np.array([r.choice(K, p=p) for p in p_true])
        return z * 2.5, y
    z_val, y_val = simulate(5000); z_te, y_te = simulate(5000)
    Ts = np.linspace(0.5, 5.0, 451)
    nlls = np.array([nll(softmax(z_val / T), y_val) for T in Ts])
    T_opt = Ts[nlls.argmin()]
    p_before, p_after = softmax(z_te), softmax(z_te / T_opt)
    e0, acc0, conf0, frac0, edges = calibration_stats(p_before, y_te)
    e1, acc1, conf1, frac1, _ = calibration_stats(p_after, y_te)
    print(f"   simulation: 5000 validation + 5000 test predictions, logits 2.5x too sharp")
    print(f"   validation NLL at T = 1: {nll(softmax(z_val), y_val):.4f}; fitted T = {T_opt:.2f} with NLL {nlls.min():.4f}")
    print(f"   test accuracy {(p_before.argmax(1) == y_te).mean():.3f} (unchanged by T); mean confidence before {p_before.max(1).mean():.3f}, after {p_after.max(1).mean():.3f}")
    print(f"   test ECE before {e0:.4f}, after {e1:.4f}; test NLL before {nll(p_before, y_te):.4f}, after {nll(p_after, y_te):.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    centers, width = (edges[:-1] + edges[1:]) / 2, 1.0 / 15
    for ax, (acc, conf, frac, e, title) in zip(axes, ((acc0, conf0, frac0, e0, "before: raw logits"), (acc1, conf1, frac1, e1, f"after temperature scaling (T = {T_opt:.2f})"))):
        ok = frac > 0
        ax.bar(centers[ok], acc[ok], width=width, color=SERIES[0], edgecolor=SURFACE, label="accuracy in bin")
        ax.bar(centers[ok], (conf - acc)[ok], bottom=acc[ok], width=width, color=SERIES[1], alpha=0.45, edgecolor=SURFACE, label="gap to mean confidence")
        ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence (max softmax)"); ax.set_ylabel("accuracy")
        ax.set_title(f"{title}, ECE = {e:.3f}"); ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "reliability_temperature")


def fig_entropy_vs_mutual_information():
    cases = {
        "clear digit": np.array([[0.96, 0.02, 0.02]] * 5),
        "ambiguous digit (4 or 9)": np.array([[0.50, 0.50, 0.00]] * 5),
        "foreign input (a shoe)": np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]),
    }
    print("\n[fig_entropy_vs_mutual_information] 5 sampled predictions over 3 classes per input (MC dropout / ensemble members):")
    rows = {}
    for name, S in cases.items():
        pbar = S.mean(0)
        H_total, H_exp = entropy(pbar), entropy(S).mean()
        rows[name] = (H_total, H_exp, H_total - H_exp)
        print(f"   {name:26s}: mean prediction ({pbar[0]:.2f}, {pbar[1]:.2f}, {pbar[2]:.2f}); predictive entropy {H_total:.3f}; expected entropy {H_exp:.3f}; mutual information {H_total - H_exp:.3f}")
    print(f"   maximum possible entropy with 3 classes: ln 3 = {math.log(3):.3f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(3); w = 0.26
    for i, (lab, idx) in enumerate((("predictive entropy (total)", 0), ("expected entropy (aleatoric part)", 1), ("mutual information (epistemic part)", 2))):
        vals = [rows[k][idx] for k in cases]
        ax.bar(x + (i - 1) * w, vals, w, color=SERIES[i], edgecolor=SURFACE, label=lab)
    for xi, k in zip(x, cases):
        ax.annotate(f"MI = {rows[k][2]:.2f}", (xi + w, rows[k][2]), ha="center", va="bottom", fontsize=8, color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(list(cases)); ax.set_ylabel("nats"); ax.set_ylim(0, 1.25)
    ax.set_title("Two kinds of not knowing: entropy vs mutual information"); ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "entropy_vs_mutual_information")


# ----------------------------------------------------------------------------------------------
# 7. Fairness: the notebook's synthetic hiring data, shared vs group-specific thresholds
# ----------------------------------------------------------------------------------------------
def fig_fairness_thresholds():
    r = np.random.default_rng(42)
    n = 20000
    group = r.integers(0, 2, n)
    skill = r.normal(0.0, 1.0, n)
    test_score = skill + r.normal(0.0, 0.6, n)
    credential = skill - 0.8 * group + r.normal(0.0, 0.6, n)
    neighbourhood = group + r.normal(0.0, 0.7, n)
    hired = (skill - 0.5 * group + r.normal(0.0, 0.5, n) > 0.3).astype(int)
    qualified = (skill > 0.3).astype(int)
    X = np.stack([test_score, credential, neighbourhood], 1)
    print("\n[fig_fairness_thresholds] synthetic applicants (same generator as the notebook); sample rows: group, test score, credential, neighbourhood, hired, qualified")
    for i in range(6):
        print(f"   {group[i]}  {test_score[i]:6.2f}  {credential[i]:6.2f}  {neighbourhood[i]:6.2f}  {hired[i]}  {qualified[i]}")
    print(f"   base rate of 'hired': group 0 {hired[group == 0].mean():.3f}, group 1 {hired[group == 1].mean():.3f}; of true qualification: group 0 {qualified[group == 0].mean():.3f}, group 1 {qualified[group == 1].mean():.3f}")
    n_tr = 12000
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0)
    W, b = train_logreg((X[:n_tr] - mu) / sd, hired[:n_tr], 2, steps=600, lr=0.05, weight_decay=0.0)
    scores = softmax(((X[n_tr:] - mu) / sd) @ W + b)[:, 1]
    g, yh, yq = group[n_tr:], hired[n_tr:], qualified[n_tr:]

    def metrics(pred, truth):
        out = {}
        for gg in (0, 1):
            m = g == gg
            out[gg] = dict(sel=pred[m].mean(), tpr=pred[m & (truth == 1)].mean(), fpr=pred[m & (truth == 0)].mean(), acc=(pred[m] == truth[m]).mean())
        return out

    pred_shared = (scores >= 0.5).astype(int)
    m_h = metrics(pred_shared, yh); m_q = metrics(pred_shared, yq)
    print("   shared threshold 0.5, against HISTORICAL labels:")
    for gg in (0, 1):
        print(f"      group {gg}: selection {m_h[gg]['sel']:.3f}, TPR {m_h[gg]['tpr']:.3f}, FPR {m_h[gg]['fpr']:.3f}, accuracy {m_h[gg]['acc']:.3f}")
    print(f"      demographic parity gap {abs(m_h[0]['sel'] - m_h[1]['sel']):.3f}, equalized odds gap {max(abs(m_h[0]['tpr'] - m_h[1]['tpr']), abs(m_h[0]['fpr'] - m_h[1]['fpr'])):.3f}")
    print("   shared threshold 0.5, against TRUE qualification:")
    for gg in (0, 1):
        print(f"      group {gg}: selection {m_q[gg]['sel']:.3f}, TPR {m_q[gg]['tpr']:.3f}, FPR {m_q[gg]['fpr']:.3f}, accuracy {m_q[gg]['acc']:.3f}")
    # equal-opportunity threshold for group 1
    pos1 = np.sort(scores[(g == 1) & (yh == 1)])[::-1]
    k = int(np.ceil(m_h[0]["tpr"] * len(pos1)))
    t1 = pos1[min(max(k, 1), len(pos1)) - 1]
    pred_eo = np.where(g == 0, scores >= 0.5, scores >= t1).astype(int)
    e_h = metrics(pred_eo, yh); e_q = metrics(pred_eo, yq)
    print(f"   group-specific threshold for group 1 that matches group 0's TPR: {t1:.3f}")
    print("   after post-processing, against HISTORICAL labels:")
    for gg in (0, 1):
        print(f"      group {gg}: selection {e_h[gg]['sel']:.3f}, TPR {e_h[gg]['tpr']:.3f}, FPR {e_h[gg]['fpr']:.3f}, accuracy {e_h[gg]['acc']:.3f}")
    print(f"      demographic parity gap {abs(e_h[0]['sel'] - e_h[1]['sel']):.3f}, equalized odds gap {max(abs(e_h[0]['tpr'] - e_h[1]['tpr']), abs(e_h[0]['fpr'] - e_h[1]['fpr'])):.3f}")
    print("   after post-processing, against TRUE qualification:")
    for gg in (0, 1):
        print(f"      group {gg}: selection {e_q[gg]['sel']:.3f}, TPR {e_q[gg]['tpr']:.3f}, FPR {e_q[gg]['fpr']:.3f}, accuracy {e_q[gg]['acc']:.3f}")
    print(f"   overall accuracy against historical labels: {(pred_shared == yh).mean():.3f} -> {(pred_eo == yh).mean():.3f}; against true qualification: {(pred_shared == yq).mean():.3f} -> {(pred_eo == yq).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 1, 31)
    ax.hist(scores[(g == 0) & (yh == 1)], bins=bins, color=SERIES[0], alpha=0.65, label="group 0, historically hired")
    ax.hist(scores[(g == 1) & (yh == 1)], bins=bins, color=SERIES[1], alpha=0.65, label="group 1, historically hired")
    ax.axvline(0.5, color=INK_2, ls="--", lw=1.2, label="shared threshold 0.5")
    ax.axvline(t1, color=SERIES[1], ls=":", lw=1.6, label=f"group-1 threshold {t1:.3f}")
    ax.set_xlabel("model score"); ax.set_ylabel("count"); ax.set_title("Scores of the positives differ by group"); ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    names = ["selection\nrate", "TPR", "FPR"]
    x = np.arange(3); w = 0.2
    vals = [[m_h[0]["sel"], m_h[0]["tpr"], m_h[0]["fpr"]], [m_h[1]["sel"], m_h[1]["tpr"], m_h[1]["fpr"]],
            [e_h[1]["sel"], e_h[1]["tpr"], e_h[1]["fpr"]]]
    labs = ["group 0 (threshold 0.5)", "group 1 (threshold 0.5)", f"group 1 (threshold {t1:.2f})"]
    for i in range(3):
        ax.bar(x + (i - 1) * w, vals[i], w, color=SERIES[i], edgecolor=SURFACE, label=labs[i])
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("rate (against historical labels)")
    ax.set_title("Shared vs group-specific thresholds"); ax.legend(loc="upper right", fontsize=8); ax.set_ylim(0, 0.9)
    save(fig, NB, "fairness_thresholds")


def main():
    fig_shortcut_learning()
    fig_ig_path_integral()
    fig_linear_probes_depth()
    fig_activation_maximization()
    fig_fgsm_pgd_2d()
    fig_linearity_argument()
    fig_smoothing_radius()
    fig_ood_scores_roc()
    fig_reliability_temperature()
    fig_entropy_vs_mutual_information()
    fig_fairness_thresholds()


if __name__ == "__main__":
    main()
