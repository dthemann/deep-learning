"""Figures and worked-example numbers for the companion guide of notebook 01 (Mathematical Foundations).

Run:  python3 notebooks/figures/scripts/make_figures_01.py
Every number quoted in notebooks/01_mathematical_foundations.md is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os, time
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # single-threaded BLAS: reproducible timings for fig_vectorisation
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)
NB = "01"
np.set_printoptions(precision=4, suppress=True)


def header(title):
    print(f"\n=== {title} " + "=" * max(0, 80 - len(title)))


# ------------------------------------------------------------------------------------------------
# Section 1: linear algebra
# ------------------------------------------------------------------------------------------------
def fig_linear_map():
    header("fig_linear_map: a 2x2 matrix acting on four points")
    A = np.array([[2.0, 1.0],
                  [0.5, 1.5]])
    names = ["p1", "p2", "p3", "p4"]
    P = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])      # one point per row
    images = P @ A.T                                                       # A p for every row p
    print("A =\n", A)
    print(f"det(A) = {np.linalg.det(A):.4f}  (area scale factor)")
    for n, p, q in zip(names, P, images):
        print(f"  {n}: x = {p}  ->  A x = {q}   (dot of row 1 with x = {A[0] @ p:.2f}, row 2 = {A[1] @ p:.2f})")
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"singular values of A: {sv}  (semi-axes of the image of the unit circle); product = {sv.prod():.4f} = |det|")
    eigvals = np.linalg.eigvals(A).real                                    # both real for this A
    print(f"eigenvalues of A: {np.sort(eigvals)[::-1]}")
    lengths = np.linalg.norm(images, axis=1) / np.linalg.norm(P, axis=1)
    print(f"stretch factor |A x| / |x| per point: {lengths}  (between the two singular values)")

    ang = np.linspace(0, 2 * np.pi, 300)
    circle = np.stack([np.cos(ang), np.sin(ang)])
    ellipse = A @ circle
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot(circle[0], circle[1], color=MUTED, lw=1, ls="--")
    ax.plot(ellipse[0], ellipse[1], color=ORANGE, lw=1, ls="--")
    ax.scatter(P[:, 0], P[:, 1], color=BLUE, s=55, zorder=3, label="original points x")
    ax.scatter(images[:, 0], images[:, 1], color=ORANGE, marker="s", s=55, zorder=3, label="images A x")
    for p, q in zip(P, images):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.9, alpha=0.7))
    ax.annotate("p3 = (1, 1) -> (3, 2)", xy=images[2], xytext=(1.6, 2.55), fontsize=9, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-2.2, 3.6); ax.set_ylim(-1.2, 3.0)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("A 2 x 2 matrix moves every point: x -> A x (dashed: unit circle and its image)")
    ax.legend(loc="upper left")
    save(fig, NB, "linear_map_2x2")


def fig_pca_tiny():
    header("fig_pca_tiny: PCA of a 4 x 3 data matrix via the SVD")
    X = np.array([[2.0, 1.0, 0.5],
                  [3.0, 2.0, 1.5],
                  [4.0, 2.0, 1.0],
                  [5.0, 3.0, 2.0]])
    mean = X.mean(axis=0)
    Xc = X - mean
    print("X (4 samples x 3 features) =\n", X)
    print("column means =", mean)
    print("centred X_c =\n", Xc)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    print("singular values sigma =", s)
    print("sigma^2 =", s ** 2, "| sum =", (s ** 2).sum().round(4), "= squared Frobenius norm of X_c =", (Xc ** 2).sum().round(4))
    print("variance along each PC = sigma^2 / (N - 1) =", (s ** 2 / 3))
    print("explained variance fraction =", explained, "| cumulative =", np.cumsum(explained))
    print("PC1 direction (row 1 of V^T) =", Vt[0], "| PC2 =", Vt[1])
    scores = Xc @ Vt.T
    print("scores Z = X_c V (rows = samples, columns = PC1, PC2, PC3) =\n", scores)
    print("check: scores equal U * sigma:", np.allclose(scores, U * s))
    for k in [1, 2]:
        Xk = scores[:, :k] @ Vt[:k]
        err = np.linalg.norm(Xc - Xk) / np.linalg.norm(Xc)
        print(f"rank-{k} reconstruction: relative error {err:.4f} | sqrt(1 - explained[:k]) = {np.sqrt(1 - explained[:k].sum()):.4f}")
    X1 = scores[:, :1] @ Vt[:1] + mean
    print("rank-1 reconstruction + mean (compare with X) =\n", X1.round(3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.axhline(0, color=AXIS, lw=0.8); ax1.axvline(0, color=AXIS, lw=0.8)
    ax1.scatter(scores[:, 0], scores[:, 1], color=BLUE, s=60, zorder=3)
    for i, (a, b) in enumerate(scores[:, :2]):
        ax1.annotate(f"sample {i + 1}", (a, b), xytext=(7, -3 if b > 0 else -12), textcoords="offset points", fontsize=9, color=INK_2)
    ax1.set_xlabel("score on PC 1"); ax1.set_ylabel("score on PC 2")
    ax1.set_xlim(-2.5, 2.7); ax1.set_ylim(-0.9, 0.9)
    ax1.set_title("The four samples in the PCA basis (PC 1 carries 95% of the variance)")
    bar(ax2, ["PC 1", "PC 2", "PC 3"], explained, label_fmt="{:.3f}")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("fraction of variance explained"); ax2.set_title("Variance explained per component")
    save(fig, NB, "pca_tiny_svd")


# ------------------------------------------------------------------------------------------------
# Section 2: calculus
# ------------------------------------------------------------------------------------------------
def fig_derivative_fd():
    header("fig_derivative_fd: tangent line and finite-difference errors for f(x) = x^3 - 2x at x0 = 1")
    f = lambda x: x ** 3 - 2 * x
    df = lambda x: 3 * x ** 2 - 2
    x0 = 1.0
    print(f"f(x0) = {f(x0):.4f} | f'(x0) = {df(x0):.4f} | f''(x0)/2 = {6 * x0 / 2:.1f}")
    print("linear approximation f(x0) + f'(x0) h versus the truth:")
    for h in [0.5, 0.1, 0.01]:
        lin = f(x0) + df(x0) * h
        print(f"  h = {h:<5} f(x0 + h) = {f(x0 + h):.6f} | linear = {lin:.6f} | remainder = {f(x0 + h) - lin:.6f} | remainder / h^2 = {(f(x0 + h) - lin) / h ** 2:.3f}")
    hs = np.logspace(-12, -1, 12)
    fwd = np.abs((f(x0 + hs) - f(x0)) / hs - df(x0))
    cen = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - df(x0))
    print("step h       forward error   central error")
    for h, a, b in zip(hs, fwd, cen):
        print(f"  {h:8.0e}   {a:12.2e}   {b:12.2e}")
    print(f"best central step: h = {hs[np.argmin(cen)]:.0e} with error {cen.min():.1e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.linspace(0.0, 2.0, 200)
    ax1.plot(xs, f(xs), color=BLUE, label="$f(x) = x^3 - 2x$")
    ax1.plot(xs, f(x0) + df(x0) * (xs - x0), color=ORANGE, ls="--", label="tangent at $x_0 = 1$, slope 1")
    ax1.scatter([x0], [f(x0)], color=INK, s=40, zorder=3)
    ax1.annotate("$(1, -1)$", (x0, f(x0)), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK_2)
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$f(x)$"); ax1.set_title("The derivative is the slope of the best line"); ax1.legend()
    ax2.loglog(hs, fwd, "o-", color=BLUE, label="forward difference")
    ax2.loglog(hs, cen, "s-", color=ORANGE, label="central difference")
    ax2.loglog(hs, 2.2e-16 / hs, ":", color=MUTED, lw=1, label=r"rounding noise $\varepsilon / h$")
    ax2.set_xlabel("step size $h$"); ax2.set_ylabel("absolute error of $f'(1)$")
    ax2.set_title("Finite differences: truncation versus rounding"); ax2.legend(fontsize=8)
    save(fig, NB, "derivative_finite_differences")


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def fig_softmax_jacobian():
    header("fig_softmax_jacobian: softmax of z = [1, 2, 0.5, -1], its Jacobian and the cross-entropy gradient")
    z = np.array([1.0, 2.0, 0.5, -1.0])
    p = softmax(z)
    print("z =", z)
    print("exp(z) =", np.exp(z), "| sum =", np.exp(z).sum().round(4))
    print("p = softmax(z) =", p, "| sum =", p.sum().round(4))
    J = np.diag(p) - np.outer(p, p)
    print("Jacobian J = diag(p) - p p^T =\n", J)
    print("row sums of J:", J.sum(axis=1).round(10), "| symmetric:", np.allclose(J, J.T))
    # finite-difference check
    h = 1e-6
    J_num = np.stack([(softmax(z + h * e) - softmax(z - h * e)) / (2 * h) for e in np.eye(4)], axis=1)
    print(f"max |J - finite difference| = {np.abs(J - J_num).max():.1e}")
    y = 1
    ce = -np.log(p[y])
    print(f"true class y = {y} (0-based): cross-entropy -log p_y = {ce:.4f}")
    onehot = np.eye(4)[y]
    print("gradient of cross-entropy w.r.t. z = p - y_onehot =", p - onehot)
    ce_grad_num = np.array([(-np.log(softmax(z + h * e)[y]) + np.log(softmax(z - h * e)[y])) / (2 * h) for e in np.eye(4)])
    print(f"max |(p - y) - finite difference| = {np.abs(p - onehot - ce_grad_num).max():.1e}")
    z_conf = np.array([10.0, 0.0, 0.0, 0.0])
    p_conf = softmax(z_conf); J_conf = np.diag(p_conf) - np.outer(p_conf, p_conf)
    print(f"saturation: for z = {z_conf}, p = {p_conf.round(5)}, largest |J| entry = {np.abs(J_conf).max():.2e}")

    fig, ax = plt.subplots(figsize=(6, 4.8))
    m = np.abs(J).max()
    im = ax.imshow(J, cmap=DIV_CMAP, vmin=-m, vmax=m)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"$z_{i + 1}$" for i in range(4)]); ax.set_yticklabels([f"$p_{i + 1}$" for i in range(4)])
    ax.set_xlabel("logit being nudged"); ax.set_ylabel("probability that changes"); ax.grid(False)
    ax.set_title("Softmax Jacobian for z = [1, 2, 0.5, -1]")
    ax.annotate(f"{J[1, 1]:+.3f} = $p_2(1 - p_2)$", (1, 1), ha="center", va="center", fontsize=9, color=SURFACE)
    ax.annotate(f"{J[0, 1]:+.3f} = $-p_1 p_2$", (1, 0), ha="center", va="center", fontsize=9, color=SURFACE)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\partial p_i / \partial z_j$")
    save(fig, NB, "softmax_jacobian")


# ------------------------------------------------------------------------------------------------
# Section 3: optimisation
# ------------------------------------------------------------------------------------------------
def gradient_descent(grad, theta0, lr, n_steps, stop_above=1e12):
    theta = np.array(theta0, dtype=np.float64); path = [theta.copy()]
    for _ in range(n_steps):
        theta = theta - lr * grad(theta); path.append(theta.copy())
        if np.abs(theta).max() > stop_above:
            break
    return np.array(path)


def fig_gd_learning_rates():
    header("fig_gd_learning_rates: gradient descent on J(theta) = 5 theta^2 (curvature lambda = 10)")
    lam = 10.0
    grad = lambda th: lam * th
    theta0 = 1.0
    lrs = [0.02, 0.15, 0.21]
    paths = {lr: gradient_descent(grad, [theta0], lr, 12)[:, 0] for lr in lrs}
    print(f"update: theta <- theta - lr * {lam:.0f} theta = (1 - {lam:.0f} lr) theta ; stability needs lr < 2/{lam:.0f} = {2 / lam}")
    print("step  " + "".join(f"lr={lr:<5} (factor {1 - lr * lam:+.2f})   " for lr in lrs))
    for t in range(7):
        print(f"  {t:<4}" + "".join(f"{paths[lr][t]:>+12.4f}             " for lr in lrs))
    for lr in lrs:
        print(f"lr = {lr}: theta after 12 steps = {paths[lr][12]:+.5f}, loss = {5 * paths[lr][12] ** 2:.3e}")

    fig, ax = plt.subplots()
    for lr, color, mk in zip(lrs, SERIES[:3], ["o", "s", "^"]):
        ax.plot(range(13), paths[lr], marker=mk, color=color, lw=1.5, ms=6, label=f"lr = {lr} (factor {1 - lr * lam:+.2f} per step)")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylim(-3.2, 3.2); ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\theta_t$")
    ax.set_title(r"Gradient descent on $J(\theta) = 5\theta^2$: three learning rates")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "gd_learning_rates_1d")


def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])


def fig_rosenbrock_and_momentum():
    header("fig_rosenbrock_and_momentum: curved valley and ill-conditioning")
    start = [-1.0, 1.5]
    H_min = np.array([[802.0, -400.0], [-400.0, 200.0]])
    ev = np.linalg.eigvalsh(H_min)
    print(f"Rosenbrock Hessian at the minimum (1, 1): eigenvalues {ev}  -> condition number {ev.max() / ev.min():.0f}, stability bound lr < 2/{ev.max():.0f} = {2 / ev.max():.2e}")
    print(f"start {start}: loss {rosenbrock(start):.3f}, gradient {rosenbrock_grad(start)}")
    N_STEPS = 3000
    paths = {}
    for lr in [5e-4, 1e-3, 3e-3]:
        path = gradient_descent(rosenbrock_grad, start, lr, N_STEPS, stop_above=1e3)
        paths[lr] = path
        losses = [rosenbrock(p) for p in path]
        marks = [t for t in [10, 100, 1000, 3000] if t < len(path)] or [len(path) - 1]
        print(f"lr = {lr:g}: steps run {len(path) - 1}; final point {path[-1].round(4)}, distance to (1,1) {np.linalg.norm(path[-1] - 1):.4f}; "
              + "loss at steps " + ", ".join(f"{t}: {losses[t]:.3e}" for t in marks))

    kappa = 50.0
    A = np.diag([1.0, kappa]); loss_q = lambda th: 0.5 * th @ A @ th; grad_q = lambda th: A @ th
    th0 = np.array([-2.0, 0.5]); lr_q = 1 / kappa; beta = 0.9
    p_gd = gradient_descent(grad_q, th0, lr_q, 300)
    theta, v, p_mom = th0.copy(), np.zeros(2), [th0.copy()]
    for _ in range(300):
        v = beta * v + grad_q(theta); theta = theta - lr_q * v; p_mom.append(theta.copy())
    p_mom = np.array(p_mom)
    l_gd = np.array([loss_q(p) for p in p_gd]); l_mom = np.array([loss_q(p) for p in p_mom])
    print(f"quadratic with condition number {kappa:.0f}, lr = 1/lambda_max = {lr_q}: per-step factor on the slow axis = {1 - lr_q:.2f}, "
          f"momentum beta = {beta} contracts by sqrt(beta) = {np.sqrt(beta):.3f}")
    for t in [0, 50, 100, 200, 300]:
        print(f"  step {t:>3}: GD loss {l_gd[t]:.2e} | momentum loss {l_mom[t]:.2e}")
    first = int(np.argmax(l_mom <= l_gd[300]))
    print(f"momentum reaches GD's 300-step loss after {first} steps")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    g1, g2 = np.meshgrid(np.linspace(-1.6, 1.6, 200), np.linspace(-0.6, 2.0, 200))
    Z = np.log10(rosenbrock(np.stack([g1, g2])) + 1e-3)
    ax1.contour(g1, g2, Z, levels=20, cmap=SEQ_CMAP, alpha=0.6, linewidths=0.8)
    for (lr_r, path), color, lw in zip(paths.items(), SERIES[:3], [2.8, 1.2, 1.4]):
        diverged = len(path) < N_STEPS + 1
        shown = path[:3] if diverged else path                      # a diverging run leaves the frame at once
        ax1.plot(shown[:, 0], shown[:, 1], color=color, lw=lw, label=f"lr = {lr_r:g}" + (" (diverges after 6 steps)" if diverged else ""))
    ax1.scatter([1], [1], marker="*", s=180, color=YELLOW, edgecolors=INK, zorder=4)
    ax1.scatter(*start, color=INK, s=30, zorder=4)
    ax1.annotate("minimum (1, 1)", (1, 1), xytext=(0.15, 1.55), fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax1.set_xlim(-1.6, 1.6); ax1.set_ylim(-0.6, 2.0); ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")
    ax1.set_title("Rosenbrock valley: 3000 steps of gradient descent"); ax1.legend(fontsize=8, loc="lower right")
    ax2.semilogy(l_gd, color=BLUE, label=f"gradient descent, lr = {lr_q}")
    ax2.semilogy(l_mom, color=ORANGE, label=f"momentum, lr = {lr_q}, beta = {beta}")
    ax2.set_xlabel("step"); ax2.set_ylabel("loss (log scale)")
    ax2.set_title(f"Quadratic with condition number {kappa:.0f}: momentum wins"); ax2.legend(fontsize=8)
    save(fig, NB, "rosenbrock_and_momentum")


# ------------------------------------------------------------------------------------------------
# Section 4: probability and information theory
# ------------------------------------------------------------------------------------------------
def fig_dice():
    header("fig_dice: expectation and variance of a fair and a loaded die")
    faces = np.arange(1, 7)
    fair = np.full(6, 1 / 6)
    loaded = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    for name, p in [("fair", fair), ("loaded", loaded)]:
        E = faces @ p; E2 = (faces ** 2) @ p; var = ((faces - E) ** 2) @ p
        print(f"{name:<7} p = {p.round(4)} | E[X] = {E:.4f} | E[X^2] = {E2:.4f} | Var = E[X^2] - E[X]^2 = {E2 - E ** 2:.4f} (direct {var:.4f}) | std = {np.sqrt(var):.4f}")
    samples = rng.choice(faces, size=10_000, p=fair)
    for n in [10, 100, 1000, 10000]:
        print(f"  running mean of the first {n:>5} fair-die rolls: {samples[:n].mean():.4f}   (expected error about {np.sqrt(35 / 12 / n):.3f})")
    fig, ax = plt.subplots()
    w = 0.38
    ax.bar(faces - w / 2, fair, width=w, color=BLUE, edgecolor=SURFACE, label="fair die, E[X] = 3.5, Var = 2.92")
    ax.bar(faces + w / 2, loaded, width=w, color=ORANGE, edgecolor=SURFACE, label="loaded die, E[X] = 4.5, Var = 3.25")
    ax.set_xticks(faces); ax.set_xlabel("face"); ax.set_ylabel("probability")
    ax.set_title("Two dice: the probability mass functions behind E[X] and Var[X]"); ax.legend()
    save(fig, NB, "dice_expectation_variance")


def fig_bayes_screening():
    header("fig_bayes_screening: disease screening with Bayes' rule")
    prev, sens, spec = 0.01, 0.90, 0.95
    N = 10_000
    sick = N * prev; healthy = N - sick
    TP = sick * sens; FN = sick - TP; FP = healthy * (1 - spec); TN = healthy - FP
    print(f"population {N:,}: prevalence {prev}, sensitivity {sens}, specificity {spec}")
    print(f"  sick {sick:.0f}: true positives {TP:.0f}, false negatives {FN:.0f}")
    print(f"  healthy {healthy:.0f}: false positives {FP:.0f}, true negatives {TN:.0f}")
    print(f"  all positives = {TP + FP:.0f}; P(disease | positive) = {TP:.0f} / {TP + FP:.0f} = {TP / (TP + FP):.4f}")
    p_pos = sens * prev + (1 - spec) * (1 - prev)
    print(f"  Bayes directly: p(+) = {p_pos:.4f}; p(D|+) = {sens * prev:.4f} / {p_pos:.4f} = {sens * prev / p_pos:.4f}")
    print(f"  P(disease | negative) = {FN:.0f} / {FN + TN:.0f} = {FN / (FN + TN):.5f}")
    prevs = np.logspace(-4, np.log10(0.5), 200)
    post = sens * prevs / (sens * prevs + (1 - spec) * (1 - prevs))
    for pv in [0.001, 0.01, 0.1, 0.5]:
        print(f"  prevalence {pv:<6} -> P(disease | positive) = {sens * pv / (sens * pv + (1 - spec) * (1 - pv)):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.barh(["test positive"], [TP], color=BLUE, label=f"true positives ({TP:.0f})")
    ax1.barh(["test positive"], [FP], left=[TP], color=ORANGE, label=f"false positives ({FP:.0f})")
    ax1.set_xlabel("people out of 10,000"); ax1.set_title("Who tests positive when 1% are sick?"); ax1.legend(loc="lower right")
    ax1.set_ylim(-0.8, 0.8)
    ax2.semilogx(prevs, post, color=BLUE)
    ax2.scatter([prev], [TP / (TP + FP)], color=INK, s=40, zorder=3)
    ax2.annotate(f"prevalence 1%: {TP / (TP + FP):.1%}", (prev, TP / (TP + FP)), xytext=(0.0012, 0.45), fontsize=9, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=AXIS))
    ax2.set_xlabel("prevalence of the disease (log scale)"); ax2.set_ylabel("P(disease | positive test)"); ax2.set_ylim(0, 1)
    ax2.set_title("The same test, different base rates")
    save(fig, NB, "bayes_screening")


def entropy(p):
    p = np.asarray(p, float); nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz])) + 0.0


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); nz = p > 0
    return -np.sum(p[nz] * np.log(q[nz]))


def fig_entropy_kl():
    header("fig_entropy_kl: entropy, cross-entropy and KL for two 3-outcome distributions")
    p = np.array([0.7, 0.2, 0.1]); q = np.array([0.5, 0.3, 0.2])
    print("p =", p, "| q =", q)
    print("surprise -log p_k =", -np.log(p), "| -log q_k =", -np.log(q))
    print(f"H(p) = {entropy(p):.4f} nats | H(q) = {entropy(q):.4f} nats | uniform over 3 = log 3 = {np.log(3):.4f} nats")
    print(f"H(p, q) = {cross_entropy(p, q):.4f} | KL(p||q) = H(p,q) - H(p) = {cross_entropy(p, q) - entropy(p):.4f}")
    print(f"H(q, p) = {cross_entropy(q, p):.4f} | KL(q||p) = {cross_entropy(q, p) - entropy(q):.4f}  (different: KL is not symmetric)")
    print(f"one-hot label on class 1 against q: cross-entropy = -log q_1 = {-np.log(q[0]):.4f} nats")
    print(f"entropy of a fair coin = {entropy([0.5, 0.5]):.4f} nats = {entropy([0.5, 0.5]) / np.log(2):.3f} bit; of a 90/10 coin = {entropy([0.9, 0.1]):.4f} nats")
    phis = np.linspace(0.005, 0.995, 300); p0 = 0.2
    kl_pq = np.array([cross_entropy([p0, 1 - p0], [f, 1 - f]) - entropy([p0, 1 - p0]) for f in phis])
    kl_qp = np.array([cross_entropy([f, 1 - f], [p0, 1 - p0]) - entropy([f, 1 - f]) for f in phis])
    for f in [0.05, 0.2, 0.5, 0.8]:
        i = np.argmin(np.abs(phis - f))
        print(f"  Bernoulli(0.2) vs Bernoulli({f}): KL(p||q) = {kl_pq[i]:.4f}, KL(q||p) = {kl_qp[i]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(3); w = 0.38
    ax1.bar(x - w / 2, p, width=w, color=BLUE, edgecolor=SURFACE, label=f"p (true), H = {entropy(p):.3f} nats")
    ax1.bar(x + w / 2, q, width=w, color=ORANGE, edgecolor=SURFACE, label=f"q (model), H(p, q) = {cross_entropy(p, q):.3f} nats")
    ax1.set_xticks(x); ax1.set_xticklabels(["class 1", "class 2", "class 3"]); ax1.set_ylabel("probability")
    ax1.set_title(f"Two distributions: KL(p||q) = {cross_entropy(p, q) - entropy(p):.3f} nats"); ax1.legend(fontsize=8)
    ax2.plot(phis, kl_pq, color=BLUE, label="KL(Bernoulli(0.2) || Bernoulli(q))")
    ax2.plot(phis, kl_qp, color=ORANGE, ls="--", label="KL(Bernoulli(q) || Bernoulli(0.2))")
    ax2.axvline(p0, color=AXIS, lw=0.8); ax2.set_ylim(0, 2.5)
    ax2.set_xlabel("parameter q of the second distribution"); ax2.set_ylabel("divergence (nats)")
    ax2.set_title("KL divergence is zero only at q = 0.2 and is asymmetric"); ax2.legend(fontsize=8)
    save(fig, NB, "entropy_cross_entropy_kl")


# ------------------------------------------------------------------------------------------------
# Section 5: numerical computation
# ------------------------------------------------------------------------------------------------
def fig_float_precision():
    header("fig_float_precision: float16 versus float32 rounding, and the one-pass variance formula")
    for name, dt in [("float16", np.float16), ("float32", np.float32), ("float64", np.float64)]:
        fi = np.finfo(dt)
        print(f"{name}: mantissa bits {fi.nmant}, exponent bits {fi.iexp}, eps {fi.eps:.1e}, max {fi.max:.3e}, smallest normal {fi.smallest_normal:.1e}")
    print("bfloat16 (by definition): mantissa bits 7, exponent bits 8, eps 2^-7 = 7.8e-03, max 3.39e+38")
    print("value        float16          rel. error   float32            rel. error")
    with np.errstate(over="ignore"):
        for x in [0.1, 1 / 3, 1234.5678, 65504.0, 70000.0]:
            x16, x32 = np.float16(x), np.float32(x)
            e16 = abs(float(x16) - x) / x if np.isfinite(x16) else float("inf")
            e32 = abs(float(x32) - x) / x
            print(f"{x:<12.6g} {float(x16):<16.10g} {e16:<12.1e} {float(x32):<18.10g} {e32:<12.1e}")
    print(f"float16: 1 + 1/1024 = {np.float16(1) + np.float16(1 / 1024)} | 1 + 1/2048 = {np.float16(1) + np.float16(1 / 2048)}")
    print(f"float32: 1e8 + 1 = {np.float32(1e8) + np.float32(1):.1f} (spacing at 1e8 is {np.spacing(np.float32(1e8)):.0f})")
    means = np.array([1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    rows = []
    print("mean of data   one-pass var (float32)   two-pass var (float32)   true var 1.0")
    noise = rng.standard_normal(10_000)
    noise = (noise - noise.mean()) / noise.std()          # exactly mean 0, variance 1 in float64
    for m in means:
        d = (m + noise).astype(np.float32)
        one = float(np.mean(d ** 2) - np.mean(d) ** 2)
        two = float(np.mean((d - d.mean()) ** 2))
        rows.append((one, two))
        print(f"{m:<14.0f} {one:<24.4f} {two:<24.4f}")
    rows = np.array(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.logspace(-3, 5, 400)
    for dt, color, name in [(np.float16, SERIES[0], "float16"), (np.float32, SERIES[1], "float32"), (np.float64, SERIES[2], "float64")]:
        with np.errstate(over="ignore"):
            sp = np.array([float(np.spacing(dt(v))) if np.isfinite(dt(v)) else np.nan for v in xs])
        ax1.loglog(xs, sp, color=color, label=name)
    ax1.axvline(65504, color=AXIS, lw=0.8)
    ax1.annotate("float16 max 65,504", (65504, 1e-9), xytext=(300, 3e-10), fontsize=8, color=INK_2)
    ax1.set_xlabel("magnitude of the number"); ax1.set_ylabel("gap to the next representable number")
    ax1.set_title("Floating-point gaps grow with magnitude"); ax1.legend(fontsize=8, loc="upper left")
    ax2.semilogx(means, rows[:, 0], "o-", color=BLUE, label="one-pass  E[x^2] - E[x]^2")
    ax2.semilogx(means, rows[:, 1], "s-", color=ORANGE, label="two-pass  E[(x - mean)^2]")
    ax2.axhline(1.0, color=AXIS, lw=0.8)
    ax2.set_xlabel("mean of the data (true variance is 1)"); ax2.set_ylabel("variance computed in float32")
    ax2.set_title("Catastrophic cancellation in the one-pass variance"); ax2.legend(fontsize=8, loc="lower left")
    save(fig, NB, "float_precision_cancellation")


def fig_vectorisation():
    header("fig_vectorisation: Python loop versus NumPy for a dot product")
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    t_loop, t_np, t_sum = [], [], []

    def dot_loop(u, v):
        acc = 0.0
        for i in range(len(u)):
            acc += u[i] * v[i]
        return acc

    def best_of(fn, repeats):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts)

    print("n           loop (ms)     np.sum(u * v) (ms)   u @ v (ms)   speed-up of u @ v")
    for n in sizes:
        u, v = rng.standard_normal(n), rng.standard_normal(n)
        tl = best_of(lambda: dot_loop(u, v), 2 if n >= 100_000 else 5)
        ts = best_of(lambda: np.sum(u * v), 20)
        tn = best_of(lambda: u @ v, 20)
        t_loop.append(tl); t_np.append(tn); t_sum.append(ts)
        print(f"{n:<11,} {tl * 1e3:<13.3f} {ts * 1e3:<20.4f} {tn * 1e3:<12.4f} {tl / tn:,.0f}x")
    print("(timings depend on the machine; the ratio is the message)")
    bcast = ((np.arange(4)[:, None] - np.arange(3)[None, :]) ** 2)
    print("broadcasting example: (4,1) - (1,3) -> shape", bcast.shape, "\n", bcast)

    fig, ax = plt.subplots()
    ax.loglog(sizes, np.array(t_loop) * 1e3, "o-", color=BLUE, label="Python for-loop")
    ax.loglog(sizes, np.array(t_sum) * 1e3, "^-", color=ORANGE, label="NumPy  np.sum(u * v)")
    ax.loglog(sizes, np.array(t_np) * 1e3, "s-", color=AQUA, label="NumPy  u @ v  (BLAS)")
    ax.annotate(f"{t_loop[-1] / t_np[-1]:,.0f}x faster at n = 1,000,000", (sizes[-1], t_np[-1] * 1e3), xytext=(-150, 25),
                textcoords="offset points", fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xlabel("vector length n"); ax.set_ylabel("time (ms)")
    ax.set_title("Dot product of two vectors: loop versus vectorised"); ax.legend()
    save(fig, NB, "vectorisation_speed")


def main():
    fig_linear_map()
    fig_pca_tiny()
    fig_derivative_fd()
    fig_softmax_jacobian()
    fig_gd_learning_rates()
    fig_rosenbrock_and_momentum()
    fig_dice()
    fig_bayes_screening()
    fig_entropy_kl()
    fig_float_precision()
    fig_vectorisation()


if __name__ == "__main__":
    main()
