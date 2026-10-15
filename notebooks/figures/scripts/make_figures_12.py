"""Figures for the companion guide of Notebook 12 · GANs and Diffusion Models.

Everything here is numpy + matplotlib (plus scikit-learn's bundled ``make_moons`` for a 2-D toy
dataset). Where the notebook trains a network, the guide uses a problem small enough for the exact
answer to be computed instead: the analytic optimal discriminator, a logistic-regression
discriminator fitted by Newton's method, the exact noise predictor of a 1-D Gaussian mixture (so that
ancestral sampling, DDIM and classifier-free guidance can be run without PyTorch), and the exact
marginal velocity field of a two-point flow-matching problem.

Run:  python3 notebooks/figures/scripts/make_figures_12.py
Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

from sklearn.datasets import make_moons

NB = "12"
np.set_printoptions(precision=4, suppress=True)
LOG2 = np.log(2.0)


def banner(title):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)


# ----------------------------------------------------------------------------------------------
# Shared helpers: densities and diffusion schedules (the notebook's conventions, in numpy)
# ----------------------------------------------------------------------------------------------
def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def linear_betas(T, beta_start=1e-4, beta_end=None):
    """Ho et al.'s linear schedule; beta_end rescaled so that sum(beta) ~ 10 for any T (as in the notebook)."""
    if beta_end is None:
        beta_end = 0.02 * 1000.0 / T
    return np.linspace(beta_start, beta_end, T)


def cosine_alpha_bar(u, s=0.008):
    """alpha_bar(t/T) of the cosine schedule (Nichol and Dhariwal, 2021)."""
    return np.cos((u + s) / (1 + s) * np.pi / 2) ** 2 / np.cos(s / (1 + s) * np.pi / 2) ** 2


class Schedule:
    """Coefficients of a DDPM with T steps; code index t = 0 ... T-1 is the paper's t = 1 ... T."""

    def __init__(self, betas):
        self.T = len(betas)
        self.betas = np.asarray(betas, dtype=float)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.alpha_bar_prev = np.concatenate([[1.0], self.alpha_bar[:-1]])
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)


class Mixture1D:
    """A 1-D Gaussian mixture p_data with the exact noise predictor epsilon*(x_t, t) of its diffusion."""

    def __init__(self, w, mu, sigma):
        self.w, self.mu, self.sigma = map(np.asarray, (w, mu, sigma))

    def pdf(self, x):
        return sum(wk * npdf(x, mk, sk) for wk, mk, sk in zip(self.w, self.mu, self.sigma))

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        return self.mu[k] + self.sigma[k] * rng.normal(size=n), k

    def eps_star(self, x, ab, k=None):
        """Exact epsilon for q_t: mixture (k=None) or the single component k (class-conditional)."""
        m = np.sqrt(ab) * self.mu
        v = ab * self.sigma ** 2 + (1.0 - ab)
        if k is not None:
            return np.sqrt(1.0 - ab) * (x - m[k]) / v[k]
        comp = np.stack([wk * npdf(x, mk, np.sqrt(vk)) for wk, mk, vk in zip(self.w, m, v)])
        r = comp / comp.sum(0)
        score = sum(rk * (-(x - mk) / vk) for rk, mk, vk in zip(r, m, v))
        return -np.sqrt(1.0 - ab) * score


def ddpm_sample_1d(eps_fn, sched, x_T, rng):
    """Ancestral sampling exactly as in the notebook's ddpm_sample (sigma_t^2 = beta_tilde_t)."""
    x = x_T.copy()
    for t in reversed(range(sched.T)):
        eps = eps_fn(x, sched.alpha_bar[t])
        mean = (x - sched.betas[t] / np.sqrt(1 - sched.alpha_bar[t]) * eps) / np.sqrt(sched.alphas[t])
        x = mean + np.sqrt(sched.posterior_var[t]) * rng.normal(size=x.shape) if t > 0 else mean
    return x


def ddim_sample_1d(eps_fn, sched, x_T, steps, eta=0.0, rng=None, trace=False):
    """DDIM exactly as in the notebook's ddim_sample (taus = round(linspace(T-1, 0, steps)))."""
    taus = np.round(np.linspace(sched.T - 1, 0, steps)).astype(int)
    x = x_T.copy()
    rows = []
    for i, t in enumerate(taus):
        ab_t = sched.alpha_bar[t]
        ab_prev = sched.alpha_bar[taus[i + 1]] if i + 1 < len(taus) else 1.0
        eps = eps_fn(x, ab_t)
        x0_hat = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        sigma = eta * np.sqrt(max((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev), 0.0))
        x_new = np.sqrt(ab_prev) * x0_hat + np.sqrt(max(1 - ab_prev - sigma ** 2, 0.0)) * eps
        if eta > 0 and i + 1 < len(taus):
            x_new = x_new + sigma * rng.normal(size=x.shape)
        if trace:
            rows.append((t, float(x[0]), ab_t, float(eps[0]), float(x0_hat[0]), ab_prev, float(x_new[0])))
        x = x_new
    return (x, rows) if trace else x


def w1_from_samples(a, b, n_q=2000):
    """1-D Wasserstein-1 distance between two sample sets via matched quantiles."""
    q = np.linspace(0.0005, 0.9995, n_q)
    return np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))


# ----------------------------------------------------------------------------------------------
# Section 1 · likelihood (mode-covering) versus mode-seeking fits of a bimodal distribution
# ----------------------------------------------------------------------------------------------
def fig_likelihood_vs_mode_seeking():
    banner("Figure likelihood_vs_mode_seeking: covering versus mode-seeking fits of a bimodal p_data")
    w, mu, sd = np.array([0.5, 0.5]), np.array([-2.0, 2.0]), np.array([0.5, 0.5])
    data = Mixture1D(w, mu, sd)
    xs = np.linspace(-6, 6, 2401); dx = xs[1] - xs[0]
    p = data.pdf(xs)
    pts, comp = data.sample(6, rng)
    print("six sample data points (component, x):")
    for c, x in zip(comp, pts):
        print(f"  mode {'A' if c == 0 else 'B'} (mean {mu[c]:+.1f}): x = {x:+.3f}")
    # maximum-likelihood Gaussian = moment matching
    m_ml = np.sum(w * mu)
    v_ml = np.sum(w * (sd ** 2 + mu ** 2)) - m_ml ** 2
    q_ml = npdf(xs, m_ml, np.sqrt(v_ml))
    kl_fwd_ml = np.sum(p * np.log(p / q_ml)) * dx
    kl_rev_ml = np.sum(q_ml * np.log(q_ml / p)) * dx
    # mode-seeking Gaussian: minimise the reverse KL(q || p) by grid search
    best = (np.inf, None, None)
    for m in np.linspace(-3, 3, 241):
        for s in np.linspace(0.2, 3.0, 141):
            q = npdf(xs, m, s)
            kl = np.sum(q * np.log(q / np.maximum(p, 1e-300))) * dx
            if kl < best[0]:
                best = (kl, m, s)
    kl_rev_ms, m_ms, s_ms = best
    q_ms = npdf(xs, m_ms, s_ms)
    kl_fwd_ms = np.sum(p * np.log(p / np.maximum(q_ms, 1e-300))) * dx
    print(f"p_data = 0.5 N({mu[0]}, {sd[0]}^2) + 0.5 N({mu[1]}, {sd[1]}^2)")
    print(f"maximum-likelihood Gaussian: mean {m_ml:+.3f}, std {np.sqrt(v_ml):.3f}  | KL(p||q) = {kl_fwd_ml:.3f}  KL(q||p) = {kl_rev_ml:.3f}")
    print(f"mode-seeking Gaussian (min KL(q||p)): mean {m_ms:+.3f}, std {s_ms:.3f}  | KL(p||q) = {kl_fwd_ms:.3f}  KL(q||p) = {kl_rev_ms:.3f}")
    print(f"density of the ML fit at the data modes x = +-2: {npdf(2.0, m_ml, np.sqrt(v_ml)):.3f}; at the empty middle x = 0: {npdf(0.0, m_ml, np.sqrt(v_ml)):.3f}")
    print(f"true density at x = 0: {data.pdf(np.array([0.0]))[0]:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, p, color=SERIES[0], label="data: two sharp modes")
    ax.plot(xs, q_ml, color=SERIES[1], label=f"likelihood fit (covers both): std {np.sqrt(v_ml):.2f}")
    ax.plot(xs, q_ms, color=SERIES[2], ls="--", label=f"mode-seeking fit (picks one): std {s_ms:.2f}")
    ax.annotate("mass where there is no data", xy=(0, npdf(0.0, m_ml, np.sqrt(v_ml))), xytext=(0.3, 0.45),
                fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.set_title("Covering versus mode-seeking: why likelihood blurs and GANs may collapse")
    ax.set_xlim(-5, 5); ax.legend(loc="upper left")
    save(fig, NB, "likelihood_vs_mode_seeking")


# ----------------------------------------------------------------------------------------------
# Section 2 · the optimal discriminator on two Gaussians, plus a Newton-fitted logistic discriminator
# ----------------------------------------------------------------------------------------------
def fig_toy_gan_optimal_discriminator():
    banner("Figure toy_gan_optimal_discriminator: p_data = N(0,1), p_g = N(2, 1.5^2)")
    xs = np.linspace(-5, 7, 1201); dx = xs[1] - xs[0]
    pd, pg = npdf(xs, 0, 1), npdf(xs, 2, 1.5)
    d_star = pd / (pd + pg)
    m = 0.5 * (pd + pg)
    js = 0.5 * np.sum(pd * np.log(pd / m)) * dx + 0.5 * np.sum(pg * np.log(pg / m)) * dx
    v_star = np.sum(pd * np.log(d_star)) * dx + np.sum(pg * np.log(1 - d_star)) * dx
    print("sample table: x, p_data(x), p_g(x), D*(x) = p_data / (p_data + p_g)")
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
        a, b = npdf(x, 0, 1), npdf(x, 2, 1.5)
        print(f"  x = {x:+.0f}: p_data = {a:.4f}  p_g = {b:.4f}  D* = {a / (a + b):.4f}")
    # crossing points where the densities agree (D* = 1/2): solve the quadratic log pd = log pg
    # -x^2/2 = -(x-2)^2/(2*2.25) - log 1.5  ->  multiply out
    a_, b_, c_ = -0.5 + 1 / 4.5, -4 / 4.5, 4 / 4.5 + np.log(1.5)
    roots = np.roots([a_, b_, c_])
    print(f"D* = 1/2 where p_data = p_g: x = {np.sort(roots)}")
    print(f"JS(p_data || p_g) = {js:.4f} nats (max possible log 2 = {LOG2:.4f})")
    print(f"V(D*, G) by numerical integration = {v_star:.4f};  -log 4 + 2 JS = {-np.log(4) + 2 * js:.4f}")
    # a logistic-regression discriminator with features [x, x^2] fitted by Newton's method on samples
    n = 4000
    xr, xf = rng.normal(0, 1, n), rng.normal(2, 1.5, n)
    X = np.concatenate([xr, xf]); y = np.concatenate([np.ones(n), np.zeros(n)])
    Phi = np.stack([np.ones_like(X), X, X ** 2], 1)
    wgt = np.zeros(3)
    for _ in range(25):
        pr = 1 / (1 + np.exp(-Phi @ wgt))
        g = Phi.T @ (pr - y)
        H = (Phi * (pr * (1 - pr))[:, None]).T @ Phi + 1e-6 * np.eye(3)
        wgt -= np.linalg.solve(H, g)
    d_learned = 1 / (1 + np.exp(-(np.stack([np.ones_like(xs), xs, xs ** 2], 1) @ wgt)))
    bce = -np.mean(np.concatenate([np.log(1 / (1 + np.exp(-(np.stack([np.ones_like(xr), xr, xr ** 2], 1) @ wgt)))),
                                   np.log(1 - 1 / (1 + np.exp(-(np.stack([np.ones_like(xf), xf, xf ** 2], 1) @ wgt))))]))
    print(f"logistic discriminator on {n}+{n} samples (features x, x^2): weights {wgt}; analytic logit = -x^2/2 + (x-2)^2/4.5 + log 1.5 -> "
          f"[{np.log(1.5) + 4 / 4.5:.4f}, {-4 / 4.5:.4f}, {-0.5 + 1 / 4.5:.4f}]")
    print(f"max |D_learned - D*| on [-5, 7] = {np.max(np.abs(d_learned - d_star)):.4f}; V(D_learned) on the training samples = {-2 * bce:.4f} <= V(D*) = {v_star:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(xs, pd, color=SERIES[0], label="$p_{data} = \\mathcal{N}(0, 1)$")
    axes[0].plot(xs, pg, color=SERIES[1], label="$p_g = \\mathcal{N}(2, 1.5^2)$")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("density"); axes[0].set_title("The two distributions"); axes[0].legend()
    axes[1].plot(xs, d_star, color=SERIES[0], label="optimal $D^* = p_{data} / (p_{data} + p_g)$")
    axes[1].plot(xs, d_learned, color=SERIES[1], ls="--", label="logistic discriminator fitted on samples")
    axes[1].axhline(0.5, color=AXIS, lw=0.8)
    x_cross = np.sort(roots)[0]
    axes[1].plot([x_cross], [0.5], "o", color=INK_2, ms=6)
    axes[1].annotate(f"densities agree at x = {x_cross:.2f}\n$D^*$ = 0.5", xy=(x_cross, 0.5), xytext=(x_cross + 1.2, 0.62), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].set_xlabel("x"); axes[1].set_ylabel("D(x)"); axes[1].set_ylim(0, 1.02); axes[1].set_title("Optimal versus fitted discriminator"); axes[1].legend(loc="upper right")
    save(fig, NB, "toy_gan_optimal_discriminator")


# ----------------------------------------------------------------------------------------------
# Section 3 · mode collapse on a ring of eight Gaussians (schematic)
# ----------------------------------------------------------------------------------------------
def fig_mode_collapse_schematic():
    banner("Figure mode_collapse_schematic: 8-Gaussian ring, healthy versus collapsed generator (schematic)")
    K, radius, sd = 8, 2.0, 0.15
    ang = 2 * np.pi * np.arange(K) / K
    centres = np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1)
    n_data, n_gen = 800, 400
    k_data = rng.integers(0, K, n_data)
    data = centres[k_data] + sd * rng.normal(size=(n_data, 2))
    k_good = rng.integers(0, K, n_gen)
    gen_good = centres[k_good] + 0.18 * rng.normal(size=(n_gen, 2))
    k_bad = rng.choice([1, 5], size=n_gen)
    gen_bad = centres[k_bad] + np.array([0.12, -0.08]) + 0.10 * rng.normal(size=(n_gen, 2))

    def coverage(pts):
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2)
        counts = np.bincount(d.argmin(1), minlength=K)
        return counts, int(np.sum(counts >= 0.05 * len(pts)))

    c_data, cov_data = coverage(data)
    c_good, cov_good = coverage(gen_good)
    c_bad, cov_bad = coverage(gen_bad)
    print("mode centres (x, y) and share of points assigned to each mode (nearest centre):")
    print("  mode | centre        | data  | healthy G | collapsed G")
    for k in range(K):
        print(f"  {k:4d} | ({centres[k, 0]:+.2f}, {centres[k, 1]:+.2f}) | {c_data[k] / n_data:.3f} | {c_good[k] / n_gen:9.3f} | {c_bad[k] / n_gen:.3f}")
    print(f"modes covered (>= 5% of samples): data {cov_data}/8, healthy generator {cov_good}/8, collapsed generator {cov_bad}/8")
    print(f"ideal share per mode = 1/8 = {1 / 8:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, g, title, cov in [(axes[0], gen_good, "Healthy generator", cov_good), (axes[1], gen_bad, "Mode collapse", cov_bad)]:
        ax.scatter(data[:, 0], data[:, 1], s=7, color=SERIES[0], alpha=0.35, label="data (8 modes)")
        ax.scatter(g[:, 0], g[:, 1], s=7, color=SERIES[1], alpha=0.6, label="generator samples")
        ax.set_title(f"{title}: {cov}/8 modes covered (schematic)")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].annotate("many different z,\nnearly the same output", xy=(centres[1, 0] + 0.12, centres[1, 1] - 0.08), xytext=(-2.9, 2.2), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    save(fig, NB, "mode_collapse_schematic")


# ----------------------------------------------------------------------------------------------
# Section 4 · Jensen-Shannon versus Wasserstein-1 for two unit intervals
# ----------------------------------------------------------------------------------------------
def fig_js_vs_wasserstein():
    banner("Figure js_vs_wasserstein_intervals: JS and W1 between U[0,1] and U[theta, theta+1]")
    grid = np.linspace(-3.0, 4.0, 7001); dx = grid[1] - grid[0]

    def upd(a, b):
        return ((grid >= a) & (grid < b)).astype(float) / (b - a)

    def jsd(p, q):
        m = 0.5 * (p + q)
        rp = np.where(p > 0, p / np.maximum(m, 1e-12), 1.0)
        rq = np.where(q > 0, q / np.maximum(m, 1e-12), 1.0)
        return (0.5 * np.sum(p * np.log(rp)) + 0.5 * np.sum(q * np.log(rq))) * dx

    def w1(p, q):
        Fp, Fq = np.cumsum(p) * dx, np.cumsum(q) * dx
        return np.sum(np.abs(Fp - Fq)) * dx

    p0 = upd(0.0, 1.0)
    thetas = np.linspace(-2.5, 2.5, 101)
    js_vals = np.array([jsd(p0, upd(th, th + 1)) for th in thetas])
    w1_vals = np.array([w1(p0, upd(th, th + 1)) for th in thetas])
    print("sample table: theta, overlap length, JS (nats), |theta| log 2, W1")
    for th in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        print(f"  theta = {th:.1f}: overlap {max(0.0, 1 - th):.1f}  JS = {jsd(p0, upd(th, th + 1)):.4f}  (|theta| log 2 capped = {min(th, 1) * LOG2:.4f})  W1 = {w1(p0, upd(th, th + 1)):.4f}")
    print(f"log 2 = {LOG2:.4f}; slope of JS for |theta| > 1: {(js_vals[-1] - js_vals[-11]) / (thetas[-1] - thetas[-11]):.4f}; slope of W1 there: {(w1_vals[-1] - w1_vals[-11]) / (thetas[-1] - thetas[-11]):.4f}")
    # CDF detail for theta = 2: the two step functions differ by 1 on [0, 2] minus the ramps -> area = 2
    Fp, Fq = np.cumsum(p0) * dx, np.cumsum(upd(2.0, 3.0)) * dx
    print(f"theta = 2: integral of |F_p - F_q| = {np.sum(np.abs(Fp - Fq)) * dx:.4f} (W1); 1-Lipschitz critic f(x) = -x gives E_p f - E_q f = {(-np.sum(grid * p0) + np.sum(grid * upd(2.0, 3.0))) * dx:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(thetas, js_vals, color=SERIES[0], label="JS( U[0,1] || U[$\\theta$, $\\theta$+1] )")
    axes[0].axhline(LOG2, color=AXIS, lw=0.8, ls="--")
    axes[0].annotate("log 2: flat, zero gradient", xy=(1.8, LOG2), xytext=(0.9, 0.5), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[0].set_xlabel("$\\theta$ (shift of the generator's interval)"); axes[0].set_ylabel("nats"); axes[0].set_title("Jensen-Shannon divergence")
    axes[0].set_ylim(0, 0.8)
    axes[1].plot(thetas, w1_vals, color=SERIES[1], label="$W_1$( U[0,1], U[$\\theta$, $\\theta$+1] )")
    axes[1].annotate("$W_1 = |\\theta|$: slope of constant size", xy=(2.0, 2.0), xytext=(0.2, 2.2), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].set_xlabel("$\\theta$ (shift of the generator's interval)"); axes[1].set_ylabel("distance"); axes[1].set_title("Wasserstein-1 distance")
    save(fig, NB, "js_vs_wasserstein_intervals")


# ----------------------------------------------------------------------------------------------
# Section 5 · the forward process on two moons (closed form) and the two schedules
# ----------------------------------------------------------------------------------------------
def fig_forward_diffusion_two_moons():
    banner("Figure forward_diffusion_two_moons: closed-form x_t for a 2-D dataset at four noise levels")
    T = 200
    sched = Schedule(linear_betas(T))
    X, lab = make_moons(n_samples=500, noise=0.06, random_state=0)
    X = (X - X.mean(0)) / X.std(0)                 # zero mean, unit variance per axis
    eps = rng.normal(size=X.shape)                  # one noise draw, reused at every level
    t_idx = [0, T // 4 - 1, T // 2 - 1, T - 1]      # code indices -> paper t = 1, 50, 100, 200
    print(f"T = {T}, linear schedule with beta from {sched.betas[0]:.1e} to {sched.betas[-1]:.3f}; data standardised to unit variance per axis")
    print("three data points x0 and their noised versions at t = T/2 (same epsilon):")
    ab_half = sched.alpha_bar[T // 2 - 1]
    for i in range(3):
        xt = np.sqrt(ab_half) * X[i] + np.sqrt(1 - ab_half) * eps[i]
        print(f"  x0 = ({X[i, 0]:+.3f}, {X[i, 1]:+.3f})  eps = ({eps[i, 0]:+.3f}, {eps[i, 1]:+.3f})  ->  x_T/2 = sqrt({ab_half:.4f}) x0 + sqrt({1 - ab_half:.4f}) eps = ({xt[0]:+.3f}, {xt[1]:+.3f})")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, t in zip(axes, t_idx):
        ab = sched.alpha_bar[t]
        Xt = np.sqrt(ab) * X + np.sqrt(1 - ab) * eps
        print(f"  t = {t + 1:3d}/{T}: alpha_bar = {ab:.4f}, sqrt(alpha_bar) = {np.sqrt(ab):.3f}, sqrt(1 - alpha_bar) = {np.sqrt(1 - ab):.3f} | "
              f"empirical std of x_t per axis = ({Xt[:, 0].std():.3f}, {Xt[:, 1].std():.3f}) (closed form: 1.000 since data variance is 1)")
        for c, col in [(0, SERIES[0]), (1, SERIES[1])]:
            ax.scatter(Xt[lab == c, 0], Xt[lab == c, 1], s=5, color=col, alpha=0.6, label=f"moon {c}")
        ax.set_title(f"t = {t + 1}/{T},  $\\bar\\alpha_t$ = {ab:.3f}", fontsize=10)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$"); axes[0].legend(loc="upper left", fontsize=8, markerscale=2)
    fig.suptitle("Forward process in closed form: $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$ (linear schedule)", fontsize=11, fontweight="bold")
    save(fig, NB, "forward_diffusion_two_moons")


def fig_noise_schedules():
    banner("Figure noise_schedules: beta_t, alpha_bar_t and log SNR for the linear and cosine schedules (T = 1000)")
    T = 1000
    lin = Schedule(linear_betas(T))
    u = np.arange(T + 1) / T
    ab_cos_full = cosine_alpha_bar(u)
    betas_cos = 1 - ab_cos_full[1:] / ab_cos_full[:-1]
    n_clip = int(np.sum(betas_cos > 0.999))
    cos = Schedule(np.minimum(betas_cos, 0.999))
    tt = (np.arange(T) + 1) / T
    print(f"linear: beta from {lin.betas[0]:.1e} to {lin.betas[-1]:.3f}, sum(beta) = {lin.betas.sum():.3f}, alpha_bar_T = {lin.alpha_bar[-1]:.2e} (e^-10 = {np.exp(-10):.2e})")
    print(f"cosine: beta from {betas_cos[0]:.2e} to {betas_cos[-1]:.4f}; betas above 0.999 (clipped): {n_clip} (the last one); second to last beta = {betas_cos[-2]:.3f}; "
          f"max |cumprod - closed form| = {np.max(np.abs(cos.alpha_bar - ab_cos_full[1:])):.1e}")
    print("sample table: t/T, beta (lin), beta (cos), alpha_bar (lin), alpha_bar (cos), log SNR (lin), log SNR (cos)")
    for f in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        i = int(round(f * T)) - 1
        s_l, s_c = lin.alpha_bar[i] / (1 - lin.alpha_bar[i]), cos.alpha_bar[i] / (1 - cos.alpha_bar[i])
        print(f"  {f:4.2f}: {lin.betas[i]:.4f}  {cos.betas[i]:.4f}  {lin.alpha_bar[i]:.4f}  {cos.alpha_bar[i]:.4f}  {np.log(s_l):+7.2f}  {np.log(max(s_c, 1e-300)):+7.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, (yl, yc), ylabel, title in [
        (axes[0], (lin.betas, cos.betas), "$\\beta_t$", "Per-step noise"),
        (axes[1], (lin.alpha_bar, cos.alpha_bar), "$\\bar\\alpha_t$", "Signal fraction"),
        (axes[2], (np.log(lin.alpha_bar / (1 - lin.alpha_bar)), np.log(np.maximum(cos.alpha_bar / (1 - cos.alpha_bar), 1e-12))), "log SNR", "Log signal-to-noise ratio"),
    ]:
        ax.plot(tt, yl, color=SERIES[0], label="linear")
        ax.plot(tt, yc, color=SERIES[1], label="cosine")
        ax.set_xlabel("t / T"); ax.set_ylabel(ylabel); ax.set_title(title)
    axes[0].set_yscale("log"); axes[0].set_ylim(5e-5, 2)
    axes[2].set_ylim(-14, 12)
    i_half = T // 2 - 1
    axes[1].annotate(f"t/T = 0.5: {lin.alpha_bar[i_half]:.2f} vs {cos.alpha_bar[i_half]:.2f}", xy=(0.5, lin.alpha_bar[i_half]), xytext=(0.52, 0.6), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].legend()
    save(fig, NB, "noise_schedules")


# ----------------------------------------------------------------------------------------------
# Section 6 · time embedding, exact 1-D DDPM / DDIM sampling, classifier-free guidance
# ----------------------------------------------------------------------------------------------
def sinusoidal(t, dim):
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = np.asarray(t, dtype=float)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], 1), freqs


def fig_sinusoidal_time_embedding():
    banner("Figure sinusoidal_time_embedding: 64-d embedding of t = 0 ... 199")
    T, dim = 200, 64
    emb, freqs = sinusoidal(np.arange(T), dim)
    print(f"dim = {dim}: {dim // 2} frequencies 10000^(-i/32): f_0 = {freqs[0]:.4f}, f_1 = {freqs[1]:.4f}, f_2 = {freqs[2]:.4f}, ..., f_31 = {freqs[31]:.6f}")
    print(f"periods (steps): fastest {2 * np.pi / freqs[0]:.2f}, dimension 8: {2 * np.pi / freqs[8]:.1f}, slowest {2 * np.pi / freqs[31]:.0f}")
    print("sample table: t, emb[0] = sin(t f_0), emb[1] = sin(t f_1), emb[8] = sin(t f_8), emb[32] = cos(t f_0), emb[40] = cos(t f_8), emb[63] = cos(t f_31)")
    for t in [0, 1, 2, 50, 199]:
        e = emb[t]
        print(f"  t = {t:3d}: {e[0]:+.4f}  {e[1]:+.4f}  {e[8]:+.4f}  {e[32]:+.4f}  {e[40]:+.4f}  {e[63]:+.4f}")
    print(f"squared norm of every embedding = {np.sum(emb[37] ** 2):.1f} (= dim/2); nearest other step to t = 100 by Euclidean distance: {np.argsort(np.linalg.norm(emb - emb[100], axis=1))[1]}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(emb.T, aspect="auto", cmap=DIV_CMAP, vmin=-1, vmax=1, origin="lower")
    ax.grid(False)
    ax.set_xlabel("time step t"); ax.set_ylabel("embedding dimension (0-31 sines, 32-63 cosines)"); ax.set_title("Sinusoidal time embedding (64-d): fast stripes at the bottom, slow at the top")
    fig.colorbar(im, ax=ax, label="value", fraction=0.04, pad=0.02)
    save(fig, NB, "sinusoidal_time_embedding")


def fig_exact_diffusion_1d_sampling():
    banner("Figure exact_diffusion_1d_sampling: ancestral and DDIM sampling with the exact epsilon* of a 1-D mixture")
    T = 200
    sched = Schedule(linear_betas(T))
    data = Mixture1D([0.6, 0.4], [-1.5, 1.0], [0.3, 0.3])
    n = 20000
    x_T = rng.normal(size=n)
    print(f"p_data = 0.6 N(-1.5, 0.3^2) + 0.4 N(1.0, 0.3^2); T = {T} (linear schedule rescaled); {n} samples from the same x_T")
    x_ddpm = ddpm_sample_1d(data.eps_star, sched, x_T, rng)
    x_ddim10, rows = ddim_sample_1d(data.eps_star, sched, x_T, 10, trace=True)
    x_ddim50 = ddim_sample_1d(data.eps_star, sched, x_T, 50)
    x_ddim3 = ddim_sample_1d(data.eps_star, sched, x_T, 3)
    ref, _ = data.sample(200000, rng)
    print("one DDIM-10 trajectory, step by step (x_T = {:+.4f}):".format(x_T[0]))
    print("  tau | x_tau    | alpha_bar | eps*    | x0_hat  | alpha_bar_prev | x_prev")
    for (t, x, ab, e, x0h, abp, xn) in rows:
        print(f"  {t:3d} | {x:+.4f}  | {ab:.4f}    | {e:+.4f} | {x0h:+.4f} | {abp:.4f}         | {xn:+.4f}")
    print(f"first ancestral step by hand: t = {T - 1}, beta = {sched.betas[-1]:.4f}, alpha = {sched.alphas[-1]:.4f}, sqrt(1 - alpha_bar) = {np.sqrt(1 - sched.alpha_bar[-1]):.4f}, "
          f"posterior std = {np.sqrt(sched.posterior_var[-1]):.4f}")
    e0 = data.eps_star(x_T[:1], sched.alpha_bar[-1])[0]
    mean0 = (x_T[0] - sched.betas[-1] / np.sqrt(1 - sched.alpha_bar[-1]) * e0) / np.sqrt(sched.alphas[-1])
    print(f"  x_T = {x_T[0]:+.4f}, eps* = {e0:+.4f} -> mean = (x_T - beta/sqrt(1-alpha_bar) eps*)/sqrt(alpha) = {mean0:+.4f}, then add N(0, {sched.posterior_var[-1]:.4f}) noise")
    print("results: sampler, fraction in the left mode (x < -0.25) [true 0.600], mean/std in the left mode [true -1.500/0.300], mean/std in the right mode [true +1.000/0.300], W1 to data")
    for name, x in [("data (200k)", ref), ("DDPM, 200 steps", x_ddpm), ("DDIM, 50 steps", x_ddim50), ("DDIM, 10 steps", x_ddim10), ("DDIM, 3 steps", x_ddim3)]:
        L, R = x[x < -0.25], x[x >= -0.25]
        print(f"  {name:16s}: left {len(L) / len(x):.3f}  left {L.mean():+.3f}/{L.std():.3f}  right {R.mean():+.3f}/{R.std():.3f}  W1 = {w1_from_samples(x, ref):.4f}")
    xs = np.linspace(-3, 2.5, 800)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(-3, 2.5, 111)
    ax.hist(x_ddpm, bins=bins, density=True, histtype="step", color=SERIES[0], lw=1.6, label="ancestral DDPM, 200 steps")
    ax.hist(x_ddim10, bins=bins, density=True, histtype="step", color=SERIES[1], lw=1.6, label="DDIM, 10 steps (same $x_T$)")
    ax.hist(x_ddim3, bins=bins, density=True, histtype="step", color=SERIES[2], lw=1.2, ls="--", label="DDIM, 3 steps (too coarse)")
    ax.plot(xs, data.pdf(xs), color=INK_2, lw=1.0, label="true $p_{data}$")
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.set_title("Sampling a 1-D diffusion model with the exact noise predictor")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "exact_diffusion_1d_sampling")


def fig_cfg_guidance_1d():
    banner("Figure cfg_guidance_1d: classifier-free guidance on the 1-D mixture, requesting the right-hand class")
    T = 200
    sched = Schedule(linear_betas(T))
    data = Mixture1D([0.6, 0.4], [-1.5, 1.0], [0.3, 0.3])
    n, steps, k_req = 20000, 50, 1
    x_T = rng.normal(size=n)
    scales = [0.0, 1.0, 3.0, 7.0]
    results = {}
    print(f"request class y = {k_req} (mean {data.mu[k_req]:+.1f}, std {data.sigma[k_req]:.1f}, prior weight {data.w[k_req]:.1f}); DDIM {steps} steps, {n} samples from the same x_T")
    print("guided eps = eps_uncond + s (eps_cond - eps_uncond); at x = 0.0, t = T/2:")
    ab_half = sched.alpha_bar[T // 2 - 1]
    eu = data.eps_star(np.array([0.0]), ab_half)[0]; ec = data.eps_star(np.array([0.0]), ab_half, k=k_req)[0]
    for s in scales:
        print(f"  s = {s:.0f}: eps_uncond = {eu:+.4f}, eps_cond = {ec:+.4f} -> guided eps = {eu + s * (ec - eu):+.4f}")
    print("results: s, fraction landing in the requested mode (x > -0.25), mean and std of those samples [true class: +1.000, 0.300]")
    for s in scales:
        fn = (lambda x, ab, s=s: data.eps_star(x, ab) + s * (data.eps_star(x, ab, k=k_req) - data.eps_star(x, ab)))
        x = ddim_sample_1d(fn, sched, x_T, steps)
        results[s] = x
        R = x[x > -0.25]
        print(f"  s = {s:.0f}: fraction {len(R) / n:.3f}, mean {R.mean():+.3f}, std {R.std():.3f}")
    xs = np.linspace(-3, 2.5, 800)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(-3, 2.5, 141)
    for i, s in enumerate(scales):
        ax.hist(results[s], bins=bins, density=True, histtype="step", color=SERIES[i], lw=1.6, label=f"guidance scale s = {s:.0f}")
    ax.plot(xs, npdf(xs, data.mu[k_req], data.sigma[k_req]), color=INK_2, lw=1.0, ls="--", label="true class-1 density")
    ax.set_xlabel("x"); ax.set_ylabel("density"); ax.set_title("Classifier-free guidance: larger s = more on-class, less diverse")
    ax.set_ylim(0, 3.2); ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "cfg_guidance_1d")


# ----------------------------------------------------------------------------------------------
# Section 7 · flow matching: straight conditional paths (2-D) and the bending marginal field (1-D)
# ----------------------------------------------------------------------------------------------
def marginal_velocity_1d(x, t, targets=(-2.0, 2.0)):
    """Exact E[x1 - x0 | x_t = x] for x1 uniform on `targets`, x0 ~ N(0,1), x_t = (1-t) x0 + t x1."""
    tg = np.asarray(targets)
    logw = -0.5 * ((x[:, None] - t * tg[None, :]) / (1 - t)) ** 2
    r = np.exp(logw - logw.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    v = tg[None, :] - (x[:, None] - t * tg[None, :]) / (1 - t)
    return np.sum(r * v, 1)


def fig_flow_matching_paths():
    banner("Figure flow_matching_paths: straight conditional paths in 2-D and the exact marginal field in 1-D")
    # left: eight (noise, data) pairs and their straight lines
    n_pairs = 8
    ang = 2 * np.pi * (np.arange(n_pairs) + 0.5) / n_pairs
    x1 = np.stack([2.2 * np.cos(ang), 2.2 * np.sin(ang)], 1)
    x0 = rng.normal(size=(n_pairs, 2))
    print("sample pairs: noise x0, data x1, velocity x1 - x0, midpoint x_0.5 = 0.5 x0 + 0.5 x1")
    for i in range(3):
        mid = 0.5 * x0[i] + 0.5 * x1[i]
        print(f"  pair {i}: x0 = ({x0[i, 0]:+.3f}, {x0[i, 1]:+.3f})  x1 = ({x1[i, 0]:+.3f}, {x1[i, 1]:+.3f})  v = ({x1[i, 0] - x0[i, 0]:+.3f}, {x1[i, 1] - x0[i, 1]:+.3f})  x_0.5 = ({mid[0]:+.3f}, {mid[1]:+.3f})")
    i0, i1 = 0, 1
    print(f"regression target at the midpoint of pair 0 is its own velocity ({x1[0, 0] - x0[0, 0]:+.3f}, {x1[0, 1] - x0[0, 1]:+.3f}); the lines of pairs {i0} and {i1} may cross, so the learned field is their average there")
    # right: exact marginal field for a two-point data distribution in 1-D
    n_traj, n_steps = 24, 100
    starts = np.array([np.sqrt(2) * 1.0 * s for s in np.linspace(-2.1, 2.1, n_traj)])
    paths = [starts.copy()]
    x = starts.copy()
    for i in range(n_steps):
        x = x + (1.0 / n_steps) * marginal_velocity_1d(x, i / n_steps)
        paths.append(x.copy())
    paths = np.array(paths)
    print("1-D two-point problem: x1 in {-2, +2} with equal probability, x0 ~ N(0, 1). Exact marginal velocity at t = 0 is -x (both targets equally likely).")
    xx = np.array([-1.0, 0.0, 0.5, 1.5])
    print(f"  v(x, t=0)   at x = {xx}: {marginal_velocity_1d(xx, 0.0)}")
    print(f"  v(x, t=0.5) at x = {xx}: {marginal_velocity_1d(xx, 0.5)}")
    print(f"  v(x, t=0.9) at x = {xx}: {marginal_velocity_1d(xx, 0.9)}")
    big = rng.normal(size=4000)
    print("Euler integration of the exact field from 4000 noise samples: steps, mean |x_1 - nearest target|, fraction within 0.1 of a target, fraction ending at +2")
    for ns in [1, 2, 5, 10, 100]:
        x = big.copy()
        for i in range(ns):
            x = x + (1.0 / ns) * marginal_velocity_1d(x, i / ns)
        err = np.minimum(np.abs(x - 2), np.abs(x + 2))
        print(f"  {ns:3d} steps: error {err.mean():.3f}, within 0.1: {np.mean(err < 0.1):.3f}, at +2: {np.mean(x > 0):.3f}")
    print(f"  (a starting x0 = +0.30 ends at {paths[-1][np.argmin(np.abs(starts - 0.3))]:+.3f} after 100 steps; x0 = -0.30 ends at {paths[-1][np.argmin(np.abs(starts + 0.3))]:+.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax = axes[0]
    for i in range(n_pairs):
        ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], color=AXIS, lw=1.0, zorder=1)
        mid = 0.5 * x0[i] + 0.5 * x1[i]
        ax.plot([mid[0]], [mid[1]], "o", color=SERIES[2], ms=4, zorder=2)
    ax.scatter(x0[:, 0], x0[:, 1], s=36, color=SERIES[0], label="noise $x_0 \\sim \\mathcal{N}(0, I)$", zorder=3)
    ax.scatter(x1[:, 0], x1[:, 1], s=36, color=SERIES[1], label="data $x_1$", zorder=3)
    ax.plot([], [], "o", color=SERIES[2], ms=4, label="$x_{0.5}$: regression target is $x_1 - x_0$")
    ax.set_title("Conditional paths: straight lines, constant velocity"); ax.set_xlabel("$x_1$ coordinate"); ax.set_ylabel("$x_2$ coordinate")
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    ts = np.linspace(0, 1, n_steps + 1)
    for j in range(n_traj):
        ax.plot(ts, paths[:, j], color=SERIES[0], lw=0.9, alpha=0.8)
    ax.axhline(2.0, color=SERIES[1], lw=1.0, ls="--"); ax.axhline(-2.0, color=SERIES[1], lw=1.0, ls="--")
    ax.annotate("data at +2 and -2", xy=(0.05, 2.0), xytext=(0.05, 2.45), fontsize=8, color=INK_2)
    ax.annotate("the marginal field bends\nwhere conditional paths cross", xy=(0.75, 0.55), xytext=(0.3, 1.1), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_title("Marginal field (exact), 24 trajectories, 100 Euler steps"); ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_ylim(-3.2, 3.2)
    save(fig, NB, "flow_matching_paths")


# ----------------------------------------------------------------------------------------------
# Section 8 · sampling cost by family
# ----------------------------------------------------------------------------------------------
def fig_sampling_cost_by_family():
    banner("Figure sampling_cost_by_family: network evaluations per generated sample")
    labels = ["VAE\n(decoder)", "GAN\n(generator)", "Flow matching\n(10 Euler steps)", "DDIM\n(50 steps)", "DDIM + CFG\n(50 steps, 2 evals)", "DDPM\n(T = 1000)"]
    values = [1, 1, 10, 50, 100, 1000]
    for l, v in zip(labels, values):
        print(f"  {l.replace(chr(10), ' '):36s}: {v:5d} evaluations per sample  ({1000 / v:7.1f}x cheaper than DDPM)")
    print(f"  notebook's own flow-matching sampler used 100 Euler steps; the DDIM comparison used 5, 10, 20 and 50 steps against T ancestral steps")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bar(ax, labels, values, color=SERIES[0], label_fmt="{:,}")
    ax.set_yscale("log"); ax.set_ylim(0.7, 3000)
    ax.set_ylabel("network evaluations per sample (log scale)"); ax.set_title("What one sample costs: one pass versus hundreds")
    ax.tick_params(axis="x", labelsize=8)
    save(fig, NB, "sampling_cost_by_family")


def main():
    fig_likelihood_vs_mode_seeking()
    fig_toy_gan_optimal_discriminator()
    fig_mode_collapse_schematic()
    fig_js_vs_wasserstein()
    fig_forward_diffusion_two_moons()
    fig_noise_schedules()
    fig_sinusoidal_time_embedding()
    fig_exact_diffusion_1d_sampling()
    fig_cfg_guidance_1d()
    fig_flow_matching_paths()
    fig_sampling_cost_by_family()


if __name__ == "__main__":
    main()
