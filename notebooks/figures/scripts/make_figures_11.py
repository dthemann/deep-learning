"""Figures for the companion guide of Notebook 11 · Autoencoders and Variational Autoencoders.

Everything here is numpy + matplotlib (plus scikit-learn's bundled ``load_digits`` for the linear
probe).  The linear autoencoder, the toy linear-Gaussian VAE, the softmax probe and the VQ codebook
are optimised with a small numpy Adam so that every experiment of the notebook has a hand-checkable
counterpart that runs without PyTorch.

Run:  python3 notebooks/figures/scripts/make_figures_11.py
Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_digits

NB = "11"
np.set_printoptions(precision=4, suppress=True)


def banner(title):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)


# ----------------------------------------------------------------------------------------------
# A tiny numpy Adam, used by the linear autoencoder, the toy VAE and the codebook experiments
# ----------------------------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, betas[0], betas[1], eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + self.eps)


def cosine_lr(lr0, t, T):
    return lr0 * 0.5 * (1 + np.cos(np.pi * t / T))


def kl_gauss(mu, var):
    """KL( N(mu, var) || N(0, 1) ) per dimension, the notebook's closed form."""
    return 0.5 * (mu ** 2 + var - 1.0 - np.log(var))


# ----------------------------------------------------------------------------------------------
# Section 1 · manifold hypothesis
# ----------------------------------------------------------------------------------------------
def fig_manifold_hypothesis():
    banner("Figure manifold_hypothesis: a 1-D curve in 2-D, PCA line vs curve coordinate; PCA spectra")
    n = 300
    t = rng.uniform(0, np.pi, n)
    X = np.stack([np.cos(t), np.sin(t)], 1) + 0.03 * rng.normal(size=(n, 2))
    Xc = X - X.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Xc, rowvar=False))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    pc1 = evecs[:, 0]
    proj = Xc @ pc1[:, None] * pc1[None, :] + X.mean(0)               # PCA-1 reconstruction
    on_curve = X / np.linalg.norm(X, axis=1, keepdims=True)           # nearest point on the unit arc
    mse_pca = np.mean(np.sum((X - proj) ** 2, 1))
    mse_curve = np.mean(np.sum((X - on_curve) ** 2, 1))
    print("sample points (first 6): t, x1, x2")
    for i in range(6):
        print(f"  t={t[i]:.2f}  x=({X[i,0]:+.3f}, {X[i,1]:+.3f})")
    print(f"covariance eigenvalues: {evals}  fraction of variance in PC1: {evals[0]/evals.sum():.3f}")
    print(f"mean squared distance to the PCA line (1 linear coordinate): {mse_pca:.4f}")
    print(f"mean squared distance to the arc (1 curved coordinate):      {mse_curve:.4f}  (noise var 2*0.03^2 = {2*0.03**2:.4f})")

    # spectra: structured 50-D data of rank 5 versus isotropic noise
    D, r, m = 50, 5, 2000
    S = rng.normal(size=(m, r)) @ (rng.normal(size=(r, D)) * np.sqrt(np.linspace(6, 1, r))[:, None]) + 0.3 * rng.normal(size=(m, D))
    Nz = rng.normal(size=(m, D))
    def cum_var(A):
        A = A - A.mean(0); s = np.linalg.svd(A, compute_uv=False) ** 2
        return np.cumsum(s) / s.sum()
    cs, cn = cum_var(S), cum_var(Nz)
    k95_s, k95_n = int(np.searchsorted(cs, 0.95)) + 1, int(np.searchsorted(cn, 0.95)) + 1
    print(f"structured rank-5 data: 95% of variance in {k95_s} of {D} components; isotropic noise: {k95_n} of {D}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=8, color=BLUE, alpha=0.6, label="data (noisy arc)")
    mu = X.mean(0); L = 1.4
    ax.plot([mu[0] - L * pc1[0], mu[0] + L * pc1[0]], [mu[1] - L * pc1[1], mu[1] + L * pc1[1]], color=ORANGE, label="PCA: first principal axis")
    th = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=AQUA, label="the curve (1 coordinate)")
    ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("A 1-D manifold inside a 2-D space"); ax.legend(loc="lower center", fontsize=8)
    ax = axes[1]
    ax.plot(np.arange(1, D + 1), cs, color=BLUE, label=f"structured data (95% in {k95_s} components)")
    ax.plot(np.arange(1, D + 1), cn, color=ORANGE, label=f"isotropic noise (95% in {k95_n} components)")
    ax.axhline(0.95, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("number of principal components"); ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA spectrum: structure vs noise"); ax.legend(fontsize=8, loc="lower right")
    save(fig, NB, "manifold_hypothesis")


# ----------------------------------------------------------------------------------------------
# Section 2 · linear AE = PCA, denoising, anomaly detection
# ----------------------------------------------------------------------------------------------
def train_linear_ae(X, k, steps=1500, lr=1e-2, seed=1):
    r = np.random.default_rng(seed)
    N, D = X.shape
    We, Wd = 0.1 * r.normal(size=(k, D)), 0.1 * r.normal(size=(D, k))
    opt = Adam([We, Wd], lr=lr)
    for t in range(1, steps + 1):
        R = X @ We.T @ Wd.T - X
        gWd = 2 * R.T @ (X @ We.T) / (N * D)
        gWe = 2 * (Wd.T @ R.T @ X) / (N * D)
        opt.step([gWe, gWd], lr=cosine_lr(lr, t, steps))
    return We, Wd


def fig_linear_ae_vs_pca():
    banner("Figure linear_ae_vs_pca: linear autoencoder (gradient descent) vs PCA on a synthetic Gaussian dataset")
    N, D = 400, 12
    spectrum = np.array([5.0, 3.0, 2.0, 1.2, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05])
    Qrot, _ = np.linalg.qr(rng.normal(size=(D, D)))
    X = rng.normal(size=(N, D)) * np.sqrt(spectrum) @ Qrot.T
    X -= X.mean(0)
    evals = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"dataset: N={N}, D={D}; sample eigenvalues of the covariance: {evals}")
    ks = [1, 2, 3, 4, 6, 8]
    mse_ae, mse_pca = [], []
    print(f"{'k':>3s} {'linear AE MSE':>14s} {'PCA-k MSE':>10s} {'ratio':>7s}")
    for k in ks:
        We, Wd = train_linear_ae(X, k)
        Vk = Vt[:k].T
        m_ae = np.mean((X @ We.T @ Wd.T - X) ** 2)
        m_pca = np.mean((X @ Vk @ Vk.T - X) ** 2)
        mse_ae.append(m_ae); mse_pca.append(m_pca)
        print(f"{k:3d} {m_ae:14.5f} {m_pca:10.5f} {m_ae/m_pca:7.4f}")
        if k == 3:
            Q, _ = np.linalg.qr(Wd)
            cos = np.linalg.svd(Q.T @ Vk, compute_uv=False)
            gap = np.linalg.norm(Q @ Q.T - Vk @ Vk.T) / np.linalg.norm(Vk @ Vk.T)
            print(f"   k=3: cosines of the principal angles {cos}; relative gap between projection matrices {gap:.2e}")
            print(f"   k=3: decoder columns are NOT the principal components: |corr| of Wd[:,0] with PC1 = "
                  f"{abs(np.dot(Wd[:,0], Vk[:,0]))/np.linalg.norm(Wd[:,0]):.3f}")
    print(f"Eckart-Young check for k=3: sum of trailing eigenvalues / D = {evals[3:].sum()/D:.5f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ks, mse_ae, "o-", color=BLUE, label="linear autoencoder (Adam, 1500 full-batch steps)", ms=8)
    ax.plot(ks, mse_pca, "s--", color=ORANGE, label="PCA with k components (closed form)", ms=6)
    ax.set_xlabel("bottleneck size k"); ax.set_ylabel("reconstruction MSE per element")
    ax.set_title("A linear autoencoder lands exactly on PCA")
    ax.annotate(f"k=3: AE {mse_ae[2]:.4f}\nPCA {mse_pca[2]:.4f}", (3, mse_ae[2]), textcoords="offset points",
                xytext=(18, 14), fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.legend()
    save(fig, NB, "linear_ae_vs_pca")


def fig_denoising_manifold():
    banner("Figure denoising_manifold: the optimal denoiser points back onto the data manifold")
    sigma = 0.35
    curve_x = np.linspace(0, 2 * np.pi, 2000)
    curve = np.stack([curve_x, np.sin(curve_x)], 1)                # dense samples of the manifold
    n = 200
    clean = curve[rng.integers(0, len(curve), n)]
    noisy = clean + sigma * rng.normal(size=(n, 2))
    # optimal denoiser for Gaussian corruption = posterior mean of the clean point given the noisy one
    d2 = ((noisy[:, None, :] - curve[None, :, :]) ** 2).sum(-1)      # (n, 2000)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * sigma ** 2))
    w /= w.sum(1, keepdims=True)
    denoised = w @ curve
    dist = lambda P: np.sqrt(((P[:, None, :] - curve[None, :, :]) ** 2).sum(-1).min(1))
    print(f"noise std {sigma}; mean distance to the curve: noisy {dist(noisy).mean():.3f} -> denoised {dist(denoised).mean():.3f}")
    print(f"mean squared error to the clean point: noisy {np.mean(((noisy-clean)**2).sum(1)):.3f} -> denoised {np.mean(((denoised-clean)**2).sum(1)):.3f}")
    for i in range(3):
        print(f"  example {i}: clean ({clean[i,0]:.2f}, {clean[i,1]:.2f})  noisy ({noisy[i,0]:.2f}, {noisy[i,1]:.2f})  denoised ({denoised[i,0]:.2f}, {denoised[i,1]:.2f})")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(curve[:, 0], curve[:, 1], color=BLUE, label="data manifold (clean points live here)")
    ax.scatter(noisy[:, 0], noisy[:, 1], s=10, color=ORANGE, alpha=0.7, label="corrupted inputs")
    sel = rng.choice(n, 25, replace=False)
    for i in sel:
        ax.annotate("", xy=denoised[i], xytext=noisy[i], arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.0))
    ax.scatter([], [], marker=r"$\rightarrow$", s=80, color=AQUA, label="optimal denoiser (points to the manifold)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Denoising is manifold learning")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(-2.2, 2.4)
    save(fig, NB, "denoising_manifold")


def auc_rank(scores, labels):
    """ROC-AUC by the rank statistic (probability that a positive outscores a negative)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def roc(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    tpr = np.concatenate([[0], np.cumsum(l) / l.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - l) / (1 - l).sum()])
    return fpr, tpr


def fig_anomaly_detection():
    banner("Figure anomaly_detection: reconstruction error as an anomaly score (PCA-3 autoencoder on rank-3 data)")
    D, r = 10, 3
    A = rng.normal(size=(r, D)); B = rng.normal(size=(r, D))
    make = lambda M, n, scale=1.0: scale * (rng.normal(size=(n, r)) @ M) + 0.15 * rng.normal(size=(n, D))
    X_train = make(A, 500)
    X_normal = make(A, 500)
    X_anom = make(B, 500)
    X_anom_low = make(B, 500, scale=0.25)                           # "simple" anomalies with little energy
    mu = X_train.mean(0)
    _, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
    V = Vt[:r].T
    err = lambda X: np.sum(((X - mu) - (X - mu) @ V @ V.T) ** 2, 1)
    e_tr, e_n, e_a, e_al = err(X_train), err(X_normal), err(X_anom), err(X_anom_low)
    thr = np.quantile(e_tr, 0.95)
    lab = np.r_[np.zeros(500), np.ones(500)]
    auc_a, auc_al = auc_rank(np.r_[e_n, e_a], lab), auc_rank(np.r_[e_n, e_al], lab)
    print(f"sample scores: normal {e_n[:4]}  anomalous {e_a[:4]}  low-energy anomalous {e_al[:4]}")
    print(f"mean reconstruction error: normal {e_n.mean():.3f} | anomalous {e_a.mean():.3f} | low-energy anomalous {e_al.mean():.3f}")
    print(f"threshold = 95th percentile of training errors = {thr:.3f}: flags {np.mean(e_n > thr):.3f} of normal, "
          f"{np.mean(e_a > thr):.3f} of anomalous, {np.mean(e_al > thr):.3f} of low-energy anomalous")
    print(f"ROC-AUC: anomalies {auc_a:.4f} | low-energy anomalies {auc_al:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    all_e = np.r_[e_n, e_a, e_al]
    bins = np.logspace(np.log10(all_e.min() * 0.9), np.log10(all_e.max() * 1.1), 45)
    ax.hist(e_n, bins=bins, color=BLUE, alpha=0.75, label="normal test points")
    ax.hist(e_a, bins=bins, color=ORANGE, alpha=0.75, label="anomalies")
    ax.hist(e_al, bins=bins, color=AQUA, alpha=0.75, label="low-energy anomalies")
    ax.set_xscale("log")
    ax.axvline(thr, color=INK_2, ls="--", lw=1)
    ax.annotate(f"threshold {thr:.2f}", (thr, ax.get_ylim()[1] * 0.9), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("reconstruction error (squared, per point, log scale)"); ax.set_ylabel("count"); ax.set_title("Score distributions"); ax.legend(fontsize=8)
    ax = axes[1]
    f1, t1 = roc(np.r_[e_n, e_a], lab); f2, t2 = roc(np.r_[e_n, e_al], lab)
    ax.plot(f1, t1, color=ORANGE, label=f"anomalies (AUC = {auc_a:.3f})")
    ax.plot(f2, t2, color=AQUA, label=f"low-energy anomalies (AUC = {auc_al:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC curves"); ax.legend(fontsize=8, loc="lower right")
    save(fig, NB, "anomaly_detection")


# ----------------------------------------------------------------------------------------------
# Section 3 · holes in an autoencoder latent space versus a Gaussian-filled one
# ----------------------------------------------------------------------------------------------
def fig_latent_holes():
    banner("Figure latent_holes: clustered AE-style codes with holes vs Gaussian-filled VAE-style codes")
    n_codes, k = 2000, 10
    centres = rng.uniform(-4, 4, size=(k, 2))
    z_ae = centres[rng.integers(0, k, n_codes)] + 0.25 * rng.normal(size=(n_codes, 2))
    z_vae = rng.normal(size=(n_codes, 2))
    lo, hi = z_ae.min(0), z_ae.max(0)
    rand_ae = lo + (hi - lo) * rng.uniform(size=(40, 2))              # uniform in the bounding box
    rand_vae = rng.normal(size=(40, 2))                               # draws from the prior
    def empty_fraction(z):
        H, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=30)
        return (H == 0).mean()
    nn_dist = lambda q, z: np.sqrt(((q[:, None, :] - z[None, :, :]) ** 2).sum(-1).min(1))
    hole_ae, hole_vae = np.mean(nn_dist(rand_ae, z_ae) > 0.5), np.mean(nn_dist(rand_vae, z_vae) > 0.5)
    print(f"empty cells in a 30x30 grid over the bounding box: AE-style {empty_fraction(z_ae):.2f} | VAE-style {empty_fraction(z_vae):.2f}")
    print(f"random points farther than 0.5 from every code ('in a hole'): AE-style {hole_ae:.2f} | VAE-style {hole_vae:.2f}")
    print(f"first 3 random AE-box points and their nearest-code distance: {np.c_[rand_ae[:3], nn_dist(rand_ae[:3], z_ae)]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, z, rnd, title in [(axes[0], z_ae, rand_ae, "Autoencoder-style codes: clusters and holes"),
                              (axes[1], z_vae, rand_vae, "VAE-style codes: the prior is filled")]:
        ax.scatter(z[:, 0], z[:, 1], s=4, color=BLUE, alpha=0.4, label="codes of training images")
        ax.scatter(rnd[:, 0], rnd[:, 1], marker="x", s=45, color=ORANGE, label="random points to decode")
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title(title); ax.set_aspect("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, NB, "latent_holes", tight=False)


# ----------------------------------------------------------------------------------------------
# Section 4 · VAE theory: KL closed form, reparameterization, ELBO gap
# ----------------------------------------------------------------------------------------------
def fig_kl_closed_form():
    banner("Figure kl_closed_form: KL( N(mu, sigma^2) || N(0, 1) ) as a function of mu and sigma")
    mus = np.array([0.5, -1.0, 2.0]); logvars = np.array([0.0, -1.0, 0.5])
    sig = np.exp(0.5 * logvars)
    kl3 = kl_gauss(mus, np.exp(logvars))
    print("notebook cell-36 cases (mu, logvar -> sigma, KL):")
    for m, lv, s, k in zip(mus, logvars, sig, kl3):
        print(f"  mu={m:+.2f} logvar={lv:+.2f} sigma={s:.3f} -> 0.5*(mu^2 + sigma^2 - 1 - log sigma^2) = {k:.4f}")
    for s in [0.1, 0.5, 1.0, 2.0]:
        print(f"  mu=0, sigma={s}: KL = {kl_gauss(0.0, s**2):.4f}")
    print(f"  mu=1, sigma=1: KL = {kl_gauss(1.0, 1.0):.4f} (= mu^2/2)")
    M, Sg = np.meshgrid(np.linspace(-3, 3, 241), np.linspace(0.1, 3, 241))
    KL = kl_gauss(M, Sg ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    im = ax.pcolormesh(M, Sg, np.minimum(KL, 6), cmap=SEQ_CMAP, shading="auto", vmin=0, vmax=6)
    cs = ax.contour(M, Sg, KL, levels=[0.25, 0.5, 1, 2, 4], colors=SURFACE, linewidths=0.8)
    ax.clabel(cs, fmt="%.2g", fontsize=7, colors=SURFACE)
    ax.scatter(mus, sig, color=ORANGE, s=45, zorder=3, label="cell-36 examples")
    for m, s, k in zip(mus, sig, kl3):
        ax.annotate(f"{k:.2f}", (m, s), xytext=(6, 4), textcoords="offset points", fontsize=8, color=SURFACE)
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\sigma$"); ax.set_title("KL to the prior (nats, clipped at 6)"); ax.grid(False)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="nats")
    ax = axes[1]
    s = np.linspace(0.1, 3, 300)
    for i, m in enumerate([0.0, 1.0, 2.0]):
        ax.plot(s, kl_gauss(m, s ** 2), color=SERIES[i], label=fr"$\mu$ = {m:.0f}")
    ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("KL (nats)"); ax.set_ylim(0, 6); ax.set_title("Zero only at $\\mu=0,\\ \\sigma=1$")
    ax.axvline(1.0, color=MUTED, ls=":", lw=1); ax.legend()
    save(fig, NB, "kl_closed_form")


def fig_reparameterization():
    banner("Figure reparameterization: z = mu + sigma*eps, and the two gradient estimators for f(z) = z^2")
    n = 200_000
    eps = rng.normal(size=n)
    mu_s, sd_s = 1.5, 0.5
    z_s = mu_s + sd_s * eps
    print(f"eps ~ N(0,1): mean {eps.mean():+.3f} std {eps.std():.3f};  z = {mu_s} + {sd_s} eps: mean {z_s.mean():.3f} std {z_s.std():.3f}")
    print(f"first 4 eps: {eps[:4]} -> z: {z_s[:4]}")
    mu_t = 1.0
    z = mu_t + eps
    g_rep, g_sf = 2 * z, z ** 2 * (z - mu_t)
    print(f"f(z)=z^2, z~N(mu,1), mu={mu_t}: exact gradient 2*mu = {2*mu_t:.3f}")
    print(f"reparameterization 2z          : mean {g_rep.mean():.3f}  std {g_rep.std():.3f}  (analytic std 2)")
    print(f"score function z^2 (z-mu)      : mean {g_sf.mean():.3f}  std {g_sf.std():.3f}  (analytic std sqrt(30) = {np.sqrt(30):.3f})")
    print(f"std ratio {g_sf.std()/g_rep.std():.2f}, variance ratio {g_sf.var()/g_rep.var():.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    b = np.linspace(-4, 4, 81)
    ax.hist(eps, bins=b, density=True, color=BLUE, alpha=0.75, label=r"$\epsilon \sim \mathcal{N}(0, 1)$ (no parameters)")
    ax.hist(z_s, bins=b, density=True, color=ORANGE, alpha=0.75, label=fr"$z = \mu + \sigma\epsilon$, $\mu$={mu_s}, $\sigma$={sd_s}")
    ax.set_xlabel("value"); ax.set_ylabel("density"); ax.set_title("Sampling as a differentiable function of noise"); ax.legend(fontsize=8)
    ax = axes[1]
    b = np.linspace(-15, 25, 121)
    ax.hist(g_rep, bins=b, density=True, color=BLUE, alpha=0.75, label=f"reparameterization: std {g_rep.std():.2f}")
    ax.hist(g_sf, bins=b, density=True, color=ORANGE, alpha=0.6, label=f"score function: std {g_sf.std():.2f}")
    ax.axvline(2 * mu_t, color=INK_2, ls="--", lw=1)
    ax.annotate("exact 2$\\mu$ = 2", (2, ax.get_ylim()[1] * 0.92), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("single-sample gradient estimate"); ax.set_ylabel("density"); ax.set_xlim(-12, 18)
    ax.set_title("Same mean, very different variance"); ax.legend(fontsize=8, loc="center right")
    save(fig, NB, "reparameterization")


def fig_elbo_gap():
    banner("Figure elbo_gap: log p(x) = ELBO + KL(q || true posterior) on a 1-D linear-Gaussian model")
    a, s, x = 1.5, 0.7, 2.0                                 # z ~ N(0,1), x | z ~ N(a z, s^2), observed x
    var_post = 1 / (1 + a ** 2 / s ** 2); mu_post = var_post * a * x / s ** 2
    log_px = -0.5 * np.log(2 * np.pi * (a ** 2 + s ** 2)) - x ** 2 / (2 * (a ** 2 + s ** 2))
    def elbo(m, v):
        rec = -0.5 * np.log(2 * np.pi * s ** 2) - ((x - a * m) ** 2 + a ** 2 * v) / (2 * s ** 2)
        return rec - kl_gauss(m, v), rec, kl_gauss(m, v)
    def kl_to_post(m, v):
        return 0.5 * (np.log(var_post / v) + (v + (m - mu_post) ** 2) / var_post - 1)
    print(f"model: a={a}, s={s}, x={x}: true posterior N({mu_post:.4f}, {var_post:.4f}); log p(x) = {log_px:.4f}")
    print(f"{'q mean':>7s} {'q var':>6s} {'E_q log p(x|z)':>15s} {'KL(q||prior)':>13s} {'ELBO':>8s} {'KL(q||post)':>12s} {'ELBO+gap':>9s}")
    for m, v in [(mu_post, var_post), (0.3, var_post), (0.3, 0.5), (0.0, 1.0)]:
        e, rec, klp = elbo(m, v); g = kl_to_post(m, v)
        print(f"{m:7.3f} {v:6.3f} {rec:15.4f} {klp:13.4f} {e:8.4f} {g:12.4f} {e+g:9.4f}")
    ms = np.linspace(-1, 3, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ms, elbo(ms, var_post)[0], color=BLUE, label=f"ELBO, q variance = posterior variance ({var_post:.3f})")
    ax.plot(ms, elbo(ms, 1.0)[0], color=ORANGE, label="ELBO, q variance = 1")
    ax.axhline(log_px, color=AQUA, ls="--", label=f"log p(x) = {log_px:.3f} (the evidence)")
    m0 = 0.3; e0 = elbo(m0, var_post)[0]
    ax.annotate("", xy=(m0, log_px), xytext=(m0, e0), arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1))
    ax.annotate(f"gap = KL(q || p(z|x))\n= {kl_to_post(m0, var_post):.2f} nats", (m0, (log_px + e0) / 2), xytext=(-120, -10),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.scatter([mu_post], [log_px], color=INK, s=40, zorder=3)
    ax.annotate("q = true posterior: gap 0", (mu_post, log_px), xytext=(8, -16), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean of the approximate posterior q(z|x)"); ax.set_ylabel("nats"); ax.set_ylim(-9, -1)
    ax.set_title("The ELBO is below the evidence by exactly the posterior mismatch"); ax.legend(fontsize=8, loc="lower right")
    save(fig, NB, "elbo_gap")


# ----------------------------------------------------------------------------------------------
# Section 5 · toy linear-Gaussian VAE: rate-distortion for beta, KL annealing
# ----------------------------------------------------------------------------------------------
def toy_vae_terms(params, var_x, s):
    """Expected rate and distortion of a 1-D linear-Gaussian VAE under data x ~ N(0, var_x).

    encoder q(z|x) = N(a x, v), decoder p(x|z) = N(w z, s^2), prior N(0, 1).
    """
    a, w, logv = params
    v = np.exp(logv)
    R = 0.5 * (a ** 2 * var_x + v - 1 - logv)
    Dq = ((1 - w * a) ** 2 * var_x + w ** 2 * v) / (2 * s ** 2) + 0.5 * np.log(2 * np.pi * s ** 2)
    return R, Dq


def toy_vae_grads(params, var_x, s, beta):
    a, w, logv = params
    v = np.exp(logv)
    dD_da = (-2 * (1 - w * a) * w * var_x) / (2 * s ** 2)
    dD_dw = (-2 * (1 - w * a) * a * var_x + 2 * w * v) / (2 * s ** 2)
    dD_dlogv = (w ** 2 * v) / (2 * s ** 2)
    dR_da = a * var_x
    dR_dlogv = 0.5 * (v - 1)
    return np.array([dD_da + beta * dR_da, dD_dw, dD_dlogv + beta * dR_dlogv])


def train_toy_vae(var_x, s, beta_fn, steps=3000, lr=0.02, init=(0.1, 0.1, 0.0)):
    p = np.array(init, dtype=float)
    opt = Adam([p], lr=lr)
    hist = []
    for t in range(steps):
        b = beta_fn(t, steps)
        opt.step([toy_vae_grads(p, var_x, s, b)])
        hist.append((*toy_vae_terms(p, var_x, s), b))
    return p, np.array(hist)


def fig_beta_rate_distortion():
    banner("Figure beta_rate_distortion: the beta sweep on a toy linear-Gaussian VAE (closed-form objective)")
    var_x, s = 4.0, 0.5
    betas = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    print(f"data x ~ N(0, {var_x}); decoder noise s = {s}; minimise D + beta * R over encoder slope a, encoder variance v, decoder weight w")
    print(f"{'beta':>5s} {'rate R (KL)':>12s} {'distortion D':>13s} {'R + D':>8s} {'enc. slope a':>13s} {'enc. var v':>11s} {'dec. w':>7s}")
    pts = {}
    for b in betas:
        p, _ = train_toy_vae(var_x, s, lambda t, T, b=b: b)
        R, Dq = toy_vae_terms(p, var_x, s)
        pts[b] = (R, Dq)
        print(f"{b:5.2f} {R:12.4f} {Dq:13.4f} {R+Dq:8.4f} {p[0]:13.4f} {np.exp(p[2]):11.4f} {p[1]:7.4f}")
    best = min(pts, key=lambda b: sum(pts[b]))
    print(f"smallest R + D (the negative ELBO) at beta = {best}: only beta = 1 optimises the bound itself")
    print(f"rate of the collapsed model: {pts[16.0][0]:.4f} nats; distortion of the collapsed model: {pts[16.0][1]:.4f} "
          f"(= var_x / (2 s^2) + 0.5 log(2 pi s^2) = {var_x/(2*s**2) + 0.5*np.log(2*np.pi*s**2):.4f})")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    Rs = [pts[b][0] for b in betas]; Ds = [pts[b][1] for b in betas]
    ax.plot(Rs, Ds, "-", color=MUTED, lw=1)
    ax.scatter(Rs, Ds, color=BLUE, s=55, zorder=3)
    for b in [0.25, 1.0, 4.0, 16.0]:
        ax.annotate(fr"$\beta$ = {b:g}", pts[b], xytext=(8, 6), textcoords="offset points", fontsize=8)
    # tangent of slope -beta at beta = 1
    R1, D1 = pts[1.0]
    rr = np.linspace(max(0, R1 - 0.8), R1 + 0.8, 10)
    ax.plot(rr, D1 - 1.0 * (rr - R1), ls=":", color=ORANGE, lw=1.2)
    ax.annotate(r"slope $-\beta$ at $\beta$ = 1", (rr[-1], D1 - (rr[-1] - R1)), xytext=(4, -4), textcoords="offset points", fontsize=8, color=ORANGE)
    ax.set_xlabel("rate R = KL(q(z|x) || p(z)) in nats"); ax.set_ylabel("distortion D = reconstruction NLL in nats")
    ax.set_ylim(-0.8, 9); ax.set_title(r"$\beta$ walks along the rate-distortion curve")
    save(fig, NB, "beta_rate_distortion")


def fig_kl_annealing():
    banner("Figure kl_annealing: the warm-up schedule and its effect on the toy VAE's KL trajectory")
    T = 3000
    linear_warmup = lambda t, total, frac=0.5: min(1.0, t / max(1.0, frac * total))
    print("linear warm-up over the first half of training, beta at steps 0, 300, 750, 1500, 2250:",
          [round(linear_warmup(t, T), 2) for t in [0, 300, 750, 1500, 2250]])
    var_x, s = 4.0, 0.5
    init = (0.02, 0.02, 0.0)                                  # start almost exactly at the prior (collapsed)
    _, h_const = train_toy_vae(var_x, s, lambda t, T_: 1.0, steps=T, lr=0.01, init=init)
    _, h_ann = train_toy_vae(var_x, s, lambda t, T_: linear_warmup(t, T_), steps=T, lr=0.01, init=init)
    for name, h in [("constant beta = 1", h_const), ("annealed beta", h_ann)]:
        i10 = np.argmax(h[:, 0] > 0.1) if (h[:, 0] > 0.1).any() else -1
        print(f"{name:18s}: KL at step 200 {h[199,0]:.4f} | step 1000 {h[999,0]:.4f} | final KL {h[-1,0]:.4f}, final D {h[-1,1]:.4f}, "
              f"first step with KL > 0.1: {i10}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    steps = np.arange(T)
    ax.plot(steps, h_const[:, 2], color=BLUE, label=r"constant $\beta$ = 1")
    ax.plot(steps, h_ann[:, 2], color=ORANGE, label="linear warm-up over the first half")
    ax.set_xlabel("training step"); ax.set_ylabel(r"KL weight $\beta_t$"); ax.set_ylim(-0.05, 1.1); ax.set_title("KL annealing schedule"); ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(steps, h_const[:, 0], color=BLUE, label=r"constant $\beta$ = 1")
    ax.plot(steps, h_ann[:, 0], color=ORANGE, label="annealed")
    ax.set_xlabel("training step"); ax.set_ylabel("KL term (rate, nats)"); ax.set_title("Toy VAE started at the prior: KL trajectory"); ax.legend()
    save(fig, NB, "kl_annealing")


# ----------------------------------------------------------------------------------------------
# Section 6 · conditional VAE on a toy with classes and a style factor
# ----------------------------------------------------------------------------------------------
def fig_cvae_toy():
    banner("Figure cvae_toy: conditioning on the class removes identity from the code (toy linear-Gaussian VAE)")
    centres = np.array([-6.0, -2.0, 2.0, 6.0]); K = len(centres)
    n = 4000
    y = rng.integers(0, K, n)
    style = rng.normal(size=n)                                    # 1-D "style" factor, variance 1
    x = centres[y] + style
    var_total, var_style = x.var(), (x - centres[y]).var()
    print(f"{K} classes at centres {centres}; style variance {var_style:.3f}; total variance of x {var_total:.3f}")
    print(f"sample rows (y, style, x): " + "; ".join(f"({y[i]}, {style[i]:+.2f}, {x[i]:+.2f})" for i in range(5)))
    s = 0.5
    p_u, _ = train_toy_vae(var_total, s, lambda t, T: 1.0)        # unconditional: must encode x itself
    p_c, _ = train_toy_vae(var_style, s, lambda t, T: 1.0)        # conditional: decoder knows the centre, code carries style only
    Ru, Du = toy_vae_terms(p_u, var_total, s); Rc, Dc = toy_vae_terms(p_c, var_style, s)
    print(f"unconditional VAE: rate {Ru:.3f} nats, distortion {Du:.3f} nats | conditional VAE: rate {Rc:.3f} nats, distortion {Dc:.3f} nats")
    print(f"rate saved by giving the label: {Ru-Rc:.3f} nats; information content of a uniform label log {K} = {np.log(K):.3f} nats; "
          f"notebook's MNIST estimate log 10 = {np.log(10):.3f} nats")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    b = np.linspace(-10, 10, 81)
    for k in range(K):
        ax.hist(x[y == k], bins=b, color=SERIES[k], alpha=0.8, label=f"class {k} (centre {centres[k]:+.0f})")
    ax.set_xlabel("x = class centre + style"); ax.set_ylabel("count"); ax.set_title("Identity (class) plus style (within-class spread)"); ax.legend(fontsize=8)
    ax = axes[1]
    xpos = np.arange(2)
    ax.bar(xpos - 0.2, [Ru, Rc], width=0.4, color=BLUE, label="rate (KL)", edgecolor=SURFACE)
    ax.bar(xpos + 0.2, [Du, Dc], width=0.4, color=ORANGE, label="distortion (rec. NLL)", edgecolor=SURFACE)
    ax.set_xticks(xpos); ax.set_xticklabels(["unconditional", "conditional on y"])
    ax.set_ylabel("nats"); ax.set_title("Code budget with and without the label"); ax.legend(fontsize=8)
    ax.annotate(f"{Ru-Rc:.2f} nats saved", (0.5, max(Ru, Rc) * 1.02), ha="center", fontsize=8, color=INK_2)
    save(fig, NB, "cvae_toy")


# ----------------------------------------------------------------------------------------------
# Section 7 · Fréchet distance between feature clouds
# ----------------------------------------------------------------------------------------------
def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    S1, S2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    def sqrtm_psd(M):
        w, V = np.linalg.eigh((M + M.T) / 2); return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T
    r1 = sqrtm_psd(S1)
    cross = sqrtm_psd(r1 @ S2 @ r1)                              # tr((S1 S2)^1/2) = tr((S1^1/2 S2 S1^1/2)^1/2)
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * cross))


def fig_frechet_toy():
    banner("Figure frechet_toy: Fréchet distance between 2-D feature clouds of 'real' data and three generators")
    centres = np.array([[-3, 0], [3, 0], [0, 3.5]], dtype=float)
    def mixture(n, which=(0, 1, 2), spread=1.0, shrink=0.0):
        lab = rng.choice(which, n)
        c = centres[lab] * (1 - shrink)
        return c + spread * rng.normal(size=(n, 2)), lab
    n = 3000
    real, _ = mixture(n)
    ref, _ = mixture(n)
    sets = {
        "good generator": mixture(n)[0],
        "mode collapse (one class)": mixture(n, which=(0,), spread=0.6)[0],
        "blurry, shrunk towards the mean": mixture(n, spread=1.8, shrink=0.6)[0],
    }
    classify = lambda F: np.argmin(((F[:, None, :] - centres[None, :, :]) ** 2).sum(-1), 1)
    def entropy(F):
        p = np.bincount(classify(F), minlength=3) / len(F); p = p[p > 0]; return float(max(0.0, -(p * np.log(p)).sum()))
    print(f"{'set':34s} {'Frechet dist.':>13s} {'class entropy':>14s}  (max entropy log 3 = {np.log(3):.3f})")
    results = {"real (second sample)": (frechet(ref, real), entropy(ref))}
    print(f"{'real (second sample)':34s} {results['real (second sample)'][0]:13.3f} {results['real (second sample)'][1]:14.3f}")
    for name, F in sets.items():
        results[name] = (frechet(F, real), entropy(F))
        print(f"{name:34s} {results[name][0]:13.3f} {results[name][1]:14.3f}")
    print(f"means: real {real.mean(0)}, collapsed {sets['mode collapse (one class)'].mean(0)}, blurry {sets['blurry, shrunk towards the mean'].mean(0)}")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
    panels = [("real features", real)] + list(sets.items())
    for ax, (name, F) in zip(axes, panels):
        ax.scatter(real[:, 0], real[:, 1], s=3, color=MUTED, alpha=0.25)
        ax.scatter(F[:, 0], F[:, 1], s=3, color=BLUE if name == "real features" else ORANGE, alpha=0.4)
        fd = results["real (second sample)"][0] if name == "real features" else results[name][0]
        ent = results["real (second sample)"][1] if name == "real features" else results[name][1]
        ax.set_title(f"{name}\nFD {fd:.2f}, entropy {ent:.2f}", fontsize=9)
        ax.set_xlabel("feature 1"); ax.set_aspect("equal")
    axes[0].set_ylabel("feature 2")
    fig.suptitle("Grey: real reference cloud. Colour: the set being scored", fontsize=9, color=INK_2, y=1.02)
    save(fig, NB, "frechet_toy")


# ----------------------------------------------------------------------------------------------
# Section 8 · VQ codebook; linear probe on load_digits
# ----------------------------------------------------------------------------------------------
def fig_vq_codebook():
    banner("Figure vq_codebook: nearest-neighbour quantisation of 2-D encoder outputs, codebook/commitment losses, usage perplexity")
    n, K = 300, 8
    blobs = np.array([[-2, -1], [2, 1.5], [0.5, -2.5]], dtype=float)
    z_e = blobs[rng.integers(0, 3, n)] + 0.6 * rng.normal(size=(n, 2))
    book = rng.uniform(-1 / K, 1 / K, size=(K, 2))                      # the notebook's uniform(-1/K, 1/K) init
    beta = 0.25
    def assign(z, cb): return np.argmin(((z[:, None, :] - cb[None, :, :]) ** 2).sum(-1), 1)
    def losses(z, cb):
        idx = assign(z, cb); zq = cb[idx]
        codebook_loss = np.mean((zq - z) ** 2)                            # gradient flows to the codebook
        commitment = np.mean((z - zq) ** 2)                               # same value, gradient flows to the encoder
        return idx, zq, codebook_loss, commitment
    idx0, _, cl0, cm0 = losses(z_e, book)
    print(f"initial codebook (uniform in +-1/{K}): codebook loss {cl0:.3f}, commitment loss {cm0:.3f}, total VQ loss = cl + {beta}*cm = {cl0 + beta*cm0:.3f}; "
          f"codes in use {len(np.unique(idx0))}/{K}")
    opt = Adam([book], lr=0.05)
    for t in range(400):                                                 # codebook loss gradient: pull each code to its assigned encoder outputs
        idx = assign(z_e, book)
        g = np.zeros_like(book)
        for k in range(K):
            m = idx == k
            if m.any():
                g[k] = -2 * (z_e[m] - book[k]).sum(0) / n
        opt.step([g])
    idx, z_q, cl, cm = losses(z_e, book)
    counts = np.bincount(idx, minlength=K); p = counts / counts.sum()
    perp = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    print(f"after 400 codebook steps: codebook loss {cl:.3f}, commitment loss {cm:.3f}, total {cl + beta*cm:.3f}")
    print(f"code usage counts {counts}; codes in use {int((counts > 0).sum())}/{K}; usage perplexity {perp:.2f} (max {K})")
    i = int(np.argmax(((z_e - z_q) ** 2).sum(1)))                      # the point farthest from its code
    print(f"example: encoder output z_e = ({z_e[i,0]:.2f}, {z_e[i,1]:.2f}) -> nearest code {idx[i]} at ({z_q[i,0]:.2f}, {z_q[i,1]:.2f}); "
          f"squared snap distance {((z_e[i]-z_q[i])**2).sum():.3f}; straight-through: forward value z_q, backward gradient copied to z_e")
    print(f"bits per image for a 7x7 grid with K=64 codes: 49 * log2(64) = {49*np.log2(64):.0f}")
    # Voronoi background
    g1, g2 = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-4.5, 4.5, 300))
    grid = np.stack([g1.ravel(), g2.ravel()], 1)
    region = assign(grid, book).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(region, origin="lower", extent=[-4.5, 4.5, -4.5, 4.5], cmap=ListedColormap(SERIES[:K]), alpha=0.22, interpolation="nearest", vmin=-0.5, vmax=K - 0.5)
    ax.scatter(z_e[:, 0], z_e[:, 1], s=8, color=INK_2, alpha=0.6, label="encoder outputs $z_e$")
    ax.scatter(book[:, 0], book[:, 1], marker="D", s=70, color=[SERIES[k] for k in range(K)], edgecolor=INK, linewidth=0.8, label="codebook vectors $e_k$", zorder=3)
    far = np.argsort(-((z_e - z_q) ** 2).sum(1))[:3]
    for j in far:
        ax.annotate("", xy=z_q[j], xytext=z_e[j], arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.annotate("snap to the\nnearest code", z_e[i], xytext=(10, -22), textcoords="offset points", fontsize=8, color=INK_2, ha="left")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title(f"Vector quantisation with K = {K} codes (perplexity {perp:.1f})")
    ax.legend(loc="upper left", fontsize=8); ax.grid(False)
    save(fig, NB, "vq_codebook")


def softmax_probe(Xtr, ytr, Xte, yte, steps=400, lr=0.1, l2=1e-3):
    m, sd = Xtr.mean(0), Xtr.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
    n, d = Xtr.shape; C = 10
    W = np.zeros((d, C)); b = np.zeros(C)
    Y = np.eye(C)[ytr]
    opt = Adam([W, b], lr=lr)
    for _ in range(steps):
        logits = Xtr @ W + b; logits -= logits.max(1, keepdims=True)
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        opt.step([Xtr.T @ G + l2 * W, G.sum(0)])
    return float(np.mean(np.argmax(Xte @ W + b, 1) == yte))


def linear_probe_digits():
    banner("Linear probe on scikit-learn's bundled 8x8 digits (1797 images): raw pixels vs PCA coordinates")
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    perm = rng.permutation(len(X)); tr, te = perm[:1200], perm[1200:]
    Xtr, Xte = X[tr], X[te]
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    acc_raw = softmax_probe(Xtr, y[tr], Xte, y[te])
    print(f"train {len(tr)} / test {len(te)} images, {X.shape[1]} pixels each; softmax regression, 400 Adam steps")
    print(f"{'features':22s} {'probe accuracy':>15s}")
    print(f"{'raw pixels (64)':22s} {acc_raw:15.3f}")
    for k in [2, 4, 8, 16]:
        acc = softmax_probe((Xtr - mu) @ Vt[:k].T, y[tr], (Xte - mu) @ Vt[:k].T, y[te])
        print(f"{f'PCA ({k})':22s} {acc:15.3f}")
    print(f"latent diffusion compression: 512*512*3 / (64*64*4) = {512*512*3/(64*64*4):.0f}x fewer numbers")


def main():
    fig_manifold_hypothesis()
    fig_linear_ae_vs_pca()
    fig_denoising_manifold()
    fig_anomaly_detection()
    fig_latent_holes()
    fig_kl_closed_form()
    fig_reparameterization()
    fig_elbo_gap()
    fig_beta_rate_distortion()
    fig_kl_annealing()
    fig_cvae_toy()
    fig_frechet_toy()
    fig_vq_codebook()
    linear_probe_digits()


if __name__ == "__main__":
    main()
