"""Figures for the companion guide of notebook 17 (Capstone Projects).

numpy + matplotlib (+ scikit-learn for two small classifiers on synthetic data) only, no PyTorch and no
downloads. Each fig_* function builds its data, prints every number quoted in the guide, and saves one
PNG under notebooks/figures/17/. Where a curve is an illustration rather than a measurement, its title
and the printed banner say so.

The synthetic-shapes dataset, the time series, the paired-bootstrap experiment, the diffusion schedule
and the FrozenLake Q-learning run replicate the notebook's own numpy code with the notebook's seeds, so
the printed values are the ones the notebook produces on its offline fallback path.
"""
import sys, pathlib, math, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

NB = "17"
SEED = 42            # the notebook's seed


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# Shared data: the notebook's synthetic-shapes fallback dataset (cell 3), replicated exactly
# --------------------------------------------------------------------------------------
CLASS_NAMES = ["filled square", "hollow square", "disc", "plus"]


def make_synthetic_shapes(n, seed=SEED):
    r_ = np.random.default_rng(seed)
    X = np.zeros((n, 28, 28), dtype=np.float32)
    y = r_.integers(0, 4, size=n)
    yy, xx = np.mgrid[0:28, 0:28]
    for i in range(n):
        cx, cy, r = r_.integers(9, 19), r_.integers(9, 19), r_.integers(4, 8)
        if y[i] == 0:
            mask = (abs(xx - cx) <= r) & (abs(yy - cy) <= r)
        elif y[i] == 1:
            mask = (abs(xx - cx) <= r) & (abs(yy - cy) <= r) & ~((abs(xx - cx) <= r - 2) & (abs(yy - cy) <= r - 2))
        elif y[i] == 2:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        else:
            mask = ((abs(xx - cx) <= 1) & (abs(yy - cy) <= r)) | ((abs(yy - cy) <= 1) & (abs(xx - cx) <= r))
        X[i] = mask * r_.uniform(0.6, 1.0) + r_.normal(0, 0.08, size=(28, 28))
    return np.clip(X, 0, 1), y


_SHAPES = {}


def shapes_data():
    """FAST_RUN sizes of the notebook: 3000 train (seed 42), 1000 val (seed 43)."""
    if not _SHAPES:
        Xtr, ytr = make_synthetic_shapes(3000, SEED)
        Xva, yva = make_synthetic_shapes(1000, SEED + 1)
        _SHAPES.update(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    return _SHAPES["Xtr"], _SHAPES["ytr"], _SHAPES["Xva"], _SHAPES["yva"]


# --------------------------------------------------------------------------------------
# Section 1: data audit (class balance, pixel range, duplicates, leakage)
# --------------------------------------------------------------------------------------
def fig_data_audit():
    banner("Section 1.2: data audit of the synthetic-shapes fallback (3000 train / 1000 val)")
    import hashlib
    Xtr, ytr, Xva, yva = shapes_data()
    counts = np.bincount(ytr, minlength=4)
    for c, n in zip(CLASS_NAMES, counts):
        print(f"  train class '{c}': {n} examples ({100 * n / len(ytr):.1f} %)")
    maj = counts.argmax()
    print(f"  majority class: '{CLASS_NAMES[maj]}' share {counts.max() / len(ytr):.3f}")
    print(f"  train range [{Xtr.min():.2f}, {Xtr.max():.2f}], mean {Xtr.mean():.3f}, std {Xtr.std():.3f}")
    h_tr = [hashlib.md5(x.tobytes()).hexdigest() for x in Xtr]
    h_va = [hashlib.md5(x.tobytes()).hexdigest() for x in Xva]
    dup = len(h_tr) - len(set(h_tr)); leak = len(set(h_tr) & set(h_va))
    print(f"  exact duplicates within train: {dup}; train/val leakage: {leak}")
    frac_zero = (Xtr == 0).mean(); frac_one = (Xtr == 1).mean()
    print(f"  share of pixels clipped to 0: {100 * frac_zero:.1f} %, clipped to 1: {100 * frac_one:.2f} %")
    val_counts = np.bincount(yva, minlength=4)
    print("  val class counts:", dict(zip(CLASS_NAMES, val_counts.tolist())))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    bar(ax, [c.replace(" ", "\n") for c in CLASS_NAMES], counts, color=BLUE, label_fmt="{:d}")
    ax.axhline(len(ytr) / 4, color=INK_2, linestyle="--", linewidth=1)
    ax.annotate("perfect balance (750)", (3.45, len(ytr) / 4), ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylabel("training examples"); ax.set_ylim(0, 900)
    ax.set_title(f"Class balance (majority share {counts.max() / len(ytr):.3f})")
    ax = axes[1]
    ax.hist(Xtr.ravel(), bins=50, range=(0, 1), color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value"); ax.set_ylabel("pixels (log scale)")
    ax.set_title(f"Value range check: {dup} duplicates, {leak} leaked")
    ax.annotate(f"{100 * frac_zero:.0f} % of pixels\nclipped to 0", (0.03, Xtr.size * frac_zero), xytext=(0.25, Xtr.size * frac_zero / 3),
                fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    save(fig, NB, "data_audit_shapes")


# --------------------------------------------------------------------------------------
# Section 1/2: baseline ladder, computed on the shapes data (majority, centroid, logistic, MLP x 5 seeds)
# --------------------------------------------------------------------------------------
_LADDER = {}


def baseline_ladder_numbers():
    if _LADDER:
        return _LADDER
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    Xtr, ytr, Xva, yva = shapes_data()
    Ftr, Fva = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    maj = np.bincount(ytr).argmax()
    majority_acc = (yva == maj).mean()
    centroids = np.stack([Ftr[ytr == c].mean(0) for c in range(4)])
    d = ((Fva[:, None, :] - centroids[None]) ** 2).sum(-1)
    centroid_acc = (d.argmin(1) == yva).mean()
    lr = LogisticRegression(max_iter=300, C=0.05).fit(Ftr, ytr)
    linear_acc = lr.score(Fva, yva)
    mlp_accs, mlp_models = [], []
    for s in range(5):
        m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60, random_state=s).fit(Ftr, ytr)
        mlp_accs.append(m.score(Fva, yva)); mlp_models.append(m)
    _LADDER.update(majority=majority_acc, centroid=centroid_acc, linear=linear_acc,
                   mlp=np.array(mlp_accs), mlp_models=mlp_models, lr_model=lr)
    return _LADDER


def fig_baseline_ladder():
    banner("Section 1.3 / 2.3: baseline ladder on the shapes data (computed with scikit-learn)")
    L = baseline_ladder_numbers()
    print(f"  majority class accuracy:     {L['majority']:.3f}")
    print(f"  nearest centroid accuracy:   {L['centroid']:.3f}")
    print(f"  logistic regression (pixels): {L['linear']:.3f}")
    print(f"  MLP 784-64-4, 5 seeds:        {L['mlp'].mean():.3f} +- {L['mlp'].std(ddof=1):.3f} "
          f"(min {L['mlp'].min():.3f}, max {L['mlp'].max():.3f}); per seed {np.round(L['mlp'], 3).tolist()}")
    names = ["majority\nclass", "nearest\ncentroid", "logistic\nregression", "MLP (64 hidden)\nmean of 5 seeds"]
    vals = [L["majority"], L["centroid"], L["linear"], L["mlp"].mean()]
    errs = [0, 0, 0, L["mlp"].std(ddof=1)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, vals, width=0.6, color=[MUTED, SERIES[1], SERIES[0], SERIES[2]], edgecolor=SURFACE, linewidth=1.5,
                  yerr=errs, capsize=5, error_kw=dict(ecolor=INK_2, lw=1))
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + 0.02), ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_ylim(0, 1.12); ax.set_ylabel("validation accuracy (1000 images)")
    ax.set_title("Baselines first: the ladder every later result is judged against")
    save(fig, NB, "baseline_ladder")


# --------------------------------------------------------------------------------------
# Section 2: Trainer history with early stopping (illustrative) + cosine lr when resuming (computed)
# --------------------------------------------------------------------------------------
def cosine_lr(base, T, epochs):
    """Learning rate used in epoch e (1-based) of a CosineAnnealingLR schedule stepped once per epoch."""
    return [base * (1 + math.cos(math.pi * (e - 1) / T)) / 2 for e in range(1, epochs + 1)]


def fig_trainer_history_and_resume():
    banner("Section 2.1 / 2.4: early stopping on an illustrative history; cosine lr on resume (computed)")
    epochs = np.arange(1, 13)
    r_ = np.random.default_rng(3)
    val_acc = 0.90 - 0.22 * np.exp(-epochs / 2.2) + r_.normal(0, 0.004, len(epochs))
    val_acc[8:] -= 0.004 * np.arange(1, len(epochs) - 7)          # slight late overfitting
    patience = 3
    best, best_ep, bad, stop_ep = -1, 0, 0, None
    for e, a in zip(epochs, val_acc):
        if a > best:
            best, best_ep, bad = a, e, 0
        else:
            bad += 1
        if bad >= patience:
            stop_ep = e; break
    print("  illustrative val_acc per epoch:", [f"{a:.3f}" for a in val_acc[:stop_ep]])
    print(f"  best epoch {best_ep} (val_acc {best:.3f}); patience {patience} -> early stop after epoch {stop_ep}; "
          f"epochs saved: {len(epochs) - stop_ep} of {len(epochs)}")
    base = 2e-3
    lr2, lr3 = cosine_lr(base, 2, 2), cosine_lr(base, 3, 3)
    print(f"  cosine lr, budget 2 epochs: {[f'{v:.1e}' for v in lr2]} (reaches 0 after the 2nd step)")
    print(f"  cosine lr, budget 3 epochs: {[f'{v:.1e}' for v in lr3]}")
    print(f"  resumed run (2 done, budget 3): rebuilt schedule gives epoch-3 lr {lr3[2]:.1e} = 0.25 x base; "
          f"continuing the finished 2-epoch schedule would give 0.0")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(epochs[:stop_ep], val_acc[:stop_ep], marker="o", color=BLUE, label="val_acc (monitored)")
    ax.plot(epochs[stop_ep:], val_acc[stop_ep:], marker="o", color=MUTED, linestyle=":", label="never run (stopped)")
    ax.scatter([best_ep], [best], s=140, facecolor="none", edgecolor=SERIES[1], linewidth=2, zorder=5, label="best.pt saved")
    ax.axvline(stop_ep, color=SERIES[1], linestyle="--", linewidth=1)
    ax.annotate(f"early stop after epoch {stop_ep}\n(no gain for {patience} epochs)", (stop_ep, val_acc[:stop_ep].min() + 0.02),
                xytext=(stop_ep - 4.6, val_acc[:stop_ep].min() + 0.01), fontsize=8, color=INK_2)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy"); ax.set_title("Early stopping (illustrative history)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot([1, 2], lr2, marker="o", color=BLUE, label="budget 2 epochs (first run)")
    ax.plot([1, 2, 3], lr3, marker="s", color=SERIES[1], label="budget 3 epochs (uninterrupted)")
    ax.scatter([3], [lr3[2]], s=160, facecolor="none", edgecolor=SERIES[2], linewidth=2, zorder=5, label="resumed epoch (rebuilt schedule)")
    ax.annotate(f"{lr3[2]:.1e}", (3, lr3[2]), xytext=(2.55, lr3[2] + 2.5e-4), fontsize=8, color=INK_2)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("epoch"); ax.set_ylabel("learning rate at the start of the epoch")
    ax.set_title("Cosine schedule when the budget grows on resume"); ax.legend(loc="upper right")
    save(fig, NB, "trainer_early_stopping_and_resume")


# --------------------------------------------------------------------------------------
# Section 2.5: confusion matrix of the MLP (seed 0) on the shapes validation set
# --------------------------------------------------------------------------------------
def fig_confusion_matrix_shapes():
    banner("Section 2.5: confusion matrix and hardest classes (MLP seed 0 on the shapes validation set)")
    from sklearn.metrics import confusion_matrix
    L = baseline_ladder_numbers()
    Xtr, ytr, Xva, yva = shapes_data()
    m = L["mlp_models"][0]
    pred = m.predict(Xva.reshape(len(Xva), -1))
    proba = m.predict_proba(Xva.reshape(len(Xva), -1))
    cm = confusion_matrix(yva, pred, labels=[0, 1, 2, 3])
    print("  confusion matrix (rows true, cols predicted), classes", CLASS_NAMES)
    for c, row in zip(CLASS_NAMES, cm):
        print(f"    {c:14s} {row.tolist()}")
    per_class = cm.diagonal() / cm.sum(1)
    order = np.argsort(per_class)
    print("  per-class accuracy:", {c: round(float(a), 3) for c, a in zip(CLASS_NAMES, per_class)})
    print(f"  hardest class: '{CLASS_NAMES[order[0]]}' ({per_class[order[0]]:.3f}); overall {np.trace(cm) / cm.sum():.3f}")
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"  most common confusion: true '{CLASS_NAMES[i]}' predicted as '{CLASS_NAMES[j]}' ({off[i, j]} times)")
    wrong = np.where(pred != yva)[0]
    conf = proba[wrong].max(1)
    top = wrong[np.argsort(-conf)][:3]
    print("  most confident mistakes (true -> predicted, confidence):",
          [(CLASS_NAMES[yva[k]], CLASS_NAMES[pred[k]], round(float(proba[k].max()), 3)) for k in top])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=SEQ_CMAP)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=10, color=SURFACE if cm[a, b] > cm.max() / 2 else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    ax.set_title(f"Confusion matrix, validation (accuracy {np.trace(cm) / cm.sum():.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="count")
    save(fig, NB, "confusion_matrix_shapes")


# --------------------------------------------------------------------------------------
# Section 3.3 (project c): synthetic series, naive and seasonal-naive forecasts, MAE table
# --------------------------------------------------------------------------------------
def make_series(n, seed=SEED):
    r_ = np.random.default_rng(seed)
    t = np.arange(n)
    y = (0.001 * t + 2.0 * np.sin(2 * np.pi * t / 24) + 0.8 * np.sin(2 * np.pi * t / 168)
         + 0.5 * ((t // 500) % 2) + 0.3 * r_.standard_normal(n))
    return y.astype(np.float32)


def make_windows(series, context, horizon):
    n = len(series) - context - horizon + 1
    X = np.stack([series[i:i + context] for i in range(n)])
    Y = np.stack([series[i + context:i + context + horizon] for i in range(n)])
    return X, Y


def fig_seasonal_series_forecast():
    banner("Section 3.3 (project c): synthetic series, naive vs seasonal-naive baselines (FAST_RUN sizes)")
    context, horizon, season = 48, 12, 24
    series = make_series(3000)
    n_split = int(0.8 * len(series))
    mean_c, std_c = float(series[:n_split].mean()), float(series[:n_split].std())
    print(f"  series length 3000, train 0..{n_split - 1}, val {n_split}..2999; train mean {mean_c:.3f}, std {std_c:.3f}")
    sn = (series - mean_c) / std_c
    Xv, Yv = make_windows(sn[n_split:], context, horizon)
    naive = np.repeat(Xv[:, -1:], horizon, 1)
    seasonal = Xv[:, -season:-season + horizon]
    mae_naive = np.abs(naive - Yv).mean() * std_c
    mae_seas = np.abs(seasonal - Yv).mean() * std_c
    mae_mean = np.abs(Xv.mean(1, keepdims=True) - Yv).mean() * std_c
    print(f"  validation windows: {len(Xv)}")
    print(f"  MAE naive (last value):      {mae_naive:.3f}  (ratio to seasonal naive {mae_naive / mae_seas:.2f})")
    print(f"  MAE window mean:             {mae_mean:.3f}  (ratio {mae_mean / mae_seas:.2f})")
    print(f"  MAE seasonal naive (24 ago): {mae_seas:.3f}  (ratio 1.00)")
    noise_floor = 0.3 * math.sqrt(2 / math.pi)
    print(f"  irreducible MAE of the 0.3-std Gaussian noise: {noise_floor:.3f} (ratio {noise_floor / mae_seas:.2f})")
    per_h_naive = (np.abs(naive - Yv).mean(0) * std_c)
    per_h_seas = (np.abs(seasonal - Yv).mean(0) * std_c)
    print(f"  MAE by horizon h=1/6/12, naive: {per_h_naive[0]:.2f}/{per_h_naive[5]:.2f}/{per_h_naive[11]:.2f}; "
          f"seasonal: {per_h_seas[0]:.2f}/{per_h_seas[5]:.2f}/{per_h_seas[11]:.2f}")
    # one window for the picture
    w = 100
    hist = Xv[w] * std_c + mean_c; target = Yv[w] * std_c + mean_c
    f_naive = naive[w] * std_c + mean_c; f_seas = seasonal[w] * std_c + mean_c
    t0 = n_split + w
    print(f"  example window starting at t={t0}: last observed value {hist[-1]:.2f}, first target {target[0]:.2f}, "
          f"seasonal-naive first forecast {f_seas[0]:.2f}, naive first forecast {f_naive[0]:.2f}")
    print(f"  example window MAE: naive {np.abs(f_naive - target).mean():.2f}, seasonal {np.abs(f_seas - target).mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw=dict(width_ratios=[2, 1]))
    ax = axes[0]
    tt = np.arange(t0, t0 + context); th = np.arange(t0 + context, t0 + context + horizon)
    ax.plot(tt, hist, color=INK_2, linewidth=1.5, label="observed (48 h context)")
    ax.plot(th, target, color=BLUE, marker="o", markersize=4, label="actual next 12 h")
    ax.plot(th, f_naive, color=SERIES[1], linestyle="--", label="naive: last value")
    ax.plot(th, f_seas, color=SERIES[2], marker="s", markersize=4, label="seasonal naive: 24 h ago")
    ax.axvline(t0 + context - 0.5, color=AXIS, linewidth=1)
    ax.set_xlabel("time step (hours)"); ax.set_ylabel("value (original units)")
    ax.set_title("One forecast window of the synthetic series"); ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    hs = np.arange(1, horizon + 1)
    ax.plot(hs, per_h_naive, marker="o", color=SERIES[1], label="naive")
    ax.plot(hs, per_h_seas, marker="s", color=SERIES[2], label="seasonal naive")
    ax.axhline(noise_floor, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("noise floor", (12, noise_floor), ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("horizon (hours ahead)"); ax.set_ylabel("MAE on validation windows"); ax.set_ylim(0, None)
    ax.set_title("Error by horizon"); ax.legend()
    save(fig, NB, "seasonal_series_forecast")


# --------------------------------------------------------------------------------------
# Section 3.4 (project d): the DDPM beta schedule and alpha_bar
# --------------------------------------------------------------------------------------
def fig_diffusion_schedule():
    banner("Section 3.4 (project d): linear beta schedule, alpha_bar_T (computed)")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for T, col, mk in [(100, BLUE, "o"), (400, SERIES[1], "s")]:
        betas = np.linspace(1e-4, 20.0 / T, T)
        abar = np.cumprod(1 - betas)
        print(f"  T={T}: beta_1 {betas[0]:.1e}, beta_T {betas[-1]:.3f}, sum(betas) {betas.sum():.3f}, "
              f"alpha_bar at t=T/4 {abar[T // 4 - 1]:.3f}, t=T/2 {abar[T // 2 - 1]:.3f}, t=T {abar[-1]:.2e}; "
              f"signal fraction sqrt(alpha_bar_T) {math.sqrt(abar[-1]):.4f}")
        t = np.arange(1, T + 1) / T
        ax.plot(t, abar, color=col, label=f"T = {T} (FAST_RUN)" if T == 100 else f"T = {T} (full run)")
        ax.plot(t[::T // 8], abar[::T // 8], linestyle="none", marker=mk, color=col, markersize=6)
    ax.annotate("alpha_bar_T = 2e-5 (T=100), 4e-5 (T=400):\nthe final x_T is pure noise", (1.0, 0.02), xytext=(0.52, 0.30),
                fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("t / T (fraction of the forward process)"); ax.set_ylabel("alpha_bar_t (remaining signal variance)")
    ax.set_title("Forward process: betas from 1e-4 to 20/T sum to 10, so alpha_bar_T is about exp(-10)")
    ax.legend()
    save(fig, NB, "diffusion_schedule")


# --------------------------------------------------------------------------------------
# Section 3.5 (project e): bootstrap CIs and the paired vs unpaired difference (exact replication of cell 28)
# --------------------------------------------------------------------------------------
def bootstrap_ci(correct, n_boot=2000, seed=SEED):
    r_ = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    boots = r_.choice(correct, size=(n_boot, len(correct)), replace=True).mean(1)
    return correct.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired_bootstrap_diff(a, b, n_boot=2000, seed=SEED):
    r_ = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boots = r_.choice(diff, size=(n_boot, len(diff)), replace=True).mean(1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fig_paired_vs_unpaired_ci():
    banner("Section 3.5 (project e): bootstrap intervals on the notebook's synthetic correctness arrays (seed 42)")
    r_ = np.random.default_rng(SEED)
    n = 500
    correct_full = r_.random(n) < 0.89
    flip = np.where(correct_full, r_.random(n) < 0.05, r_.random(n) < 0.10)
    correct_lora = correct_full ^ flip
    both_right = (correct_full & correct_lora).sum(); both_wrong = (~correct_full & ~correct_lora).sum()
    only_full = (correct_full & ~correct_lora).sum(); only_lora = (~correct_full & correct_lora).sum()
    print(f"  items: both right {both_right}, both wrong {both_wrong}, only full right {only_full}, only LoRA right {only_lora}")
    rows = []
    for name, c in [("full fine-tuning", correct_full), ("LoRA", correct_lora)]:
        m, lo, hi = bootstrap_ci(c)
        rows.append((name, m, lo, hi))
        print(f"  {name:18s} accuracy {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  half-width {(hi - lo) / 2:.3f}")
    d, lo_p, hi_p = paired_bootstrap_diff(correct_full, correct_lora)
    print(f"  paired difference (full - LoRA): {d:+.3f}  95% CI [{lo_p:+.3f}, {hi_p:+.3f}]  "
          + ("-> not significant" if lo_p < 0 < hi_p else "-> significant"))
    se_ind = math.sqrt(correct_full.var() / n + correct_lora.var() / n)
    lo_u, hi_u = d - 1.96 * se_ind, d + 1.96 * se_ind
    print(f"  unpaired normal approximation: {d:+.3f}  95% CI [{lo_u:+.3f}, {hi_u:+.3f}]; "
          f"paired interval is {(hi_p - lo_p) / (hi_u - lo_u):.2f}x as wide")
    print(f"  overlap of the two marginal intervals: {rows[1][3]:.3f} > {rows[0][2]:.3f} -> they overlap, yet the paired test is clear")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["full fine-tuning\naccuracy", "LoRA\naccuracy", "difference,\nunpaired normal", "difference,\npaired bootstrap"]
    centres = [rows[0][1], rows[1][1], d, d]
    los = [rows[0][2], rows[1][2], lo_u, lo_p]; his = [rows[0][3], rows[1][3], hi_u, hi_p]
    cols = [BLUE, BLUE, SERIES[1], SERIES[2]]
    # accuracies on the left axis region, differences shifted so both fit on one numeric axis: use offsets
    for k, (lab, c, lo, hi, col) in enumerate(zip(labels, centres, los, his, cols)):
        y = 3 - k
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.5)
        ax.plot([c], [y], marker="o", color=col, markersize=8)
        ax.annotate(f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]" if k >= 2 else f"{c:.3f} [{lo:.3f}, {hi:.3f}]",
                    (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(labels)
    ax.set_xlim(-0.05, 1.08); ax.set_xlabel("accuracy, or accuracy difference (500 test items)")
    ax.set_title("Marginal intervals overlap; the paired interval of the difference does not touch 0")
    save(fig, NB, "paired_vs_unpaired_ci")


# --------------------------------------------------------------------------------------
# Section 3.6 (project f): tabular Q-learning on the built-in FrozenLake clone (exact replication of cell 30)
# --------------------------------------------------------------------------------------
class MiniFrozenLake:
    MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]

    def __init__(self):
        self.nS, self.nA, self.s = 16, 4, 0

    def reset(self, seed=None):
        self.s = 0
        return self.s, {}

    def step(self, a):
        r, c = divmod(self.s, 4)
        r, c = [(r, max(c - 1, 0)), (min(r + 1, 3), c), (r, min(c + 1, 3)), (max(r - 1, 0), c)][a]
        self.s = 4 * r + c
        tile = self.MAP[r][c]
        return self.s, float(tile == "G"), tile in "HG", False, {}


def q_learning(env, episodes, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, max_steps=100, seed=SEED):
    r_ = np.random.default_rng(seed)
    Q, returns, eps_log = np.zeros((env.nS, env.nA)), [], []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.8 * episodes)))
        eps_log.append(eps)
        s, _ = env.reset(seed=int(r_.integers(2**31 - 1)))
        total = 0.0
        for _ in range(max_steps):
            a = int(r_.integers(env.nA)) if r_.random() < eps else int(np.argmax(Q[s] + 1e-9 * r_.random(env.nA)))
            s2, r, terminated, truncated, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * (0.0 if terminated else Q[s2].max()) - Q[s, a])
            s, total = s2, total + r
            if terminated or truncated:
                break
        returns.append(total)
    return Q, np.array(returns), np.array(eps_log)


def evaluate_policy(env, policy, episodes=100, max_steps=100, seed=SEED + 1):
    r_ = np.random.default_rng(seed)
    scores = []
    for _ in range(episodes):
        s, _ = env.reset(seed=int(r_.integers(2**31 - 1)))
        total = 0.0
        for _ in range(max_steps):
            s, r, terminated, truncated, _ = env.step(policy(s, r_))
            total += r
            if terminated or truncated:
                break
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


def fig_q_learning_frozenlake():
    banner("Section 3.6 (project f): Q-learning on MiniFrozenLake, 300 episodes (FAST_RUN), exact replication")
    env = MiniFrozenLake()
    Q, returns, eps = q_learning(env, 300)
    rand = evaluate_policy(env, lambda s, r_: int(r_.integers(4)))
    greedy = evaluate_policy(env, lambda s, r_: int(np.argmax(Q[s])))
    print(f"  random policy:      success rate {rand[0]:.2f} +- {rand[1]:.2f} over 100 episodes")
    print(f"  Q-learning greedy:  success rate {greedy[0]:.2f} +- {greedy[1]:.2f} over 100 episodes")
    window = 20
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    first_half = np.argmax(smooth >= 0.5) + window - 1 if (smooth >= 0.5).any() else None
    first_09 = np.argmax(smooth >= 0.9) + window - 1 if (smooth >= 0.9).any() else None
    print(f"  training returns: first 20 episodes mean {returns[:20].mean():.2f}, last 20 mean {returns[-20:].mean():.2f}; "
          f"moving average first reaches 0.5 at episode {first_half}, 0.9 at episode {first_09}")
    print(f"  epsilon: {eps[0]:.2f} at episode 0, {eps[120]:.2f} at episode 120, floor {eps[-1]:.2f} from episode {int(np.argmax(eps <= 0.05))}")
    arrows = ["<", "v", ">", "^"]
    greedy_grid = np.argmax(Q, 1).reshape(4, 4)
    V = Q.max(1).reshape(4, 4)
    print(f"  V(start) = max_a Q[0, a] = {V[0, 0]:.3f}  (optimal with gamma 0.99 and 6 steps: 0.99^5 = {0.99 ** 5:.3f})")
    print("  greedy policy (rows top to bottom):", [" ".join(arrows[a] for a in row) for row in greedy_grid])
    Q2, returns2, _ = q_learning(env, 2000)
    greedy2 = evaluate_policy(env, lambda s, r_: int(np.argmax(Q2[s])))
    print(f"  full budget (2000 episodes): greedy success {greedy2[0]:.2f} +- {greedy2[1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    ax.plot(np.arange(len(returns)), returns, color=GRID, linewidth=0.8, label="raw episode return (0 or 1)")
    ax.plot(np.arange(window - 1, len(returns)), smooth, color=BLUE, label=f"moving average ({window} episodes)")
    ax.plot(np.arange(len(eps)), eps, color=SERIES[1], linestyle="--", linewidth=1.2, label="exploration rate epsilon")
    ax.set_xlabel("episode"); ax.set_ylabel("return / epsilon"); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Tabular Q-learning: greedy success {greedy[0]:.2f} after 300 episodes"); ax.legend(loc="center right", fontsize=8)
    ax = axes[1]
    im = ax.imshow(V, cmap=SEQ_CMAP, vmin=0, vmax=1)
    for r in range(4):
        for c in range(4):
            tile = MiniFrozenLake.MAP[r][c]
            txt = {"H": "hole", "G": "goal", "S": "S " + arrows[greedy_grid[r, c]]}.get(tile, arrows[greedy_grid[r, c]])
            ax.text(c, r, txt, ha="center", va="center", fontsize=10 if tile in "HG" else 13,
                    color=SURFACE if V[r, c] > 0.5 else INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Learned state value and greedy action")
    fig.colorbar(im, ax=ax, fraction=0.046, label="V(s) = max_a Q(s, a)")
    save(fig, NB, "q_learning_frozenlake")


# --------------------------------------------------------------------------------------
# Section 4.2: the seed lottery and a paired comparison over seeds (computed with the sklearn MLP)
# --------------------------------------------------------------------------------------
def fig_seed_lottery_paired():
    banner("Section 4.2: seed variance and a paired comparison (MLP 64 vs MLP 16 hidden units, 5 shared seeds)")
    from sklearn.neural_network import MLPClassifier
    from scipy import stats
    Xtr, ytr, Xva, yva = shapes_data()
    Ftr, Fva = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    L = baseline_ladder_numbers()
    seeds = [0, 1, 2, 3, 4]
    acc_a = L["mlp"]                                         # config A: 64 hidden units (already trained)
    acc_b = np.array([MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=60, random_state=s).fit(Ftr, ytr).score(Fva, yva)
                      for s in seeds])
    print(f"  config A (64 hidden): per seed {np.round(acc_a, 3).tolist()}  mean {acc_a.mean():.4f} +- {acc_a.std(ddof=1):.4f}")
    print(f"  config B (16 hidden): per seed {np.round(acc_b, 3).tolist()}  mean {acc_b.mean():.4f} +- {acc_b.std(ddof=1):.4f}")
    print(f"  spread (max - min): A {acc_a.max() - acc_a.min():.3f}, B {acc_b.max() - acc_b.min():.3f}")
    diff = acc_a - acc_b
    n = len(diff)
    se = diff.std(ddof=1) / math.sqrt(n)
    t_stat = diff.mean() / se
    d_cohen = diff.mean() / diff.std(ddof=1)
    p_paired = stats.ttest_rel(acc_a, acc_b).pvalue
    p_indep = stats.ttest_ind(acc_a, acc_b).pvalue
    r_ = np.random.default_rng(SEED)
    boots = r_.choice(diff, size=(5000, n), replace=True).mean(1)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"  per-seed differences A - B: {np.round(diff, 3).tolist()}")
    print(f"  mean difference {diff.mean():+.4f}, s.e. {se:.4f}, t = {t_stat:.2f}, Cohen's d = {d_cohen:.2f}, "
          f"paired t-test p = {p_paired:.4f}; unpaired t-test p = {p_indep:.4f}")
    print(f"  bootstrap 95% CI of the mean difference (5000 resamples of the 5 differences): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  correlation of A and B across seeds: {np.corrcoef(acc_a, acc_b)[0, 1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for i, s in enumerate(seeds):
        ax.plot([0, 1], [acc_b[i], acc_a[i]], color=GRID, linewidth=1, zorder=1)
    ax.scatter([0] * n, acc_b, color=SERIES[1], s=50, zorder=3, label="config B: 16 hidden units")
    ax.scatter([1] * n, acc_a, color=BLUE, s=50, zorder=3, label="config A: 64 hidden units")
    for i, s in enumerate(seeds):
        ax.annotate(f"seed {s}", (1, acc_a[i]), xytext=(8, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
    ax.errorbar([-0.15, 1.15], [acc_b.mean(), acc_a.mean()], yerr=[acc_b.std(ddof=1), acc_a.std(ddof=1)],
                fmt="_", color=INK_2, capsize=5, markersize=14, label="mean +- 1 std")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["B (16)", "A (64)"]); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("validation accuracy"); ax.set_title("The seed lottery: same seeds, two configs")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    ax.bar([f"s{s}" for s in seeds], diff, width=0.6, color=SERIES[2], edgecolor=SURFACE, linewidth=1.5, label="difference A - B per seed")
    ax.axhline(diff.mean(), color=INK, linestyle="--", linewidth=1, label=f"mean {diff.mean():+.3f}")
    ax.axhspan(lo, hi, color=SERIES[2], alpha=0.15, label=f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("accuracy difference"); ax.set_title(f"Paired by seed: t = {t_stat:.1f}, p = {p_paired:.3f}")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, NB, "seed_lottery_paired")


# --------------------------------------------------------------------------------------
# Section 4.4: raw per-step loss vs moving average with a seed band (synthetic curves, labelled)
# --------------------------------------------------------------------------------------
def synthetic_step_losses(n_seeds=5, steps=240, k=10):
    r_ = np.random.default_rng(11)
    t = np.arange(steps)
    curves = []
    for s in range(n_seeds):
        base = 0.25 + 1.75 * np.exp(-t / (55 + 6 * s))
        noise = r_.normal(0, 0.12, steps) * (0.4 + base / 2)
        curves.append(base + noise)
    curves = np.array(curves)
    smooth = np.array([np.convolve(c, np.ones(k) / k, mode="valid") for c in curves])
    return t, curves, smooth


def fig_moving_average_vs_raw():
    banner("Section 4.4: raw per-step loss vs moving average with seed band (synthetic curves, illustrative)")
    k = 10
    t, curves, smooth = synthetic_step_losses(k=k)
    x = np.arange(smooth.shape[1]) + k - 1
    mean_c, std_c = smooth.mean(0), smooth.std(0)
    raw = curves[0]
    print(f"  5 synthetic seeds, {curves.shape[1]} steps, window {k}")
    print(f"  raw seed 0: step 100 loss {raw[100]:.3f}, step 101 loss {raw[101]:.3f} (jump {raw[101] - raw[100]:+.3f}); "
          f"std of step-to-step changes {np.diff(raw).std():.3f}")
    print(f"  moving average over seeds at step 100: mean {mean_c[100 - k + 1]:.3f}, std across seeds {std_c[100 - k + 1]:.3f}")
    print(f"  final 20 steps: raw seed 0 mean {raw[-20:].mean():.3f}; mean curve {mean_c[-20:].mean():.3f} +- {std_c[-20:].mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    axes[0].plot(t, raw, color=BLUE, linewidth=0.8)
    axes[0].set_title("before: one seed, raw per-step loss, no labels")
    axes[1].plot(x, mean_c, color=BLUE, label="mean of 5 seeds")
    axes[1].fill_between(x, mean_c - std_c, mean_c + std_c, color=BLUE, alpha=0.25, linewidth=0, label="+- 1 std across seeds")
    axes[1].set_xlabel("optimizer step"); axes[1].set_ylabel(f"training loss (moving average, {k} steps)")
    axes[1].set_title("after: smoothed, averaged over seeds, labelled")
    axes[1].legend()
    axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("training loss")
    save(fig, NB, "moving_average_vs_raw")


# --------------------------------------------------------------------------------------
# Section 4.4 / 4.5: small multiples instead of a dual axis (synthetic history, labelled)
# --------------------------------------------------------------------------------------
def fig_small_multiples():
    banner("Section 4.4: two small multiples (loss, accuracy) instead of one dual-axis chart (synthetic, illustrative)")
    epochs = np.arange(1, 13)
    r_ = np.random.default_rng(5)
    runs = {}
    for name, a0, tau, acc_max, col in [("AdamW, lr 2e-3", 2.0, 2.5, 0.905, BLUE), ("SGD+momentum, lr 5e-2", 2.1, 4.0, 0.885, SERIES[1])]:
        loss = 0.3 + (a0 - 0.3) * np.exp(-epochs / tau) + r_.normal(0, 0.01, len(epochs))
        acc = acc_max - 0.35 * np.exp(-epochs / tau) + r_.normal(0, 0.004, len(epochs))
        runs[name] = (loss, acc, col)
        print(f"  {name:22s} epoch 1 loss {loss[0]:.3f} acc {acc[0]:.3f} | epoch 12 loss {loss[-1]:.3f} acc {acc[-1]:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, (loss, acc, col) in runs.items():
        axes[0].plot(epochs, loss, marker="o", markersize=4, color=col, label=name)
        axes[1].plot(epochs, acc, marker="o", markersize=4, color=col, label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].set_title("Training loss per epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy"); axes[1].set_title("Validation accuracy per epoch")
    axes[1].set_ylim(0.6, 0.95)
    axes[0].legend(); axes[1].legend(loc="lower right")
    fig.suptitle("Two small multiples with a shared x-axis replace one chart with two y-axes", fontsize=11, fontweight="bold")
    save(fig, NB, "small_multiples_loss_acc")


# --------------------------------------------------------------------------------------
# Section 5: the reading list by area (hand-counted from the notebook's Section 5.1)
# --------------------------------------------------------------------------------------
def fig_reading_list_by_area():
    banner("Section 5.1: papers in the reading list by area (counted from the notebook text)")
    areas = {"Vision": ["AlexNet 2012", "ResNet 2016", "ViT 2021", "ConvNeXt 2022"],
             "Language": ["Transformer 2017", "BERT 2019", "GPT-3 2020", "LoRA 2021", "InstructGPT 2022", "DPO 2023"],
             "Generative": ["VAE 2014", "GAN 2014", "DDPM 2020", "Latent diffusion 2022", "Flow matching 2023"],
             "Reinforcement\nlearning": ["DQN 2015", "PPO 2017"],
             "Scaling and\nefficiency": ["Scaling laws 2020", "Chinchilla 2022", "FlashAttention 2022", "QLoRA 2023"],
             "Trust": ["Integrated Gradients 2017", "Grad-CAM 2017", "Adversarial training 2018", "Calibration 2017",
                       "Deep ensembles 2017", "Model cards 2019", "Datasheets 2021"]}
    total = sum(len(v) for v in areas.values())
    years = [int(p.split()[-1]) for v in areas.values() for p in v]
    for k, v in areas.items():
        print(f"  {k.replace(chr(10), ' '):24s} {len(v)} papers: {', '.join(v)}")
    print(f"  total {total} papers, years {min(years)}-{max(years)}, median year {int(np.median(years))}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = list(areas.keys()); counts = [len(v) for v in areas.values()]
    bar(ax, names, counts, color=BLUE, label_fmt="{:d}")
    ax.set_ylabel("papers in the reading list"); ax.set_ylim(0, 8.5)
    ax.set_title(f"Reading list of Section 5.1: {total} papers by area")
    save(fig, NB, "reading_list_by_area")


def main():
    fig_data_audit()
    fig_baseline_ladder()
    fig_trainer_history_and_resume()
    fig_confusion_matrix_shapes()
    fig_seasonal_series_forecast()
    fig_diffusion_schedule()
    fig_paired_vs_unpaired_ci()
    fig_q_learning_frozenlake()
    fig_seed_lottery_paired()
    fig_moving_average_vs_raw()
    fig_small_multiples()
    fig_reading_list_by_area()


if __name__ == "__main__":
    main()
