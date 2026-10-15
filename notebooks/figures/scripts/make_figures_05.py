"""Figures for the companion guide of notebook 05 (PyTorch workflow and best practices).

Everything here is numpy + matplotlib only. Each fig_* function builds its data, prints the
numbers quoted in the guide, and saves one PNG under notebooks/figures/05/.
Where a chart is an illustrative model rather than a measurement, its title says so.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

NB = "05"
SEED = 42   # the notebook's seed, reused where we replicate its synthetic data exactly


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# Section 1: nn.Module anatomy -> parameter counts of SmallCNN, layer by layer
# --------------------------------------------------------------------------------------
def fig_smallcnn_parameter_counts():
    banner("Section 1: SmallCNN parameter count, layer by layer")
    c1, c2, n_classes = 32, 64, 10
    layers = [
        ("conv1 (1->32, 3x3)", 1 * c1 * 9 + c1),
        ("bn1", 2 * c1),
        ("conv2 (32->64, 3x3)", c1 * c2 * 9 + c2),
        ("bn2", 2 * c2),
        ("fc1 (3136->128)", c2 * 7 * 7 * 128 + 128),
        ("fc2 (128->10)", 128 * n_classes + n_classes),
    ]
    total = sum(v for _, v in layers)
    for name, v in layers:
        print(f"  {name:22s} {v:>9,d} parameters  ({100 * v / total:5.1f} %)")
    print(f"  {'total':22s} {total:>9,d}")
    mlp = 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10
    print(f"  MLP(512, 256) for comparison: {mlp:,d} parameters")
    # the demo Sequential(Standardize, Linear(2,3), BatchNorm1d(3)) of Section 1.2
    sd = [("0.mean", (2,), "buffer"), ("0.std", (2,), "buffer"), ("1.weight", (3, 2), "parameter"),
          ("1.bias", (3,), "parameter"), ("2.weight", (3,), "parameter"), ("2.bias", (3,), "parameter"),
          ("2.running_mean", (3,), "buffer"), ("2.running_var", (3,), "buffer"), ("2.num_batches_tracked", (), "buffer")]
    n_par = sum(int(np.prod(s)) for _, s, k in sd if k == "parameter")
    n_buf = sum(int(np.prod(s)) for _, s, k in sd if k == "buffer")
    print(f"  demo Sequential: {len(sd)} state_dict entries, {n_par} parameter values, {n_buf} buffer values")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [n for n, _ in layers]
    vals = [v for _, v in layers]
    bar(ax, names, vals, color=BLUE, label_fmt="{:,}")
    ax.set_yscale("log")
    ax.set_ylim(10, 2e6)
    ax.set_ylabel("parameters (log scale)")
    ax.set_title("SmallCNN: 95 % of the parameters sit in the first linear layer")
    ax.tick_params(axis="x", labelrotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    save(fig, NB, "smallcnn_parameter_counts")


# --------------------------------------------------------------------------------------
# Section 2: the synthetic table of the notebook (replicated exactly), batching, imbalance
# --------------------------------------------------------------------------------------
def make_table():
    g = np.random.default_rng(SEED)
    features = g.normal(size=(500, 5)) * np.array([1.0, 5.0, 0.1, 2.0, 1.0]) + np.array([0.0, 10.0, 0.0, -3.0, 1.0])
    labels = (features[:, 0] + 0.5 * features[:, 3] + g.normal(scale=0.5, size=500) > 0).astype(int)
    return features, labels


def fig_imbalance_weighted_sampling():
    banner("Section 2: synthetic table, batches, class imbalance and weighted sampling")
    features, labels = make_table()
    print("  first 6 rows of the notebook's synthetic table (f0..f4, label):")
    for i in range(6):
        print("   ", i, np.round(features[i], 2).tolist(), int(labels[i]), " score f0+0.5*f3 =", round(features[i, 0] + 0.5 * features[i, 3], 2))
    print(f"  rows {len(labels)}, positives {labels.sum()} ({labels.mean():.3f})")
    print(f"  batches of 64: {int(np.ceil(500 / 64))} per epoch, last batch {500 % 64} rows; with drop_last: {500 // 64}")

    g_imb = np.random.default_rng(SEED + 1)
    keep = [i for i in range(500) if labels[i] == 0 or g_imb.random() < 0.05]
    y = labels[keep]
    counts = np.bincount(y, minlength=2)
    w_class = 1.0 / counts
    print(f"  imbalanced subset: {counts.tolist()} examples per class, {len(keep)} rows, {int(np.ceil(len(keep) / 32))} batches of 32")
    print(f"  per-example weights: class 0 -> 1/{counts[0]} = {w_class[0]:.6f}, class 1 -> 1/{counts[1]} = {w_class[1]:.6f}")
    mass = counts * w_class
    print(f"  total weight per class: {mass.tolist()} -> probability of drawing class 1 = {mass[1] / mass.sum():.2f}")
    print(f"  expected positives in a batch of 32: shuffling {32 * counts[1] / counts.sum():.2f}, weighted sampler {32 * 0.5:.0f}")

    # simulate one epoch of each loader (numpy stand-in for RandomSampler / WeightedRandomSampler)
    sim = np.random.default_rng(SEED)
    n, bs = len(keep), 32
    order = sim.permutation(n)
    shuffled = [y[order[i:i + bs]].mean() for i in range(0, n, bs)]
    p = w_class[y] / w_class[y].sum()
    draws = sim.choice(n, size=n, replace=True, p=p)
    weighted = [y[draws[i:i + bs]].mean() for i in range(0, n, bs)]
    print(f"  simulated fraction of class 1 per batch: shuffle mean {np.mean(shuffled):.2f}, min {min(shuffled):.2f}, "
          f"batches with none: {sum(f == 0 for f in shuffled)} of {len(shuffled)}")
    print(f"                                           sampler mean {np.mean(weighted):.2f}, min {min(weighted):.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(shuffled))
    ax.bar(x - 0.2, shuffled, width=0.4, color=SERIES[0], edgecolor=SURFACE, label="shuffle=True")
    ax.bar(x + 0.2, weighted, width=0.4, color=SERIES[1], edgecolor=SURFACE, label="WeightedRandomSampler")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", label="balanced (0.5)")
    ax.set_xlabel("batch index (one simulated epoch, 427 rows, batch size 32)")
    ax.set_ylabel("fraction of class 1 in the batch")
    ax.set_title("Class imbalance (421 vs 6): plain shuffling vs inverse-frequency sampling")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    save(fig, NB, "imbalance_weighted_sampling")


def fig_dataloader_workers_model():
    banner("Section 2.6: data-loading time vs number of workers (illustrative model)")
    n_batches, t_load, t_compute = 400, 0.040, 0.010   # seconds per batch
    workers = np.arange(0, 9)
    per_batch = np.where(workers == 0, t_load + t_compute, np.maximum(t_compute, t_load / np.maximum(workers, 1)))
    epoch = n_batches * per_batch
    floor = n_batches * t_compute
    print(f"  model: {n_batches} batches/epoch, {t_load * 1000:.0f} ms to load+augment a batch in Python, {t_compute * 1000:.0f} ms to compute it")
    for w, t in zip(workers, epoch):
        print(f"   workers {w}: {t:5.1f} s per epoch")
    print(f"  compute-bound floor: {floor:.1f} s; the loader stops being the bottleneck at {int(np.ceil(t_load / t_compute))} workers")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(workers, epoch, "o-", color=SERIES[0], label="epoch time (model)")
    ax.axhline(floor, color=SERIES[1], lw=1.2, ls="--", label="compute-bound floor")
    ax.annotate(f"num_workers=0: {epoch[0]:.0f} s", (0, epoch[0]), xytext=(10, -4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.annotate(f"4 workers: {epoch[4]:.0f} s", (4, epoch[4]), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("num_workers")
    ax.set_ylabel("time per epoch (s)")
    ax.set_ylim(0, 22)
    ax.set_title("Data loading vs workers (illustrative model, not a measurement)")
    ax.legend()
    save(fig, NB, "dataloader_workers_model")


# --------------------------------------------------------------------------------------
# Section 3: training history with early stopping; exact epoch mean; 4-class confusion matrix
# --------------------------------------------------------------------------------------
def fig_training_history_early_stopping():
    banner("Section 3: simulated training history, early stopping, exact epoch mean")
    sizes = np.array([64, 64, 52]); losses = np.array([0.70, 0.60, 0.50])
    naive = losses.mean(); exact = (sizes * losses).sum() / sizes.sum()
    print(f"  three batches of sizes {sizes.tolist()} with mean losses {losses.tolist()}:")
    print(f"   naive mean of batch means = {naive:.4f}; sum(loss*size)/sum(size) = {(sizes * losses).sum():.1f}/{sizes.sum()} = {exact:.4f}")

    epochs = np.arange(1, 17)
    g = np.random.default_rng(3)
    train = 1.6 * np.exp(-0.28 * epochs) + 0.12 + g.normal(0, 0.01, len(epochs))
    val = 1.5 * np.exp(-0.30 * epochs) + 0.30 + 0.0035 * (epochs - 9) ** 2 + g.normal(0, 0.008, len(epochs))
    patience = 3
    best, best_epoch, bad, stop_epoch = np.inf, None, 0, None
    log = []
    for e, v in zip(epochs, val):
        improved = v < best - 1e-6
        if improved:
            best, best_epoch, bad = v, e, 0
        else:
            bad += 1
        log.append((int(e), round(float(v), 4), "*" if improved else "", bad))
        if bad >= patience:
            stop_epoch = int(e)
            break
    print("  epoch, val loss, improved?, bad_epochs:")
    for row in log:
        print("   ", row)
    print(f"  best epoch {best_epoch} (val loss {best:.4f}); early stop after epoch {stop_epoch} with patience {patience}")
    shown = epochs <= stop_epoch

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(epochs[shown], train[shown], "o-", color=SERIES[0], label="train loss")
    ax.plot(epochs[shown], val[shown], "o-", color=SERIES[1], label="validation loss")
    ax.plot(epochs[~shown], val[~shown], "o:", color=SERIES[1], alpha=0.35, label="validation (if not stopped)")
    ax.axvline(best_epoch, color=MUTED, lw=0.8, ls="--")
    ax.annotate(f"best val loss {best:.3f}: checkpoint saved (epoch {best_epoch})", (best_epoch, best), xytext=(-20, 95),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"patience {patience} exhausted:\nstop after epoch {stop_epoch}", (stop_epoch, val[stop_epoch - 1]), xytext=(-6, 40),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("A fit() run: best-model checkpoint and early stopping (simulated curves)")
    ax.legend()
    save(fig, NB, "training_history_early_stopping")


def fig_confusion_matrix_4class():
    banner("Section 3.3: a 4-class confusion matrix and the metrics derived from it")
    classes = ["cat", "dog", "fox", "owl"]
    cm = np.array([[8, 1, 1, 0],
                   [2, 7, 0, 1],
                   [0, 1, 9, 0],
                   [0, 0, 2, 8]])
    total = cm.sum(); correct = np.trace(cm)
    per_class = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    print("  rows = true class, columns = predicted class:")
    for c, row in zip(classes, cm):
        print(f"   {c:4s} {row.tolist()}  row sum {row.sum()}")
    print(f"  overall accuracy = trace/total = {correct}/{total} = {correct / total:.3f}")
    for c, r, p, rs, cs, d in zip(classes, per_class, precision, cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)):
        print(f"   {c:4s} per-class accuracy (recall) = {d}/{rs} = {r:.3f} | precision = {d}/{cs} = {p:.3f}")
    i, j = divmod(int(np.argmax(cm - np.diag(np.diag(cm)))), 4)
    print(f"  largest confusion: true {classes[i]} predicted as {classes[j]} ({cm[i, j]} examples)")

    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=cm.max())
    ax.set_xticks(range(4)); ax.set_xticklabels(classes); ax.set_yticks(range(4)); ax.set_yticklabels(classes)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=10,
                    color="white" if cm[a, b] > cm.max() / 2 else INK)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title(f"Confusion matrix of 40 predictions (accuracy {correct / total:.2f})")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("count")
    save(fig, NB, "confusion_matrix_4class")


# --------------------------------------------------------------------------------------
# Section 4: what TensorBoard shows: raw and smoothed step loss, one-cycle learning rate
# --------------------------------------------------------------------------------------
def one_cycle_lr(step, total_steps, max_lr, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    """Exact OneCycleLR schedule (cosine annealing, two phases), as a function of the step index."""
    initial = max_lr / div_factor
    minimum = initial / final_div_factor
    end1 = pct_start * total_steps - 1
    end2 = total_steps - 1

    def cos_anneal(start, end, pct):
        return end + (start - end) / 2.0 * (1 + np.cos(np.pi * pct))
    step = np.asarray(step, dtype=float)
    lr = np.where(step <= end1, cos_anneal(initial, max_lr, step / end1),
                  cos_anneal(max_lr, minimum, (step - end1) / (end2 - end1)))
    return lr


def fig_tensorboard_step_loss_and_lr():
    banner("Section 4: step loss with TensorBoard-style smoothing, and the one-cycle learning rate")
    raw = np.array([2.30, 2.10, 2.25, 1.90, 1.95, 1.70])
    w = 0.6
    smoothed, last = [], 0.0
    for k, v in enumerate(raw):
        last = last * w + (1 - w) * v
        smoothed.append(last / (1 - w ** (k + 1)))      # debiased EMA, as in TensorBoard's smoothing slider
    print(f"  raw step losses {raw.tolist()}; TensorBoard smoothing weight {w}")
    print(f"  smoothed (debiased EMA): {[round(float(s), 3) for s in smoothed]}")

    steps_per_epoch, epochs, max_lr = 47, 3, 2e-3
    total = steps_per_epoch * epochs
    steps = np.arange(total)
    lr = one_cycle_lr(steps, total, max_lr)
    peak = int(np.argmax(lr))
    print(f"  one-cycle schedule: {steps_per_epoch} steps/epoch x {epochs} epochs = {total} steps, max_lr {max_lr:g}")
    print(f"   initial lr = max_lr/25 = {lr[0]:.2e}; peak lr {lr[peak]:.2e} at step {peak} (30 % of the run); "
          f"final lr = initial/1e4 = {lr[-1]:.2e}; lr at step 100 = {lr[100]:.2e}")

    g = np.random.default_rng(5)
    step_loss = 2.3 * np.exp(-steps / 35) + 0.55 + g.normal(0, 0.08, total)
    sm, last = [], 0.0
    for k, v in enumerate(step_loss):
        last = last * w + (1 - w) * v
        sm.append(last / (1 - w ** (k + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(steps, step_loss, color=SERIES[0], lw=0.9, alpha=0.5, label="train/step_loss (raw)")
    ax1.plot(steps, sm, color=SERIES[1], lw=2.0, label=f"smoothed (weight {w})")
    for e in range(1, epochs):
        ax1.axvline(e * steps_per_epoch, color=MUTED, lw=0.7, ls=":")
    ax1.set_xlabel("optimizer step"); ax1.set_ylabel("loss"); ax1.set_title("Scalars tab: per-step loss (simulated)")
    ax1.legend()
    ax2.plot(steps, lr, color=SERIES[0])
    ax2.annotate(f"peak {lr[peak]:.1e} at step {peak}", (peak, lr[peak]), xytext=(12, -2), textcoords="offset points", fontsize=8, color=INK_2)
    ax2.annotate(f"start {lr[0]:.1e}", (0, lr[0]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_2)
    ax2.set_xlabel("optimizer step"); ax2.set_ylabel("learning rate"); ax2.set_title("train/lr: OneCycleLR over 141 steps (exact)")
    save(fig, NB, "tensorboard_step_loss_and_lr")


# --------------------------------------------------------------------------------------
# Section 5: a complete checkpoint reproduces the uninterrupted run (momentum SGD on a quadratic)
# --------------------------------------------------------------------------------------
def fig_resume_reproduces_run():
    banner("Section 5: checkpoint and resume on a 2-D quadratic with momentum SGD")
    A = np.array([[3.0, 0.4], [0.4, 0.6]])          # loss = 0.5 w^T A w
    lr0, mom, total = 0.15, 0.9, 40
    ckpt_step = 20

    def lr_at(step):                                   # cosine schedule, needs the step counter
        return lr0 * 0.5 * (1 + np.cos(np.pi * step / total))

    def run(w, v, start, end, lr_fn):
        hist = []
        for t in range(start, end):
            g = A @ w
            v = mom * v + g
            w = w - lr_fn(t) * v
            hist.append(0.5 * w @ A @ w)
        return w, v, np.array(hist)

    w0 = np.array([2.0, -1.5])
    # run A: uninterrupted
    _, _, loss_a = run(w0, np.zeros(2), 0, total, lr_at)
    # run B: stop at the checkpoint, save (weights, momentum buffer, step), resume
    w_b, v_b, loss_b1 = run(w0, np.zeros(2), 0, ckpt_step, lr_at)
    checkpoint = {"weights": w_b.copy(), "momentum": v_b.copy(), "step": ckpt_step}
    _, _, loss_b2 = run(checkpoint["weights"], checkpoint["momentum"], checkpoint["step"], total, lr_at)
    loss_b = np.concatenate([loss_b1, loss_b2])
    # run C: resume from the weights only (fresh momentum, step counter reset)
    _, _, loss_c2 = run(checkpoint["weights"], np.zeros(2), 0, total - ckpt_step, lr_at)
    loss_c = np.concatenate([loss_b1, loss_c2])
    print(f"  checkpoint after step {ckpt_step}: weights {np.round(checkpoint['weights'], 4).tolist()}, "
          f"momentum buffer {np.round(checkpoint['momentum'], 4).tolist()}, step {checkpoint['step']}")
    print(f"  loss at step 20: A {loss_a[19]:.5f} | at step 21: A {loss_a[20]:.5f}, full resume {loss_b[20]:.5f}, weights-only resume {loss_c[20]:.5f}")
    print(f"  final loss (step 40): A {loss_a[-1]:.6f}, full resume {loss_b[-1]:.6f}, weights-only resume {loss_c[-1]:.6f}")
    print(f"  max |A - full resume| = {np.abs(loss_a - loss_b).max():.1e}; max |A - weights-only| = {np.abs(loss_a - loss_c).max():.3f}")

    steps = np.arange(1, total + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, loss_a, color=SERIES[0], label="run A: uninterrupted")
    ax.plot(steps[ckpt_step:], loss_b[ckpt_step:], "o", ms=6, mfc="none", color=SERIES[1], label="run B: full checkpoint, resumed")
    ax.plot(steps[ckpt_step:], loss_c[ckpt_step:], "--", color=SERIES[2], label="run C: weights only, fresh momentum")
    ax.axvline(ckpt_step, color=MUTED, lw=0.8, ls="--")
    ax.annotate(f"checkpoint after step {ckpt_step}", (ckpt_step, loss_a[ckpt_step - 1]), xytext=(-6, 40),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss (log scale)")
    ax.set_title("Resuming reproduces the run only if the optimizer state comes along")
    ax.legend()
    save(fig, NB, "resume_reproduces_run")


# --------------------------------------------------------------------------------------
# Section 6: the seed-to-seed noise floor (simulated accuracies)
# --------------------------------------------------------------------------------------
def fig_seed_noise_floor():
    banner("Section 6: seed-to-seed spread as the noise floor (simulated accuracies)")
    g = np.random.default_rng(11)
    seeds = [0, 1, 2, 3, 4]
    acc_a = np.round(g.normal(0.885, 0.004, 5), 4)
    acc_b = np.round(g.normal(0.889, 0.004, 5), 4)
    for name, a in (("config A (dropout 0.0)", acc_a), ("config B (dropout 0.3)", acc_b)):
        print(f"  {name}: {a.tolist()} -> mean {a.mean():.4f}, std {a.std(ddof=1):.4f}")
    diff = acc_b.mean() - acc_a.mean()
    pooled = np.sqrt((acc_a.std(ddof=1) ** 2 + acc_b.std(ddof=1) ** 2) / 2)
    print(f"  difference of means {diff:+.4f} vs pooled seed std {pooled:.4f} -> ratio {diff / pooled:.2f}")
    print(f"  same seed, run twice: identical to the last digit; verdict: {'not' if abs(diff) < pooled else ''} distinguishable with 5 seeds")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (a, name) in enumerate(((acc_a, "config A: dropout 0.0"), (acc_b, "config B: dropout 0.3"))):
        ax.plot(np.full(5, k) + g.uniform(-0.08, 0.08, 5), a, "o", ms=7, color=SERIES[k], label=name)
        ax.hlines(a.mean(), k - 0.25, k + 0.25, color=SERIES[k], lw=2)
        ax.annotate(f"mean {a.mean():.4f}\nstd {a.std(ddof=1):.4f}", (k + 0.28, a.mean()), fontsize=8, color=INK_2, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["config A", "config B"]); ax.set_xlim(-0.6, 1.9)
    ax.set_ylabel("validation accuracy")
    ax.set_title("Five seeds per configuration (simulated): is the gap above the noise?")
    ax.legend(loc="upper left")
    save(fig, NB, "seed_noise_floor")


# --------------------------------------------------------------------------------------
# Section 7: the complete project: confident errors and the hyperparameter sweep grid
# --------------------------------------------------------------------------------------
def fig_sweep_grid_val_acc():
    banner("Section 7: confident errors (toy table) and the lr x weight-decay sweep (illustrative values)")
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    rows = [(17, 6, 0, 0.93), (42, 2, 4, 0.61), (58, 6, 6, 0.88), (71, 4, 2, 0.97), (90, 9, 7, 0.55), (133, 0, 0, 0.99)]
    print("  toy prediction table (index, true, pred, probability of the prediction):")
    for i, t, p, pr in rows:
        print(f"   {i:4d} {classes[t]:12s} {classes[p]:12s} {pr:.2f} {'wrong' if t != p else 'correct'}")
    wrong = sorted([r for r in rows if r[1] != r[2]], key=lambda r: -r[3])
    print("  wrong predictions sorted by confidence:", [(i, classes[t], classes[p], pr) for i, t, p, pr in wrong])
    print(f"  accuracy on the 6 rows: {sum(t == p for _, t, p, _ in rows)}/6 = {sum(t == p for _, t, p, _ in rows) / 6:.3f}")

    lrs = [5e-4, 2e-3, 8e-3]; wds = [0.0, 0.1]
    grid = np.array([[0.842, 0.840], [0.868, 0.866], [0.875, 0.871]])   # illustrative one-epoch validation accuracies
    print("  illustrative sweep (rows lr, cols weight decay):")
    for lr, row in zip(lrs, grid):
        print(f"   lr {lr:g}: {row.tolist()}")
    bi, bj = divmod(int(np.argmax(grid)), 2)
    print(f"  best by validation accuracy: lr {lrs[bi]:g}, weight decay {wds[bj]:g} -> {grid[bi, bj]:.3f}; "
          f"spread across the grid {grid.max() - grid.min():.3f}; effect of wd at the best lr {grid[bi, 0] - grid[bi, 1]:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(grid, cmap=SEQ_CMAP, vmin=0.83, vmax=0.88, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"wd {w:g}" for w in wds])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"lr {lr:g}" for lr in lrs])
    for a in range(3):
        for b in range(2):
            ax.text(b, a, f"{grid[a, b]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if grid[a, b] > 0.86 else INK)
    ax.set_title("Sweep: best validation accuracy per configuration (illustrative)")
    ax.set_xlabel("weight decay"); ax.set_ylabel("peak learning rate")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("validation accuracy")
    save(fig, NB, "sweep_grid_val_acc")


# --------------------------------------------------------------------------------------
# Section 8: debugging: the sqrt(0) gradient, the shape error, overfit one batch with a numpy MLP
# --------------------------------------------------------------------------------------
def fig_overfit_one_batch_numpy():
    banner("Section 8: anomaly example, shape error arithmetic, overfit-one-batch with a numpy MLP")
    x = np.array([0.0, 1.0, 4.0]); mask = np.array([0.0, 1.0, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = (np.sqrt(x) * mask).sum()
        grad = mask / (2 * np.sqrt(x))
    print(f"  sqrt example: x {x.tolist()}, mask {mask.tolist()} -> forward {fwd:.1f}, gradient {grad.tolist()}")
    print(f"  shape error: Linear(784, 10) applied to (2, 1, 28, 28) sees {2 * 1 * 28} rows of 28 features: (56x28) @ (784x10) fails")

    g = np.random.default_rng(1)
    n, d, h, k = 8, 3, 16, 4
    X = g.normal(size=(n, d)); y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    W1 = g.normal(size=(d, h)) * np.sqrt(2 / d); b1 = np.zeros(h)
    W2 = g.normal(size=(h, k)) * np.sqrt(1 / h); b2 = np.zeros(k)
    lr, steps = 0.5, 300
    losses, accs = [], []
    for t in range(steps + 1):
        z1 = X @ W1 + b1; a1 = np.maximum(z1, 0); logits = a1 @ W2 + b2
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(n), y]).mean()
        losses.append(loss); accs.append((p.argmax(1) == y).mean())
        if t == steps:
            break
        dlogits = p.copy(); dlogits[np.arange(n), y] -= 1; dlogits /= n
        dW2 = a1.T @ dlogits; db2 = dlogits.sum(0)
        da1 = dlogits @ W2.T; dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    print(f"  numpy MLP {d}->{h}->{k} on one batch of {n} examples, lr {lr}: uniform-guess loss ln({k}) = {np.log(k):.4f}")
    for t in (0, 10, 50, 100, 200, 300):
        print(f"   step {t:3d}: loss {losses[t]:.4f}, accuracy on the batch {accs[t]:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(np.arange(steps + 1), losses, color=SERIES[0])
    ax.axhline(np.log(k), color=MUTED, lw=0.8, ls="--")
    ax.annotate(f"ln {k} = {np.log(k):.2f}: a uniform guess", (steps, np.log(k)), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.annotate(f"step 100: {losses[100]:.4f}", (100, losses[100]), xytext=(10, 10), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("gradient step"); ax.set_ylabel("cross-entropy on the batch (log scale)")
    ax.set_title("Overfit-one-batch test: 8 examples, loss must go to ~0 (numpy MLP)")
    save(fig, NB, "overfit_one_batch_numpy")


# --------------------------------------------------------------------------------------
# Section 9: how many lines of the notebook move into each module of the project
# --------------------------------------------------------------------------------------
def fig_notebook_to_package_lines():
    banner("Section 9: non-empty code lines of the notebook per target module (counted from the notebook's code cells)")
    modules = ["models.py\n(Section 1)", "data.py\n(Section 2)", "engine.py\n(Sections 3-4)",
               "checkpoint.py\n(Sections 5-6)", "scripts/train.py\n(Section 7)", "tests/\n(Section 8)"]
    lines = [102, 165, 191, 104, 79, 63]
    for m, n in zip(modules, lines):
        print(f"  {m.replace(chr(10), ' '):32s} {n:4d} lines")
    print(f"  total {sum(lines)} lines; largest module engine.py = {100 * lines[2] / sum(lines):.0f} % of the code")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ypos = np.arange(len(modules))
    bars = ax.barh(ypos, lines, color=BLUE, edgecolor=SURFACE, height=0.6)
    for b, v in zip(bars, lines):
        ax.annotate(str(v), (v, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(ypos); ax.set_yticklabels(modules, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("non-empty code lines in the notebook")
    ax.set_title("From notebook sections to package modules (line counts)")
    save(fig, NB, "notebook_to_package_lines")


def main():
    fig_smallcnn_parameter_counts()
    fig_imbalance_weighted_sampling()
    fig_dataloader_workers_model()
    fig_training_history_early_stopping()
    fig_confusion_matrix_4class()
    fig_tensorboard_step_loss_and_lr()
    fig_resume_reproduces_run()
    fig_seed_noise_floor()
    fig_sweep_grid_val_acc()
    fig_overfit_one_batch_numpy()
    fig_notebook_to_package_lines()


if __name__ == "__main__":
    main()
