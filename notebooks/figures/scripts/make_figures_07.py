"""Figures for the companion guide of Notebook 07 · Transfer Learning and Modern Vision.

Everything here is numpy + matplotlib. There is no PyTorch, no download and no real image:
- the transfer-learning experiment is a tiny teacher-student task re-implemented in numpy
  (a "pretrained" one-hidden-layer network, a linear probe, fine-tuning with discriminative
  learning rates and training from scratch),
- Grad-CAM and Grad-CAM++ are computed by hand on a 4x4 feature map with three channels,
- the ViT, Dice/IoU, box-IoU/NMS and NT-Xent examples use hand-typed numbers.

Run:  python3 notebooks/figures/scripts/make_figures_07.py
Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

import math
from matplotlib.patches import Rectangle, FancyBboxPatch

NB = "07"


def banner(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# ----------------------------------------------------------------------------------------------
# A tiny numpy "deep learning" toolbox: one-hidden-layer ReLU network, softmax cross-entropy, Adam
# ----------------------------------------------------------------------------------------------
def softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def init_mlp(d_in, hidden, n_out, seed):
    """He-style init for the hidden layer, small init for the head. Returns [W1, b1, W2, b2]."""
    r = np.random.default_rng(seed)
    return [r.normal(0, np.sqrt(2.0 / d_in), (d_in, hidden)), np.zeros(hidden),
            r.normal(0, np.sqrt(1.0 / hidden), (hidden, n_out)), np.zeros(n_out)]


def mlp_features(params, X):
    return np.maximum(X @ params[0] + params[1], 0.0)


def mlp_logits(params, X):
    return mlp_features(params, X) @ params[2] + params[3]


def mlp_loss_grads(params, X, y):
    """Mean softmax cross-entropy and its gradients for all four parameters."""
    W1, b1, W2, b2 = params
    h = np.maximum(X @ W1 + b1, 0.0)
    p = softmax(h @ W2 + b2)
    N = len(y)
    loss = -np.mean(np.log(p[np.arange(N), y] + 1e-12))
    d = p.copy(); d[np.arange(N), y] -= 1.0; d /= N
    gW2, gb2 = h.T @ d, d.sum(0)
    dh = d @ W2.T; dh[h <= 0] = 0.0
    return loss, [X.T @ dh, dh.sum(0), gW2, gb2]


def linear_loss_grads(params, F_, y):
    """Softmax regression on fixed features F_ (the linear probe)."""
    W, b = params
    p = softmax(F_ @ W + b)
    N = len(y)
    loss = -np.mean(np.log(p[np.arange(N), y] + 1e-12))
    d = p.copy(); d[np.arange(N), y] -= 1.0; d /= N
    return loss, [F_.T @ d, d.sum(0)]


def adam_train(params, grad_fn, lrs, steps, eval_fn=None, every=10, batches=None, seed=0):
    """Adam (betas 0.9/0.999) with one learning rate per parameter (= parameter groups).
    batches: None for full-batch steps, otherwise (X, y, batch_size) for mini-batches."""
    r = np.random.default_rng(seed)
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    hist = []
    for t in range(1, steps + 1):
        if batches is None:
            loss, grads = grad_fn(params)
        else:
            X, y, bs = batches
            idx = r.choice(len(y), bs, replace=False)
            loss, grads = grad_fn(params, X[idx], y[idx])
        for i, (g, lr) in enumerate(zip(grads, lrs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** t); vh = v[i] / (1 - 0.999 ** t)
            params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if eval_fn is not None and (t % every == 0 or t == 1):
            hist.append((t, eval_fn(params)))
    return params, hist


# ----------------------------------------------------------------------------------------------
# The toy transfer task: a teacher network defines a SOURCE task (10 classes, plenty of data)
# and a TARGET task (4 classes, little data) that share the same hidden "features".
# ----------------------------------------------------------------------------------------------
D_IN, HID_TEACHER, HID_STUDENT, N_SRC, N_TGT = 20, 32, 64, 10, 4


def make_teacher(seed=7):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 1 / np.sqrt(D_IN), (D_IN, HID_TEACHER)); b1 = r.normal(0, 0.5, HID_TEACHER)
    V_src = r.normal(0, 1, (HID_TEACHER, N_SRC)); V_tgt = r.normal(0, 1, (HID_TEACHER, N_TGT))
    Xc = r.normal(size=(20_000, D_IN)); hc = np.maximum(Xc @ W1 + b1, 0)
    off_src, off_tgt = (hc @ V_src).mean(0), (hc @ V_tgt).mean(0)      # centre the class scores -> balanced classes
    return dict(W1=W1, b1=b1, V_src=V_src, V_tgt=V_tgt, off_src=off_src, off_tgt=off_tgt)


def teacher_data(teacher, n, task, seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, D_IN)).astype(np.float64)
    h = np.maximum(X @ teacher["W1"] + teacher["b1"], 0)
    S = h @ teacher["V_" + task] - teacher["off_" + task]
    return X, S.argmax(1)


def accuracy_fn(X, y):
    return lambda params: float((mlp_logits(params, X).argmax(1) == y).mean())


TEACHER = make_teacher()
X_src, y_src = teacher_data(TEACHER, 20_000, "src", seed=1)
X_tgt_test, y_tgt_test = teacher_data(TEACHER, 4_000, "tgt", seed=2)
X_tgt_pool, y_tgt_pool = teacher_data(TEACHER, 2_000, "tgt", seed=3)       # labelled target pool to subsample


def pretrain_student(seed=11, steps=3_000):
    params = init_mlp(D_IN, HID_STUDENT, N_SRC, seed)
    params, _ = adam_train(params, lambda p, X, y: mlp_loss_grads(p, X, y), [2e-3] * 4, steps,
                           batches=(X_src, y_src, 200), seed=seed)
    return params


def run_target_methods(pre_params, n_train, seed, steps=300, record=False):
    """Four strategies on n_train labelled target examples. Returns dict name -> final acc (or history)."""
    r = np.random.default_rng(seed)
    idx = r.choice(len(y_tgt_pool), n_train, replace=False)
    X, y = X_tgt_pool[idx], y_tgt_pool[idx]
    ev = accuracy_fn(X_tgt_test, y_tgt_test)
    out = {}
    # (a) linear probe on frozen PRETRAINED features
    F_tr, F_te = mlp_features(pre_params, X), mlp_features(pre_params, X_tgt_test)
    head = [np.zeros((HID_STUDENT, N_TGT)), np.zeros(N_TGT)]
    probe_eval = lambda p: float(((F_te @ p[0] + p[1]).argmax(1) == y_tgt_test).mean())
    _, h = adam_train(head, lambda p: linear_loss_grads(p, F_tr, y), [1e-2, 1e-2], steps, probe_eval, every=10)
    out["linear probe (pretrained features)"] = h
    # (b) linear probe on frozen RANDOM features (control)
    rand_params = init_mlp(D_IN, HID_STUDENT, N_SRC, seed + 100)
    F_tr, F_te = mlp_features(rand_params, X), mlp_features(rand_params, X_tgt_test)
    head = [np.zeros((HID_STUDENT, N_TGT)), np.zeros(N_TGT)]
    _, h = adam_train(head, lambda p: linear_loss_grads(p, F_tr, y), [1e-2, 1e-2], steps, probe_eval, every=10)
    out["linear probe (random features)"] = h
    # (c) fine-tuning from the pretrained weights, new head, discriminative LR (hidden layer: lr / 2.6)
    ft = [pre_params[0].copy(), pre_params[1].copy(), np.zeros((HID_STUDENT, N_TGT)), np.zeros(N_TGT)]
    _, h = adam_train(ft, lambda p: mlp_loss_grads(p, X, y), [1e-2 / 2.6, 1e-2 / 2.6, 1e-2, 1e-2], steps, ev, every=10)
    out["fine-tuning (discriminative LR)"] = h
    # (d) the same architecture from scratch
    sc = init_mlp(D_IN, HID_STUDENT, N_TGT, seed + 200)
    _, h = adam_train(sc, lambda p: mlp_loss_grads(p, X, y), [1e-2] * 4, steps, ev, every=10)
    out["from scratch"] = h
    return out if record else {k: v[-1][1] for k, v in out.items()}


PRE_PARAMS = None


def get_pretrained():
    global PRE_PARAMS
    if PRE_PARAMS is None:
        PRE_PARAMS = pretrain_student()
    return PRE_PARAMS


# ----------------------------------------------------------------------------------------------
# Section 1: decision matrix + feature reuse versus target-set size
# ----------------------------------------------------------------------------------------------
def fig_decision_matrix():
    banner("Figure decision_matrix (Section 1.2): the four cells")
    cells = {
        (0, 1): ("SMALL data, SIMILAR domain", "Feature extraction:\nfreeze the backbone,\ntrain a linear head.", BLUE),
        (1, 1): ("SMALL data, DIFFERENT domain", "Hardest case: features from\nearlier layers + small classifier,\nstrong regularisation or\nself-supervised pretraining.", ORANGE),
        (0, 0): ("LARGE data, SIMILAR domain", "Fine-tune the whole network\nwith a small learning rate.", AQUA),
        (1, 0): ("LARGE data, DIFFERENT domain", "Fine-tune everything, possibly\nwith a larger learning rate;\nthe init still speeds convergence.", YELLOW),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for (cx, cy), (title, body, col) in cells.items():
        ax.add_patch(FancyBboxPatch((cx + 0.04, cy + 0.04), 0.92, 0.92, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor=NEUTRAL_MID, edgecolor=col, linewidth=2.2))
        ax.text(cx + 0.5, cy + 0.84, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)
        ax.text(cx + 0.5, cy + 0.42, body, ha="center", va="center", fontsize=8.8, color=INK_2)
        print(f"  {title:30s} -> {body.replace(chr(10), ' ')}")
    ax.plot([0.22], [1.78], marker="o", color=INK, markersize=7)
    ax.annotate("Notebook 07: CIFAR-10 at 96 px\n(10k images, similar to ImageNet)", (0.22, 1.78), xytext=(0.3, 1.93),
                fontsize=7.5, color=INK, va="center")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["similar to source\n(natural photos)", "different from source\n(medical, satellite, sketches)"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["large\n(tens of\nthousands+)", "small\n(hundreds to\na few thousand)"])
    ax.set_xlabel("Similarity of the target data to the source data (ImageNet)")
    ax.set_ylabel("Size of the target dataset")
    ax.grid(False); ax.set_title("Which transfer strategy? The decision matrix of Section 1.2")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, NB, "decision_matrix")


def fig_feature_reuse_vs_data():
    banner("Figure feature_reuse_vs_data (Section 1.1): toy transfer experiment, accuracy vs target-set size")
    pre = get_pretrained()
    acc_src = float((mlp_logits(pre, X_src[:5000]).argmax(1) == y_src[:5000]).mean())
    print(f"  teacher-student toy: inputs {D_IN}-d, student hidden {HID_STUDENT} ReLU units; source task {N_SRC} classes, "
          f"{len(y_src):,} examples; target task {N_TGT} classes (chance {100 / N_TGT:.0f} %)")
    print(f"  source-task training accuracy of the 'pretrained' student: {100 * acc_src:.1f} %")
    print(f"  source class frequencies: {np.round(np.bincount(y_src) / len(y_src), 3)}")
    print(f"  target class frequencies (test): {np.round(np.bincount(y_tgt_test) / len(y_tgt_test), 3)}")
    print("  sample target rows (first 3 of the 20 inputs, label):")
    for i in range(5):
        print(f"    x = ({X_tgt_pool[i, 0]:+.2f}, {X_tgt_pool[i, 1]:+.2f}, {X_tgt_pool[i, 2]:+.2f}, ...)   y = {y_tgt_pool[i]}")
    sizes = [25, 50, 100, 200, 400, 800]
    seeds = [0, 1, 2]
    names = ["linear probe (pretrained features)", "linear probe (random features)", "fine-tuning (discriminative LR)", "from scratch"]
    table = {n: [] for n in names}
    for n_train in sizes:
        accs = {n: [] for n in names}
        for s in seeds:
            res = run_target_methods(pre, n_train, seed=s)
            for n in names:
                accs[n].append(res[n])
        for n in names:
            table[n].append(np.mean(accs[n]))
    print(f"\n  test accuracy (%, mean over {len(seeds)} seeds, {len(y_tgt_test):,} test examples), 300 Adam steps each")
    print(f"  {'n_train':>8s} " + " ".join(f"{n[:22]:>24s}" for n in names))
    for i, n_train in enumerate(sizes):
        print(f"  {n_train:8d} " + " ".join(f"{100 * table[n][i]:24.1f}" for n in names))
    fig, ax = plt.subplots()
    markers = ["o", "s", "D", "^"]
    for k, n in enumerate(names):
        ax.plot(sizes, [100 * a for a in table[n]], marker=markers[k], color=SERIES[k], label=n)
    ax.axhline(100 / N_TGT, color=MUTED, linestyle=":", linewidth=1)
    ax.text(sizes[-1], 100 / N_TGT + 1.2, "chance (25 %)", ha="right", fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("labelled target examples (log scale)"); ax.set_ylabel("target test accuracy (%)")
    ax.set_title("Reusing features beats learning them when target data are scarce (numpy toy task)")
    ax.legend(loc="lower right")
    save(fig, NB, "feature_reuse_vs_data")
    return table


# ----------------------------------------------------------------------------------------------
# Section 2: ResNet-18 anatomy and preprocessing arithmetic
# ----------------------------------------------------------------------------------------------
def resnet18_module_params():
    def basic_block(cin, cout, down):
        p = 9 * cin * cout + 2 * cout + 9 * cout * cout + 2 * cout       # two 3x3 convs (no bias) + two BNs
        if down:
            p += cin * cout + 2 * cout                                  # 1x1 projection + BN
        return p
    mods = [("conv1", 49 * 3 * 64), ("bn1", 128),
            ("layer1", 2 * basic_block(64, 64, False)),
            ("layer2", basic_block(64, 128, True) + basic_block(128, 128, False)),
            ("layer3", basic_block(128, 256, True) + basic_block(256, 256, False)),
            ("layer4", basic_block(256, 512, True) + basic_block(512, 512, False)),
            ("fc", 512 * 1000 + 1000)]
    return mods


def fig_resnet18_modules():
    banner("Figure resnet18_modules_and_maps (Section 2): parameters per module, feature-map sizes, normalisation")
    mods = resnet18_module_params()
    total = sum(p for _, p in mods)
    for name, p in mods:
        print(f"  {name:8s} {p:>12,} params  ({100 * p / total:5.1f} %)")
    print(f"  total    {total:>12,}   backbone (without fc) {total - mods[-1][1]:,}")
    stages = ["input", "conv1 (/2)", "maxpool (/4)", "layer1 (/4)", "layer2 (/8)", "layer3 (/16)", "layer4 (/32)"]
    div = [1, 2, 4, 4, 8, 16, 32]
    s224 = [224 // d for d in div]; s96 = [96 // d for d in div]
    print("  feature-map side length at 224 px:", dict(zip(stages, s224)))
    print("  feature-map side length at  96 px:", dict(zip(stages, s96)))
    print(f"  pixel cost relative to 32x32: 224 px -> {(224 / 32) ** 2:.0f}x, 96 px -> {(96 / 32) ** 2:.0f}x")
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    pixels = np.array([[120, 200, 60], [255, 255, 255], [0, 0, 0], [124, 116, 104]])
    print("  ImageNet normalisation (x/255 - mean) / std for sample uint8 pixels:")
    for px in pixels:
        scaled = px / 255.0; normed = (scaled - mean) / std
        print(f"    uint8 {tuple(px)} -> [0,1] {np.round(scaled, 4)} -> normalised {np.round(normed, 3)}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    names = [m for m, _ in mods][::-1]; vals = [p / 1e6 for _, p in mods][::-1]
    axes[0].barh(names, vals, color=BLUE, edgecolor=SURFACE)
    for i, v in enumerate(vals):
        axes[0].text(v + 0.1, i, f"{v * 1e6:,.0f}", va="center", fontsize=8, color=INK_2)
    axes[0].set_xlim(0, 10.2); axes[0].set_xlabel("parameters (millions)")
    axes[0].set_title("ResNet-18: 11.69 M parameters, 72 % of them in layer4")
    x = np.arange(len(stages))
    axes[1].plot(x, s224, marker="o", color=SERIES[0], label="224 px input (ImageNet recipe)")
    axes[1].plot(x, s96, marker="s", color=SERIES[1], label="96 px input (Section 3)")
    axes[1].set_yscale("log", base=2); axes[1].set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 224])
    axes[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[1].set_xticks(x); axes[1].set_xticklabels([s.split(" ")[0] for s in stages], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("feature-map side (pixels)"); axes[1].set_title("Spatial size shrinks 32x: 7x7 at 224 px, 3x3 at 96 px")
    axes[1].annotate("3x3 map for Grad-CAM", (6, 3), xytext=(3.6, 1.3), fontsize=8, color=SERIES[1],
                     arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=0.8))
    axes[1].legend(loc="upper right")
    save(fig, NB, "resnet18_modules_and_maps")


# ----------------------------------------------------------------------------------------------
# Section 3: discriminative learning rates, warm-up/cosine schedule, learning curves of the toy
# ----------------------------------------------------------------------------------------------
def fig_lr_groups_and_schedule():
    banner("Figure lr_groups_and_schedule (Section 3.3): parameter groups and the schedule")
    groups = ["fc (head)", "layer4", "layer3", "layer2", "conv1+bn1+layer1"]
    base, decay = 0.01, 2.6
    lrs = [base / decay ** d for d in range(len(groups))]
    for g, lr in zip(groups, lrs):
        print(f"  {g:18s} depth {groups.index(g)}  lr = {base} / {decay}^{groups.index(g)} = {lr:.5f}")
    steps_per_epoch, epochs, warmup_frac = 312, 3, 0.05
    total, warm = steps_per_epoch * epochs, int(warmup_frac * steps_per_epoch * epochs)
    t = np.arange(total)
    mult = np.where(t < warm, (t + 1) / warm, 0.5 * (1 + np.cos(np.pi * (t - warm) / max(total - warm, 1))))
    print(f"  schedule: {epochs} epochs x {steps_per_epoch} steps = {total} steps, warm-up {warmup_frac:.0%} = {warm} steps")
    for s in [0, 10, warm - 1, warm, total // 4, total // 2, 3 * total // 4, total - 1]:
        print(f"    step {s:4d}: multiplier {mult[s]:.4f}  -> head lr {base * mult[s]:.5f}, layer4 lr {lrs[1] * mult[s]:.5f}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bar(axes[0], groups, lrs, color=BLUE, label_fmt="{:.5f}")
    axes[0].set_yscale("log"); axes[0].set_ylim(1e-4, 3e-2)
    axes[0].set_ylabel("learning rate (log scale)"); axes[0].set_title("Discriminative LRs: divide by 2.6 per group from the head")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].plot(t, mult, color=BLUE)
    axes[1].axvline(warm, color=MUTED, linestyle=":", linewidth=1)
    axes[1].annotate(f"end of warm-up\n(step {warm})", (warm, 1.0), xytext=(warm + 90, 0.93), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    for e in range(1, epochs):
        axes[1].axvline(e * steps_per_epoch, color=GRID, linewidth=0.8)
    axes[1].set_xlabel("optimiser step"); axes[1].set_ylabel("multiplier of each group's learning rate")
    axes[1].set_title("5 % linear warm-up, then cosine decay to 0 (3 epochs)")
    save(fig, NB, "lr_groups_and_schedule")


def fig_finetune_learning_curves():
    banner("Figure finetune_learning_curves (Section 3): toy task, 100 labelled target examples")
    pre = get_pretrained()
    n_train = 100
    runs = [run_target_methods(pre, n_train, seed=s, record=True) for s in [0, 1, 2]]
    names = list(runs[0].keys())
    fig, ax = plt.subplots()
    markers = ["o", "s", "D", "^"]
    for k, n in enumerate(names):
        steps = [t for t, _ in runs[0][n]]
        acc = np.mean([[a for _, a in r[n]] for r in runs], axis=0) * 100
        ax.plot(steps, acc, color=SERIES[k], label=n, marker=markers[k], markevery=[len(steps) - 1])
        print(f"  {n:36s} step 10: {acc[1]:5.1f} %   step 50: {acc[5]:5.1f} %   step 100: {acc[10]:5.1f} %   step 300: {acc[-1]:5.1f} %")
    ax.axhline(25, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("Adam steps on the 100 target examples (full batch)"); ax.set_ylabel("target test accuracy (%)")
    ax.set_title("Learning curves with 100 labelled examples (toy task, mean of 3 seeds)")
    ax.legend(loc="lower right")
    save(fig, NB, "finetune_learning_curves")


# ----------------------------------------------------------------------------------------------
# Section 4: Grad-CAM and Grad-CAM++ by hand on a 4x4 map with three channels
# ----------------------------------------------------------------------------------------------
def fig_gradcam_toy():
    banner("Figure gradcam_toy (Section 4): hand-typed activations and gradients")
    A = np.array([
        [[0.2, 0.8, 0.6, 0.1], [0.4, 1.0, 0.7, 0.1], [0.1, 0.3, 0.2, 0.0], [0.0, 0.0, 0.1, 0.0]],     # channel 1: blob top-left
        [[0.0, 0.0, 0.1, 0.0], [0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.9, 1.0], [0.0, 0.1, 0.6, 0.8]],     # channel 2: blob bottom-right
        [[0.5, 0.5, 0.4, 0.6], [0.5, 0.6, 0.5, 0.5], [0.4, 0.5, 0.5, 0.6], [0.6, 0.5, 0.4, 0.5]],     # channel 3: texture everywhere
    ])
    g = np.stack([
        np.full((4, 4), 0.5),                                                                         # channel 1: +0.5 everywhere
        np.array([[-1.0, -1.0, 1.0, 1.0]] * 4),                                                       # channel 2: -1 left half, +1 right half
        np.full((4, 4), -0.25),                                                                       # channel 3: -0.25 everywhere
    ])
    alpha = g.mean(axis=(1, 2))
    weighted = alpha[:, None, None] * A
    cam = np.maximum(weighted.sum(0), 0)
    print("  activation maps A^k (4x4), k = 1..3:")
    for k in range(3):
        print(f"    A^{k + 1} =\n" + "\n".join("      " + " ".join(f"{v:4.1f}" for v in row) for row in A[k]))
    print("  gradients dy^c/dA^k: channel 1 = +0.5 everywhere, channel 2 = -1 (left half) / +1 (right half), channel 3 = -0.25 everywhere")
    print(f"  alpha_k = spatial mean of the gradient: {np.round(alpha, 4)}")
    print(f"  sum_k alpha_k A^k before ReLU:\n" + "\n".join("      " + " ".join(f"{v:+.3f}" for v in row) for row in weighted.sum(0)))
    print(f"  Grad-CAM = ReLU(...):\n" + "\n".join("      " + " ".join(f"{v:.3f}" for v in row) for row in cam))
    print(f"  example cell (row 2, col 2): 0.5*1.0 + 0*0.1 - 0.25*0.6 = {0.5 * 1.0 + 0 * 0.1 - 0.25 * 0.6:.3f}")
    print(f"  example cell (row 3, col 3): 0.5*0.2 + 0*0.9 - 0.25*0.5 = {0.5 * 0.2 + 0 * 0.9 - 0.25 * 0.5:+.3f} -> ReLU -> 0")
    # Grad-CAM++
    g2, g3 = g ** 2, g ** 3
    denom = 2 * g2 + A.sum(axis=(1, 2), keepdims=True) * g3
    a_pp = np.where(denom != 0, g2 / np.where(denom != 0, denom, 1), 0)
    w_pp = (a_pp * np.maximum(g, 0)).sum(axis=(1, 2))
    cam_pp = np.maximum((w_pp[:, None, None] * A).sum(0), 0)
    print(f"  Grad-CAM++: sum of A per channel = {np.round(A.sum(axis=(1, 2)), 2)}")
    print(f"  Grad-CAM++ pixel weights a_ij: channel 1 = 0.25/(0.5 + {A[0].sum():.1f}*0.125) = {a_pp[0, 0, 0]:.4f}; "
          f"channel 2 (right half) = 1/(2 + {A[1].sum():.1f}) = {a_pp[1, 0, 2]:.4f}; channel 3 (negative gradient) -> ReLU(g) = 0")
    print(f"  Grad-CAM++ channel weights w_k = {np.round(w_pp, 4)}  (16 x {a_pp[0, 0, 0]:.4f} x 0.5 = {16 * a_pp[0, 0, 0] * 0.5:.4f}; 8 x {a_pp[1, 0, 2]:.4f} x 1 = {8 * a_pp[1, 0, 2]:.4f})")
    print(f"  Grad-CAM++ map:\n" + "\n".join("      " + " ".join(f"{v:.3f}" for v in row) for row in cam_pp))
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    vmaxA = A.max()
    for k in range(3):
        ax = axes[0, k]
        ax.imshow(A[k], cmap=SEQ_CMAP, vmin=0, vmax=vmaxA)
        ax.set_title(f"A$^{k + 1}$  (alpha = {alpha[k]:+.2f})", fontsize=10)
        ax = axes[1, k]
        ax.imshow(g[k], cmap=DIV_CMAP, vmin=-1, vmax=1)
        ax.set_title(f"gradient of y$^c$ w.r.t. A$^{k + 1}$", fontsize=10)
        for i in range(4):
            for j in range(4):
                axes[0, k].text(j, i, f"{A[k, i, j]:.1f}", ha="center", va="center", fontsize=8,
                                color=SURFACE if A[k, i, j] > 0.55 else INK)
                axes[1, k].text(j, i, f"{g[k, i, j]:+.2f}", ha="center", va="center", fontsize=8,
                                color=SURFACE if abs(g[k, i, j]) > 0.6 else INK)
    for ax, m, title in [(axes[0, 3], cam, "Grad-CAM = ReLU(sum alpha_k A$^k$)"), (axes[1, 3], cam_pp, "Grad-CAM++ (weights w_k)")]:
        ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=m.max())
        ax.set_title(title, fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color=SURFACE if m[i, j] > 0.55 * m.max() else INK)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Grad-CAM by hand: channel 2 is ignored (its gradient averages to zero), Grad-CAM++ recovers it", fontsize=11, fontweight="bold")
    save(fig, NB, "gradcam_toy")


# ----------------------------------------------------------------------------------------------
# Section 5: ViT patches, tokens, parameter count and attention cost versus patch size
# ----------------------------------------------------------------------------------------------
def vit_params(image=32, P=4, D=192, depth=6, ncls=10, mlp_ratio=2.0, C=3):
    N = (image // P) ** 2
    patch = C * P * P * D + D
    cls, pos = D, (N + 1) * D
    hid = int(D * mlp_ratio)
    block = 2 * D + (3 * D * D + 3 * D) + (D * D + D) + 2 * D + (D * hid + hid) + (hid * D + D)
    head = 2 * D + D * ncls + ncls
    return dict(N=N, values=C * P * P, patch=patch, cls=cls, pos=pos, block=block, blocks=depth * block, head=head,
                total=patch + cls + pos + depth * block + head)


def fig_vit_patches():
    banner("Figure vit_patches_and_cost (Section 5): patchify an 8x8 image with P = 2, parameter counts")
    img = np.zeros((8, 8))
    yy, xx = np.mgrid[0:8, 0:8]
    img = np.exp(-((xx - 4.5) ** 2 + (yy - 3.5) ** 2) / 6.0)
    P = 2
    n_patches = (8 // P) ** 2
    print(f"  8x8 image, patch size {P}: N = (8/{P})^2 = {n_patches} patches, each flattened to {P}x{P}x3 = {P * P * 3} values; "
          f"sequence with class token = {n_patches + 1} tokens")
    for D, depth, label in [(96, 3, "FAST_RUN ViT"), (192, 6, "full-run ViT")]:
        print(f"  {label} (dim {D}, depth {depth}, mlp ratio 2, 10 classes) on 32x32 images:")
        print(f"    {'P':>3s} {'tokens N':>9s} {'values/patch':>13s} {'patch embed':>12s} {'pos embed':>10s} {'blocks':>10s} {'head':>7s} {'total':>10s} {'N^2':>7s}")
        for Pq in [2, 4, 8, 16]:
            r = vit_params(32, Pq, D, depth)
            print(f"    {Pq:3d} {r['N']:9d} {r['values']:13d} {r['patch']:12,} {r['pos']:10,} {r['blocks']:10,} {r['head']:7,} {r['total']:10,} {r['N'] ** 2:7,}")
    r4, r8 = vit_params(32, 4), vit_params(32, 8)
    print(f"  full ViT: one block = {r4['block']:,} params; patch 4 -> 8 changes patch embed by {r8['patch'] - r4['patch']:+,}, "
          f"pos embed by {r8['pos'] - r4['pos']:+,}, total by {r8['total'] - r4['total']:+,} ({100 * (r8['total'] - r4['total']) / r4['total']:.2f} %)")
    print(f"  ImageNet: 224 px, P = 16 -> N = {(224 // 16) ** 2}, {16 * 16 * 3} values per patch; P = 8 -> N = {(224 // 8) ** 2}, "
          f"attention cost x{((224 // 8) ** 2 / (224 // 16) ** 2) ** 2:.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1, 1.4, 1.2]})
    ax = axes[0]
    ax.imshow(img, cmap=SEQ_CMAP, extent=(0, 8, 8, 0))
    for k in range(0, 9):
        lw = 2.2 if k % P == 0 else 0.4
        ax.axhline(k, color=SURFACE, lw=lw); ax.axvline(k, color=SURFACE, lw=lw)
    for idx in range(n_patches):
        py, px = divmod(idx, 8 // P)
        ax.text(px * P + P / 2, py * P + P / 2, str(idx + 1), ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("8x8 image, P = 2: 16 patches", fontsize=10)
    ax = axes[1]
    ax.set_xlim(0, 18); ax.set_ylim(0, 3); ax.axis("off")
    ax.add_patch(Rectangle((0.1, 1.1), 0.8, 0.8, facecolor=ORANGE, edgecolor=SURFACE))
    ax.text(0.5, 1.5, "cls", ha="center", va="center", fontsize=8, color=SURFACE, fontweight="bold")
    for idx in range(n_patches):
        ax.add_patch(Rectangle((1.1 + idx, 1.1), 0.8, 0.8, facecolor=BLUE, edgecolor=SURFACE))
        ax.text(1.5 + idx, 1.5, str(idx + 1), ha="center", va="center", fontsize=7.5, color=SURFACE)
    ax.text(9, 2.5, "17 tokens of dimension D  =  [class token ; 16 patch embeddings]  +  positional embedding", ha="center", fontsize=8.5, color=INK_2)
    ax.text(9, 0.55, "each patch: 12 pixel values  ->  linear map E (= conv, kernel 2, stride 2)  ->  D numbers", ha="center", fontsize=8.5, color=INK_2)
    ax.set_title("Token sequence z_0", fontsize=10)
    ax = axes[2]
    Ps = [2, 4, 8, 16]
    tot = [vit_params(32, p)["total"] for p in Ps]
    n2 = [vit_params(32, p)["N"] ** 2 for p in Ps]
    ax.plot(Ps, tot, marker="o", color=SERIES[0], label="parameters (full ViT, dim 192, depth 6)")
    ax.plot(Ps, n2, marker="s", color=SERIES[1], label="token pairs per attention layer, N$^2$")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(Ps); ax.set_xticklabels([str(p) for p in Ps])
    ax.set_xlabel("patch size P (32x32 image)"); ax.set_ylabel("count (log scale)")
    ax.set_title("Patch size: cost falls 16x per doubling,\nparameters barely move", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, NB, "vit_patches_and_cost")


# ----------------------------------------------------------------------------------------------
# Section 6: Dice versus IoU; box IoU and NMS
# ----------------------------------------------------------------------------------------------
def fig_dice_vs_iou():
    banner("Figure dice_vs_iou (Section 6.1): three mask pairs and the curve")
    cases = []
    for name, (a0, a1), (b0, b1) in [("large overlap", (1, 6), (2, 7)), ("half overlap", (1, 5), (3, 7)), ("small overlap", (1, 4), (3, 7))]:
        Am = np.zeros((8, 8), bool); Am[a0:a1, a0:a1] = True
        Bm = np.zeros((8, 8), bool); Bm[b0:b1, b0:b1] = True
        I, U = (Am & Bm).sum(), (Am | Bm).sum()
        iou, dice = I / U, 2 * I / (Am.sum() + Bm.sum())
        cases.append((name, Am, Bm, I, U, iou, dice))
        print(f"  {name:14s} |A| = {Am.sum():2d}, |B| = {Bm.sum():2d}, intersection = {I:2d}, union = {U:2d}  ->  "
              f"IoU = {I}/{U} = {iou:.4f}, Dice = 2*{I}/({Am.sum()}+{Bm.sum()}) = {dice:.4f}, 2*IoU/(1+IoU) = {2 * iou / (1 + iou):.4f}")
    print("  trivial 'all background' model on the shapes data (89 % background): per-image mIoU ~ (0.89 + 0.3 + 0.3)/3 = "
          f"{(0.89 + 0.3 + 0.3) / 3:.3f}; dataset-level mIoU = 0.89/3 = {0.89 / 3:.3f}; pixel accuracy = 0.89")
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.9]})
    for ax, (name, Am, Bm, I, U, iou, dice) in zip(axes[:3], cases):
        canvas = np.zeros((8, 8)); canvas[Am] = 1; canvas[Bm] += 2
        ax.imshow(canvas, cmap=SEQ_CMAP, vmin=0, vmax=3)
        ax.set_title(f"{name}\nIoU {iou:.2f}, Dice {dice:.2f}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].text(0.02, -0.08, "light: A only, mid: B only, dark: both", transform=axes[0].transAxes, fontsize=7.5, color=MUTED)
    ax = axes[3]
    x = np.linspace(0, 1, 200)
    ax.plot(x, 2 * x / (1 + x), color=BLUE, label="Dice = 2 IoU / (1 + IoU)")
    ax.plot(x, x, color=MUTED, linestyle="--", linewidth=1, label="Dice = IoU (reference)")
    for k, (name, *_, iou, dice) in enumerate(cases):
        ax.plot([iou], [dice], marker="o", color=SERIES[1], markersize=7)
    ax.annotate("the three examples", (cases[1][5], cases[1][6]), xytext=(0.55, 0.42), fontsize=8, color=SERIES[1],
                arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=0.8))
    ax.set_xlabel("IoU"); ax.set_ylabel("Dice"); ax.set_title("Dice is always at least IoU", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, NB, "dice_vs_iou")


def box_iou(a, b):
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def fig_box_iou_and_nms():
    banner("Figure box_iou_and_nms (Section 6.4): box IoU worked example and NMS step by step")
    a = np.array([[10., 10., 40., 45.]]); b = np.array([[20., 20., 50., 55.]])
    tl, br = np.maximum(a[0, :2], b[0, :2]), np.minimum(a[0, 2:], b[0, 2:])
    inter = max(br[0] - tl[0], 0) * max(br[1] - tl[1], 0)
    area_a, area_b = (a[0, 2] - a[0, 0]) * (a[0, 3] - a[0, 1]), (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1])
    print(f"  boxes A = {a[0].tolist()}, B = {b[0].tolist()} (x1, y1, x2, y2)")
    print(f"  intersection corner: max of top-lefts = {tl.tolist()}, min of bottom-rights = {br.tolist()} -> "
          f"{br[0] - tl[0]:.0f} x {br[1] - tl[1]:.0f} = {inter:.0f}")
    print(f"  areas: |A| = {area_a:.0f}, |B| = {area_b:.0f}; union = {area_a:.0f} + {area_b:.0f} - {inter:.0f} = {area_a + area_b - inter:.0f}; "
          f"IoU = {inter:.0f}/{area_a + area_b - inter:.0f} = {inter / (area_a + area_b - inter):.4f}")
    pairs = [("A vs B", a[0], b[0]), ("A vs shifted by (2, 2)", a[0], a[0] + 2), ("A vs disjoint", a[0], np.array([50., 50., 80., 85.]))]
    for name, p, q in pairs:
        print(f"  IoU {name:24s} = {box_iou(p[None], q[None])[0, 0]:.4f}")
    boxes = np.array([[10., 10., 40., 45.], [12., 8., 42., 44.], [8., 14., 38., 48.],
                      [55., 20., 90., 50.], [58., 22., 92., 52.], [35., 15., 60., 45.]])
    scores = np.array([0.90, 0.75, 0.60, 0.85, 0.70, 0.50])
    M = box_iou(boxes, boxes)
    print("  six proposals (x1, y1, x2, y2, score):")
    for i, (bx, s) in enumerate(zip(boxes, scores)):
        print(f"    box {i}: {bx.tolist()}  score {s:.2f}")
    print("  IoU matrix:\n" + "\n".join("    " + " ".join(f"{v:5.2f}" for v in row) for row in M))
    order = list(np.argsort(-scores)); keep = []; step = 0
    print("  NMS with threshold 0.5:")
    while order:
        i = order[0]; keep.append(i); step += 1
        rest = order[1:]
        ious = M[i, rest] if rest else np.array([])
        dropped = [j for j, u in zip(rest, ious) if u > 0.5]
        order = [j for j, u in zip(rest, ious) if u <= 0.5]
        print(f"    step {step}: keep box {i} (score {scores[i]:.2f}); IoU with remaining {dict(zip(rest, np.round(ious, 2)))} -> drop {dropped}, left {order}")
    print(f"  kept: {keep}  (box 5 survives: its IoU with every kept box is below 0.5, a typical NMS false positive)")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axes[0]
    cols = [SERIES[0], SERIES[1], SERIES[2]]
    for (name, p, q), col, dy in zip(pairs, cols, [0, 0, 0]):
        pass
    demo = [(a[0], b[0], "IoU 0.31"), (a[0], a[0] + 2, f"IoU {box_iou(a, a + 2)[0, 0]:.2f}"), (a[0], np.array([50., 50., 80., 85.]), "IoU 0.00")]
    offsets = [(0, 0), (60, 0), (0, 55)]
    for (p, q, label), (ox, oy), col in zip(demo, offsets, cols):
        for bx, ls in [(p, "-"), (q, "--")]:
            ax.add_patch(Rectangle((bx[0] + ox, bx[1] + oy), bx[2] - bx[0], bx[3] - bx[1], fill=False, edgecolor=col, lw=1.8, linestyle=ls))
        ax.text(p[0] + ox, p[1] + oy - 2, label, fontsize=8.5, color=col)
    ax.set_xlim(0, 150); ax.set_ylim(140, 0); ax.set_aspect("equal"); ax.grid(False)
    ax.set_title("Box IoU: three pairs (solid vs dashed)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for ax, idx, title in [(axes[1], range(len(boxes)), "6 proposals with scores"), (axes[2], keep, "after NMS (IoU > 0.5 suppressed)")]:
        for j in idx:
            x1, y1, x2, y2 = boxes[j]
            col = SEQ_CMAP(0.25 + 0.75 * (scores[j] - 0.5) / 0.4)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.8))
            ax.text(x1 + 1, y1 - 1.5, f"{j}: {scores[j]:.2f}", fontsize=8, color=col)
        ax.set_xlim(0, 100); ax.set_ylim(100, 0); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    save(fig, NB, "box_iou_and_nms")


# ----------------------------------------------------------------------------------------------
# Section 7: NT-Xent on a batch of 4 pairs
# ----------------------------------------------------------------------------------------------
def nt_xent_np(z, tau):
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n2 = len(z); N = n2 // 2
    sim = z @ z.T / tau
    np.fill_diagonal(sim, -np.inf)
    targets = np.concatenate([np.arange(N, n2), np.arange(0, N)])
    m = sim.max(axis=1, keepdims=True)
    logp = sim - (m + np.log(np.exp(sim - m).sum(axis=1, keepdims=True)))
    return -logp[np.arange(n2), targets].mean(), np.exp(logp)


def fig_ntxent():
    banner("Figure ntxent_similarity_and_temperature (Section 7.2): 4 pairs = 8 views, 2-d unit embeddings")
    angles = np.array([0, 90, 180, 270, 12, 78, 200, 262], dtype=float)         # view i and view i+4 are a positive pair
    z = np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], axis=1)
    print("  embeddings (unit vectors by angle): " + ", ".join(f"view {i}: {a:.0f} deg" for i, a in enumerate(angles)))
    print("  positive pairs: (0,4) (1,5) (2,6) (3,7); angle differences " +
          ", ".join(f"{abs(angles[i] - angles[i + 4]):.0f} deg" for i in range(4)))
    S = z @ z.T
    print("  cosine-similarity matrix (8x8):\n" + "\n".join("    " + " ".join(f"{v:+.2f}" for v in row) for row in S))
    for tau in [0.5, 0.1]:
        loss, probs = nt_xent_np(z, tau)
        row = S[0].copy(); row[0] = np.nan
        logits = row / tau
        print(f"  tau = {tau}: view 0 logits sim/tau = {np.round(logits, 2).tolist()}")
        print(f"    softmax over the 7 other views = {np.round(probs[0], 4).tolist()}  -> p(partner = view 4) = {probs[0, 4]:.4f}, "
              f"-log = {-math.log(probs[0, 4]):.4f}")
        print(f"    mean loss over all 8 views = {loss:.4f}")
    print(f"  chance level log(2N - 1) = log 7 = {math.log(7):.4f} (every view equally similar to every other)")
    z_collapsed = np.ones((8, 2))
    print(f"  collapsed encoder (all views identical): loss = {nt_xent_np(z_collapsed, 0.5)[0]:.4f} at any temperature")
    r = np.random.default_rng(5)
    rand_losses = [nt_xent_np(r.normal(size=(8, 64)), 0.5)[0] for _ in range(200)]
    print(f"  random 64-d embeddings, tau 0.5: mean loss over 200 draws = {np.mean(rand_losses):.3f}")
    taus = np.linspace(0.05, 2.0, 100)
    curve = [nt_xent_np(z, t)[0] for t in taus]
    print(f"  loss of the good encoder at tau = 0.05: {curve[0]:.4f}, tau = 1: {nt_xent_np(z, 1.0)[0]:.4f}, tau = 2: {curve[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(S, cmap=DIV_CMAP, vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center", fontsize=6.8, color=SURFACE if abs(S[i, j]) > 0.6 else INK)
    for i in range(4):
        ax.add_patch(Rectangle((i + 4 - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=2))
        ax.add_patch(Rectangle((i - 0.5, i + 4 - 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=2))
    ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.grid(False)
    ax.set_xlabel("view index (0-3: view 1, 4-7: view 2)"); ax.set_ylabel("view index")
    ax.set_title("Cosine similarities; boxes = positive pairs", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = axes[1]
    ax.plot(taus, curve, color=BLUE, label="good encoder (the 8 views on the left)")
    ax.axhline(math.log(7), color=SERIES[1], linestyle="--", linewidth=1.2, label="random or collapsed encoder: log 7 = 1.95")
    for t in [0.1, 0.5]:
        l = nt_xent_np(z, t)[0]
        ax.plot([t], [l], marker="o", color=BLUE, markersize=7)
        ax.annotate(f"tau = {t}: loss {l:.2f}", (t, l), xytext=(t + 0.12, l + 0.12), fontsize=8, color=INK_2)
    ax.set_xlabel("temperature tau"); ax.set_ylabel("NT-Xent loss")
    ax.set_title("Lower temperature makes the same similarities count for more", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, NB, "ntxent_similarity_and_temperature")


def main():
    fig_decision_matrix()
    fig_feature_reuse_vs_data()
    fig_resnet18_modules()
    fig_lr_groups_and_schedule()
    fig_finetune_learning_curves()
    fig_gradcam_toy()
    fig_vit_patches()
    fig_dice_vs_iou()
    fig_box_iou_and_nms()
    fig_ntxent()


if __name__ == "__main__":
    main()
