"""Figures for the companion guide of Notebook 06 · Convolutional Neural Networks.

Everything here is numpy + matplotlib (plus scikit-learn's bundled ``china.jpg`` sample image and
``make_moons`` generator; nothing is downloaded). Cross-correlation, pooling, im2col, receptive-field
arithmetic and the cost formulas are re-implemented in numpy so that the notebook's numbers can be
reproduced without PyTorch. The "degradation" experiment of Section 6 is a small numpy MLP on the
two-moons problem, trained with hand-written backpropagation: a toy, clearly labelled as such.

Run:  python3 notebooks/figures/scripts/make_figures_06.py
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
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.patches import Rectangle

NB = "06"


# ----------------------------------------------------------------------------------------------
# numpy building blocks (same conventions as the notebook)
# ----------------------------------------------------------------------------------------------
def corr2d(x, w, stride=1, padding=0, dilation=1):
    """Valid 2-D cross-correlation of x (H, W) with w (kh, kw), with zero padding, stride, dilation."""
    if padding:
        x = np.pad(x, padding)
    kh, kw = w.shape
    eh, ew = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1
    win = sliding_window_view(x, (eh, ew))[::stride, ::stride, ::dilation, ::dilation]   # (Ho, Wo, kh, kw)
    return np.einsum("ijab,ab->ij", win, w)


def conv_out_size(n, k, stride=1, padding=0, dilation=1):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def pool2d(x, k=2, mode="max"):
    win = sliding_window_view(x, (k, k))[::k, ::k]
    return win.max(axis=(2, 3)) if mode == "max" else win.mean(axis=(2, 3))


def receptive_field(layers):
    """layers: list of (kernel, stride) -> list of (rf, jump)."""
    rf, jump, rows = 1, 1, []
    for k, s in layers:
        rf = rf + (k - 1) * jump
        jump = jump * s
        rows.append((rf, jump))
    return rows


def load_gray_photo():
    """sklearn's bundled china.jpg as a float grayscale array in [0, 1] (no download)."""
    from sklearn.datasets import load_sample_image
    img = load_sample_image("china.jpg").astype(np.float64) / 255.0      # (427, 640, 3)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img, gray


def annotate_grid(ax, M, fmt="{:.0f}", fontsize=9, color_rule=None):
    """Write the value of every cell of a small matrix on top of its imshow."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            c = color_rule(v) if color_rule else INK
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color=c)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def heat(ax, M, signed=False, vmax=None):
    if signed:
        m = vmax or np.abs(M).max() or 1.0
        return ax.imshow(M, cmap=DIV_CMAP, vmin=-m, vmax=m)
    return ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=vmax or (M.max() or 1.0))


# ----------------------------------------------------------------------------------------------
# Section 1: dense versus convolutional parameter counts
# ----------------------------------------------------------------------------------------------
def fig_dense_vs_conv_params():
    print("\n=== fig_dense_vs_conv_params ===")
    images = [("MNIST 28x28x1", 1, 28, 28), ("CIFAR 32x32x3", 3, 32, 32),
              ("ImageNet 224x224x3", 3, 224, 224), ("photo 427x640x3", 3, 427, 640)]
    hidden, c_out, k = 1024, 64, 3
    dense, conv = [], []
    print(f"{'image':22s} {'inputs':>10s} {'dense (1024 hidden)':>20s} {'conv 3x3 -> 64':>16s} {'ratio':>10s}")
    for name, C, H, W in images:
        n_in = C * H * W
        d = n_in * hidden + hidden
        c = c_out * C * k * k + c_out
        dense.append(d); conv.append(c)
        print(f"{name:22s} {n_in:>10,d} {d:>20,d} {c:>16,d} {d / c:>10,.0f}x")
    print(f"dense layer on the photo in float32: {dense[-1] * 4 / 1e9:.2f} GB")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(images)); wdt = 0.38
    ax.bar(xs - wdt / 2, dense, wdt, color=SERIES[0], label="dense layer, 1,024 hidden units", edgecolor=SURFACE)
    ax.bar(xs + wdt / 2, conv, wdt, color=SERIES[1], label="3x3 convolution, 64 output channels", edgecolor=SURFACE)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([n for n, *_ in images], fontsize=9)
    ax.set_ylabel("parameters (log scale)")
    ax.set_title("A dense layer grows with the image; a convolution does not")
    ax.annotate(f"{dense[2] / conv[2]:,.0f}x more", (xs[2] - wdt / 2, dense[2]), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.legend(loc="upper left")
    save(fig, NB, "dense_vs_conv_params")


# ----------------------------------------------------------------------------------------------
# Section 1.3: translation equivariance, checked numerically
# ----------------------------------------------------------------------------------------------
def fig_translation_equivariance():
    print("\n=== fig_translation_equivariance ===")
    n = 24
    img = np.zeros((n, n))
    img[6:12, 5:11] = 1.0                      # a bright square
    img[14:16, 4:20] = 0.6                     # a horizontal bar
    kernel = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])   # Sobel x
    shift = 5
    shifted = np.roll(img, shift, axis=(0, 1))
    conv_then_shift = np.roll(corr2d(img, kernel, padding=1), shift, axis=(0, 1))
    shift_then_conv = corr2d(shifted, kernel, padding=1)
    interior = (slice(shift + 1, -1), slice(shift + 1, -1))
    diff = np.abs(conv_then_shift - shift_then_conv)[interior].max()
    print(f"image {img.shape}, Sobel-x kernel, shift {shift} px; max |conv(shift x) - shift(conv x)| away from border = {diff:.1e}")
    print(f"max |conv(shift x) - shift(conv x)| including the wrap-around seam and border = {np.abs(conv_then_shift - shift_then_conv).max():.1f}")

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    panels = [(img, "input x", False), (shifted, f"shift(x) by {shift} px", False),
              (shift_then_conv, "conv(shift(x))", True), (conv_then_shift, "shift(conv(x))", True)]
    m = np.abs(shift_then_conv).max()
    for ax, (M, title, signed) in zip(axes, panels):
        heat(ax, M, signed=signed, vmax=m if signed else 1.0)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Translation equivariance: shifting the input shifts the feature map (same colours = same values)",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "translation_equivariance")


# ----------------------------------------------------------------------------------------------
# Section 2.2: a 5x5 image, a 3x3 edge kernel, every output cell worked out
# ----------------------------------------------------------------------------------------------
def fig_cross_correlation_worked():
    print("\n=== fig_cross_correlation_worked ===")
    X = np.array([[0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=float) * 9           # brightness 0 (dark) and 9 (bright)
    K = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)   # vertical-edge (Prewitt) kernel
    Y = corr2d(X, K)
    print("input X (5x5):\n", X.astype(int))
    print("kernel K (3x3):\n", K.astype(int))
    print("output Y = corr2d(X, K) (3x3):\n", Y.astype(int))
    # worked cell (0, 1): patch rows 0..2, cols 1..3
    patch = X[0:3, 1:4]
    terms = patch * K
    print("worked cell Y[0,1]: patch =\n", patch.astype(int), "\nelementwise products =\n", terms.astype(int),
          "\nsum =", int(terms.sum()))
    patch2 = X[0:3, 2:5]
    print("worked cell Y[0,2]: patch =\n", patch2.astype(int), "\nsum of products =", int((patch2 * K).sum()))
    print("output size: (5 - 3)//1 + 1 =", conv_out_size(5, 3))
    print("with padding 1 the output is", conv_out_size(5, 3, padding=1), "x", conv_out_size(5, 3, padding=1))
    Yp = corr2d(X, K, padding=1)
    print("padded output (5x5):\n", Yp.astype(int))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), gridspec_kw={"width_ratios": [5, 3, 3]})
    heat(axes[0], X, vmax=9); annotate_grid(axes[0], X, color_rule=lambda v: SURFACE if v > 4 else INK)
    axes[0].set_title("input X (5x5)")
    axes[0].add_patch(Rectangle((0.5, -0.5), 3, 3, fill=False, edgecolor=SERIES[1], linewidth=2))
    heat(axes[1], K, signed=True, vmax=1); annotate_grid(axes[1], K, fmt="{:+.0f}")
    axes[1].set_title("kernel K (3x3)")
    heat(axes[2], Y, signed=True, vmax=27); annotate_grid(axes[2], Y, fmt="{:.0f}",
                                                           color_rule=lambda v: SURFACE if abs(v) > 20 else INK)
    axes[2].set_title("output Y = X * K (3x3)")
    axes[2].add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor=SERIES[1], linewidth=2))
    fig.suptitle("Cross-correlation worked out: the orange patch times the kernel gives the orange output cell",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "cross_correlation_worked")


# ----------------------------------------------------------------------------------------------
# Section 2.3 + 2.6: convolution arithmetic (output size, receptive field)
# ----------------------------------------------------------------------------------------------
def fig_conv_arithmetic():
    print("\n=== fig_conv_arithmetic ===")
    configs = [("3x3, s=1, p=0 (valid)", 3, 1, 0, 1), ("3x3, s=1, p=1 (same)", 3, 1, 1, 1),
               ("3x3, s=2, p=1 (halve)", 3, 2, 1, 1), ("3x3, s=1, p=0, d=2 (dilated)", 3, 1, 0, 2),
               ("7x7, s=2, p=3 (ResNet stem)", 7, 2, 3, 1)]
    print(f"{'configuration':30s} " + " ".join(f"{n:>6d}" for n in (8, 16, 28, 32, 64, 224)))
    for name, k, s, p, d in configs:
        print(f"{name:30s} " + " ".join(f"{conv_out_size(n, k, s, p, d):>6d}" for n in (8, 16, 28, 32, 64, 224)))
    print("pitfall: 3x3, s=2, p=0 on 32 ->", conv_out_size(32, 3, 2, 0), "(not 16)")
    print("dilated 3x3 d=2 effective size:", 2 * (3 - 1) + 1)

    depth = np.arange(1, 9)
    plain3 = [rf for rf, _ in receptive_field([(3, 1)] * 8)]
    plain5 = [rf for rf, _ in receptive_field([(5, 1)] * 8)]
    eff = [(d * (3 - 1) + 1) for d in (1, 2, 4, 8, 16, 32, 64, 128)]     # effective size d*(k-1)+1 of a dilated 3x3
    dil = [rf for rf, _ in receptive_field([(e, 1) for e in eff])]
    vgg = [("conv", 3, 1), ("conv", 3, 1), ("pool", 2, 2), ("conv", 3, 1), ("conv", 3, 1), ("pool", 2, 2),
           ("conv", 3, 1), ("conv", 3, 1), ("pool", 2, 2)]
    vgg_rf = receptive_field([(k, s) for _, k, s in vgg])
    print("\nreceptive field after each layer (depth 1..8):")
    print("  plain 3x3 stack    :", plain3)
    print("  plain 5x5 stack    :", plain5)
    print("  dilated 3x3, d=1,2,4,8,...:", dil, "(effective sizes", eff, ")")
    print("  VGG-like conv,conv,pool x3 (9 layers): rf", [r for r, _ in vgg_rf], "jump", [j for _, j in vgg_rf])
    lenet = receptive_field([(5, 1), (2, 2), (5, 1), (2, 2)])
    print("  LeNet-5 features   : rf", [r for r, _ in lenet], "jump", [j for _, j in lenet])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ns = np.arange(8, 65)
    for i, (name, k, s, p, d) in enumerate(configs[:4]):
        axes[0].plot(ns, [conv_out_size(n, k, s, p, d) for n in ns], color=SERIES[i], label=name)
    axes[0].set(xlabel="input size n (pixels)", ylabel="output size (pixels)", title="Output size for one axis")
    axes[0].annotate("32 -> 16", (32, 16), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK_2)
    axes[0].legend(fontsize=8)
    axes[1].plot(depth, plain3, marker="o", color=SERIES[0], label="3x3 convolutions, stride 1")
    axes[1].plot(depth, plain5, marker="s", color=SERIES[1], label="5x5 convolutions, stride 1")
    axes[1].plot(np.arange(1, 10), [r for r, _ in vgg_rf], marker="^", color=SERIES[2], label="conv, conv, pool (VGG-like)")
    axes[1].plot(depth[:6], dil[:6], marker="D", color=SERIES[3], label="3x3 dilated, d = 1, 2, 4, 8, ...")
    axes[1].set(xlabel="layer", ylabel="receptive field (input pixels)", title="Receptive field versus depth")
    axes[1].set_yscale("log")
    axes[1].annotate(f"{vgg_rf[-1][0]} px after 9 layers", (9, vgg_rf[-1][0]), xytext=(-70, 8),
                     textcoords="offset points", fontsize=9, color=INK_2)
    axes[1].legend(fontsize=8, loc="upper left")
    save(fig, NB, "conv_arithmetic")


# ----------------------------------------------------------------------------------------------
# Section 2.5: hand-designed kernels on the bundled photograph
# ----------------------------------------------------------------------------------------------
def fig_hand_designed_kernels():
    print("\n=== fig_hand_designed_kernels ===")
    _, gray = load_gray_photo()
    gray = gray[::2, ::2]                                       # 214 x 320, enough for a figure
    kernels = {
        "box blur 5x5": np.full((5, 5), 1 / 25),
        "Gaussian 5x5": np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256,
        "Sobel x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "Sobel y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
        "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    }
    out = {name: corr2d(gray, k, padding=k.shape[0] // 2) for name, k in kernels.items()}
    out["gradient magnitude"] = np.sqrt(out["Sobel x"] ** 2 + out["Sobel y"] ** 2)
    print(f"photo (downsampled) {gray.shape}, mean brightness {gray.mean():.3f}")
    print(f"{'kernel':20s} {'sum of weights':>14s} {'min out':>9s} {'max out':>9s} {'mean |out|':>10s}")
    for name, k in kernels.items():
        o = out[name]
        print(f"{name:20s} {k.sum():>14.2f} {o.min():>9.2f} {o.max():>9.2f} {np.abs(o).mean():>10.3f}")
    o = out["gradient magnitude"]
    print(f"{'gradient magnitude':20s} {'':>14s} {o.min():>9.2f} {o.max():>9.2f} {np.abs(o).mean():>10.3f}")
    I = np.zeros((3, 3)); I[1, 1] = 1
    print("sharpen == identity - Laplacian:", np.array_equal(I - kernels["Laplacian"], kernels["sharpen"]))

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    axes = list(axes.flat)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1); axes[0].set_title("grayscale photo", fontsize=10)
    order = ["box blur 5x5", "Gaussian 5x5", "sharpen", "Sobel x", "Sobel y", "Laplacian", "gradient magnitude"]
    for ax, name in zip(axes[1:], order):
        o = out[name]
        if name in ("box blur 5x5", "Gaussian 5x5", "sharpen"):
            ax.imshow(np.clip(o, 0, 1), cmap="gray", vmin=0, vmax=1)
        elif name == "gradient magnitude":
            ax.imshow(o, cmap=SEQ_CMAP, vmin=0, vmax=np.percentile(o, 99))
        else:
            m = np.percentile(np.abs(o), 99)
            ax.imshow(o, cmap=DIV_CMAP, vmin=-m, vmax=m)
        ax.set_title(name, fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Hand-designed kernels: blurs smooth, Sobel/Laplacian respond to edges (blue/red = negative/positive)",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "hand_designed_kernels")


# ----------------------------------------------------------------------------------------------
# Section 2.7: im2col on a tiny example, plus the loop-versus-matmul timing
# ----------------------------------------------------------------------------------------------
def demo_im2col():
    print("\n=== demo_im2col (table in the guide, no figure) ===")
    X = np.arange(1, 17, dtype=float).reshape(4, 4)
    K = np.array([[1., 0.], [0., -1.]])
    win = sliding_window_view(X, (2, 2))                        # (3, 3, 2, 2)
    P = win.reshape(9, 4).T                                      # (k*k, L) = (4, 9)
    Wm = K.reshape(1, 4)
    Y = (Wm @ P).reshape(3, 3)
    print("X (4x4):\n", X.astype(int)); print("K (2x2):\n", K.astype(int))
    print("patch matrix P (4 x 9), one column per patch, row = kernel tap (top-left, top-right, bottom-left, bottom-right):")
    for r, lab in zip(P, ["tl", "tr", "bl", "br"]):
        print(f"  {lab}: " + " ".join(f"{int(v):3d}" for v in r))
    print("W (1 x 4) =", Wm.astype(int).tolist())
    print("W @ P reshaped (3x3):\n", Y.astype(int))
    print("check against direct corr2d:", np.array_equal(Y, corr2d(X, K)))
    print(f"memory: input {X.size} values, patch matrix {P.size} values ({P.size / X.size:.2f}x)")
    # same-padding 3x3 duplication factor
    Xs = rng.standard_normal((3, 10, 10))
    cols = np.stack([sliding_window_view(np.pad(c, 1), (3, 3)).reshape(100, 9) for c in Xs], axis=1).reshape(100, 27).T
    print(f"3x3 same padding on (3, 10, 10): patch matrix {cols.shape}, {cols.size / Xs.size:.1f}x the input")
    # timing: pure Python double loop vs the einsum (BLAS-backed) version
    x = rng.standard_normal((64, 64)); w = rng.standard_normal((3, 3)); xp = np.pad(x, 1)
    t0 = time.perf_counter()
    y_loop = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            y_loop[i, j] = np.sum(xp[i:i + 3, j:j + 3] * w)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        y_fast = corr2d(x, w, padding=1)
    t_fast = (time.perf_counter() - t0) / 20
    print(f"64x64 image, 3x3 kernel: Python double loop {t_loop * 1e3:.1f} ms, vectorised {t_fast * 1e3:.3f} ms "
          f"({t_loop / t_fast:.0f}x), max diff {np.abs(y_loop - y_fast).max():.1e}  (timings vary per machine)")


# ----------------------------------------------------------------------------------------------
# Section 3.1: pooling on a 4x4 map and small-shift invariance
# ----------------------------------------------------------------------------------------------
def fig_pooling():
    print("\n=== fig_pooling ===")
    X = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 7, 3, 1], [2, 1, 8, 6]], dtype=float)
    mx, av = pool2d(X, 2, "max"), pool2d(X, 2, "avg")
    print("input (4x4):\n", X.astype(int)); print("max pool 2x2:\n", mx.astype(int)); print("avg pool 2x2:\n", av)
    edges = {}
    for col in (2, 3, 4):
        e = np.zeros((2, 8)); e[:, col] = 1
        edges[col] = (e, pool2d(e, 2, "max"))
        print(f"edge at column {col}: max-pooled row = {edges[col][1][0].astype(int).tolist()}")

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), gridspec_kw={"height_ratios": [4, 2.2]})
    heat(axes[0, 0], X, vmax=8); annotate_grid(axes[0, 0], X, color_rule=lambda v: SURFACE if v > 5 else INK)
    axes[0, 0].set_title("input map (4x4)")
    for (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        axes[0, 0].add_patch(Rectangle((j - 0.5, i - 0.5), 2, 2, fill=False, edgecolor=SERIES[1], linewidth=1.5))
    heat(axes[0, 1], mx, vmax=8); annotate_grid(axes[0, 1], mx, color_rule=lambda v: SURFACE if v > 5 else INK)
    axes[0, 1].set_title("max pool 2x2 (2x2)")
    heat(axes[0, 2], av, vmax=8); annotate_grid(axes[0, 2], av, fmt="{:.2f}", color_rule=lambda v: SURFACE if v > 5 else INK)
    axes[0, 2].set_title("average pool 2x2 (2x2)")
    for ax, col in zip(axes[1], (2, 3, 4)):
        e, p = edges[col]
        strip = np.vstack([e[0], np.repeat(p[0], 2)])          # row 0: the edge, row 1: pooled (stretched to 8 cells)
        heat(ax, strip, vmax=1); ax.set_xticks([]); ax.set_yticks([0, 1]); ax.set_yticklabels(["edge", "pooled"], fontsize=8)
        ax.grid(False)
        for jj in range(8):
            ax.text(jj, 0, int(e[0, jj]), ha="center", va="center", fontsize=8, color=SURFACE if e[0, jj] else INK)
        for jj in range(4):
            ax.text(2 * jj + 0.5, 1, int(p[0, jj]), ha="center", va="center", fontsize=8, color=SURFACE if p[0, jj] else INK)
        same = "same as column 2" if col == 3 else ("different" if col == 4 else "reference")
        ax.set_title(f"edge at column {col}: {same}", fontsize=9)
    fig.suptitle("Pooling: each 2x2 window becomes one number; a 1-pixel shift inside a window is invisible",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "pooling")


# ----------------------------------------------------------------------------------------------
# Section 3.2-3.6: costs of the downsampling and channel-mixing operations (printed table)
# ----------------------------------------------------------------------------------------------
def demo_downsampling_costs():
    print("\n=== demo_downsampling_costs (tables in the guide) ===")
    H = W = 32; cin, cout = 64, 128
    p = cout * cin * 9 + cout
    macs_s1 = H * W * cout * cin * 9; macs_s2 = (H // 2) * (W // 2) * cout * cin * 9
    print(f"conv 3x3 {cin}->{cout} on {H}x{H}: params {p:,}; MACs stride 1 = {macs_s1 / 1e6:.1f} M, stride 2 = {macs_s2 / 1e6:.1f} M ({macs_s1 / macs_s2:.0f}x fewer)")
    print(f"GAP: (N, 512, 4, 4) -> (N, 512): classifier 512*10+10 = {512 * 10 + 10:,} params vs flatten 8192*10+10 = {8192 * 10 + 10:,}")
    print(f"1x1 conv 256->64: params {256 * 64 + 64:,} = a dense layer 256->64 applied at every pixel")
    for n, k, s, pd in [(8, 4, 2, 1), (16, 4, 2, 1), (3, 3, 2, 0), (7, 3, 2, 1)]:
        out = (n - 1) * s - 2 * pd + (k - 1) + 1
        print(f"transposed conv: n={n}, k={k}, s={s}, p={pd} -> {out}")


# ----------------------------------------------------------------------------------------------
# Section 3.5: depthwise-separable cost ratio versus channel count
# ----------------------------------------------------------------------------------------------
def fig_separable_ratio():
    print("\n=== fig_separable_ratio ===")
    C = np.arange(1, 1025)
    print(f"{'C_out':>6s} {'k=3 ratio':>10s} {'saving':>8s} {'k=5 ratio':>10s} {'saving':>8s}")
    for c in (8, 32, 128, 512):
        r3, r5 = 1 / c + 1 / 9, 1 / c + 1 / 25
        print(f"{c:>6d} {r3:>10.3f} {1 / r3:>7.1f}x {r5:>10.3f} {1 / r5:>7.1f}x")
    cin = cout = 128; size = 32
    p_std = cout * cin * 9 + cout; m_std = size * size * cout * cin * 9
    p_dw = cin * 9 + cin; m_dw = size * size * cin * 9
    p_pw = cout * cin + cout; m_pw = size * size * cout * cin
    print(f"128->128 on 32x32: standard {p_std:,} params / {m_std:,} MACs; depthwise {p_dw:,} / {m_dw:,}; "
          f"pointwise {p_pw:,} / {m_pw:,}; separable {p_dw + p_pw:,} / {m_dw + m_pw:,}")
    print(f"reduction: params {p_std / (p_dw + p_pw):.1f}x, MACs {m_std / (m_dw + m_pw):.1f}x, theory {1 / (1 / cout + 1 / 9):.2f}x")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(C, 1 / C + 1 / 9, color=SERIES[0], label="k = 3")
    ax.plot(C, 1 / C + 1 / 25, color=SERIES[1], label="k = 5")
    ax.axhline(1 / 9, color=SERIES[0], linestyle=":", linewidth=1)
    ax.axhline(1 / 25, color=SERIES[1], linestyle=":", linewidth=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set(xlabel="output channels C_out (log scale)", ylabel="separable cost / standard cost",
           title="Depthwise-separable convolution: cost ratio 1/C_out + 1/k^2")
    ax.plot([128], [1 / 128 + 1 / 9], marker="o", color=SERIES[0])
    ax.annotate(f"C_out = 128, k = 3: {1 / 128 + 1 / 9:.3f} (an {1 / (1 / 128 + 1 / 9):.1f}x saving)", (128, 1 / 128 + 1 / 9),
                xytext=(10, 12), textcoords="offset points", fontsize=9, color=INK_2)
    ax.text(1.1, 1 / 9 + 0.02, "limit 1/9", fontsize=8, color=SERIES[0])
    ax.text(1.1, 1 / 25 + 0.02, "limit 1/25", fontsize=8, color=SERIES[1])
    ax.legend()
    save(fig, NB, "separable_cost_ratio")


# ----------------------------------------------------------------------------------------------
# Section 4: LeNet-5 shapes and parameters per layer
# ----------------------------------------------------------------------------------------------
def fig_lenet_layers():
    print("\n=== fig_lenet_layers ===")
    layers = [("input", (1, 28, 28), 0), ("C1 conv5x5, 6", (6, 28, 28), 6 * 25 + 6), ("S2 pool", (6, 14, 14), 0),
              ("C3 conv5x5, 16", (16, 10, 10), 16 * 6 * 25 + 16), ("S4 pool", (16, 5, 5), 0),
              ("F5 400->120", (120,), 400 * 120 + 120), ("F6 120->84", (84,), 120 * 84 + 84), ("out 84->10", (10,), 84 * 10 + 10)]
    total = sum(p for _, _, p in layers); lin = sum(p for n, _, p in layers if n.startswith(("F", "out")))
    print(f"{'layer':16s} {'output shape':>14s} {'values':>8s} {'params':>8s}")
    for name, shape, p in layers:
        print(f"{name:16s} {str(shape):>14s} {int(np.prod(shape)):>8d} {p:>8,d}")
    print(f"total parameters {total:,}; in the three linear layers {lin:,} ({100 * lin / total:.1f} %)")
    print("output sizes: 28 -> conv5 p2 ->", conv_out_size(28, 5, padding=2), "-> pool2 ->", conv_out_size(28, 2, 2),
          "-> conv5 ->", conv_out_size(14, 5), "-> pool2 ->", conv_out_size(10, 2, 2))
    rf = receptive_field([(5, 1), (2, 2), (5, 1), (2, 2)])
    print("receptive field after C1, S2, C3, S4:", [r for r, _ in rf])

    names = [n for n, _, _ in layers]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vals = [int(np.prod(s)) for _, s, _ in layers]
    bar(axes[0], names, vals, color=SERIES[0], label_fmt="{:,}")
    axes[0].set(ylabel="values in the activation", title="Activation size shrinks: 784 -> 4,704 -> ... -> 10")
    axes[0].tick_params(axis="x", rotation=40)
    params = [p for _, _, p in layers]
    bar(axes[1], names, params, color=SERIES[0], label_fmt="{:,}")
    axes[1].set(ylabel="parameters", title=f"Where the {total:,} parameters live (96 % in F5, F6, out)")
    axes[1].tick_params(axis="x", rotation=40)
    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    save(fig, NB, "lenet_layers")


# ----------------------------------------------------------------------------------------------
# Section 5: the VGG-style network's per-layer parameters and MACs
# ----------------------------------------------------------------------------------------------
def fig_vgg_stage_costs():
    print("\n=== fig_vgg_stage_costs ===")
    w = 64
    convs = [("conv1 3->64 @32", 3, 64, 32), ("conv2 64->64 @32", 64, 64, 32), ("conv3 64->128 @16", 64, 128, 16),
             ("conv4 128->128 @16", 128, 128, 16), ("conv5 128->256 @8", 128, 256, 8), ("conv6 256->256 @8", 256, 256, 8)]
    rows = []
    for name, ci, co, res in convs:
        p = co * ci * 9                       # bias=False (BatchNorm follows); BN adds 2*co
        m = res * res * co * ci * 9
        rows.append((name, p, m))
    rows.append(("fc1 4096->256", 4096 * 256 + 256, 4096 * 256))
    rows.append(("fc2 256->10", 256 * 10 + 10, 256 * 10))
    print(f"{'layer':20s} {'params':>10s} {'MACs':>14s}")
    for name, p, m in rows:
        print(f"{name:20s} {p:>10,d} {m:>14,d}")
    conv_p = sum(p for n, p, _ in rows if n.startswith("conv")); bn_p = 2 * (64 + 64 + 128 + 128 + 256 + 256)
    print(f"six convolutions {conv_p:,} params (+ BatchNorm {bn_p:,}); fc1 {rows[6][1]:,}; total {conv_p + bn_p + rows[6][1] + rows[7][1]:,}")
    print("shapes: (3,32,32) -> (64,32,32) -> (128,16,16) -> (256,8,8) -> pool -> (256,4,4) -> 4096 -> 256 -> 10")
    print("two stacked 3x3 = one 5x5 receptive field with 18 vs 25 weights; three stacked = 7x7 with 27 vs 49")
    print("VGG-16 classifier: 25088*4096+4096 + 4096*4096+4096 + 4096*1000+1000 =",
          f"{25088 * 4096 + 4096 + 4096 * 4096 + 4096 + 4096 * 1000 + 1000:,}")

    names = [n for n, _, _ in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bar(axes[0], names, [p for _, p, _ in rows], color=SERIES[0])
    axes[0].set_yscale("log"); axes[0].set(ylabel="parameters (log scale)", title="Parameters per layer: the first dense layer dominates")
    bar(axes[1], names, [m / 1e6 for _, _, m in rows], color=SERIES[0])
    axes[1].set(ylabel="multiply-accumulates (millions)", title="MACs per layer: halve resolution, double width = constant cost")
    for ax in axes:
        ax.tick_params(axis="x", rotation=40)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    save(fig, NB, "vgg_stage_costs")


# ----------------------------------------------------------------------------------------------
# Section 5.3: crop-and-flip augmentation on a 32x32 photo patch (numpy)
# ----------------------------------------------------------------------------------------------
def fig_crop_flip_augmentation():
    print("\n=== fig_crop_flip_augmentation ===")
    img, _ = load_gray_photo()
    patch = img[60:316, 180:436]                                   # 256 x 256 x 3 crop of the photo
    small = patch.reshape(32, 8, 32, 8, 3).mean(axis=(1, 3))      # block average to 32 x 32 x 3
    mean, std = small.mean(axis=(0, 1)), small.std(axis=(0, 1))
    print(f"32x32x3 image: per-channel mean (R,G,B) = {np.round(mean, 3).tolist()}, std = {np.round(std, 3).tolist()}")
    normed = (small - mean) / std
    print(f"after normalisation: mean {np.round(normed.mean(axis=(0, 1)), 3).tolist()}, std {np.round(normed.std(axis=(0, 1)), 3).tolist()}")
    padded = np.pad(small, ((4, 4), (4, 4), (0, 0)))
    r = np.random.default_rng(42)
    augmented, table = [], []
    for i in range(5):
        oy, ox = r.integers(0, 9, size=2); flip = bool(r.integers(0, 2))
        a = padded[oy:oy + 32, ox:ox + 32]
        if flip:
            a = a[:, ::-1]
        augmented.append(a); table.append((i + 1, int(oy), int(ox), flip, int(oy) - 4, int(ox) - 4))
    print("sample  offset_y offset_x flip   shift_y shift_x   (offset 4 = no shift; shift = offset - 4)")
    for row in table:
        print("  {:>4d}  {:>8d} {:>8d} {!s:6s} {:>7d} {:>7d}".format(*row))
    print("padding 4 and cropping 32 gives 9 x 9 = 81 possible shifts, times 2 for the flip = 162 variants")

    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    axes[0].imshow(small); axes[0].set_title("original 32x32", fontsize=9)
    for ax, a, row in zip(axes[1:], augmented, table):
        ax.imshow(a); ax.set_title(f"shift ({row[4]:+d}, {row[5]:+d}){', flip' if row[3] else ''}", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Random crop (pad 4) and horizontal flip: the same image, five slightly different views",
                 fontsize=11, fontweight="bold")
    save(fig, NB, "crop_flip_augmentation")


# ----------------------------------------------------------------------------------------------
# Section 6: plain versus residual MLPs on two moons (toy, numpy, hand-written backprop)
# ----------------------------------------------------------------------------------------------
def make_mlp(depth, width, residual, seed):
    r = np.random.default_rng(seed)
    params = {"Win": r.standard_normal((width, 2)) * np.sqrt(2 / 2), "bin": np.zeros(width),
              "W": [r.standard_normal((width, width)) * np.sqrt(2 / width) for _ in range(depth)],
              "b": [np.zeros(width) for _ in range(depth)],
              "Wout": r.standard_normal((1, width)) * np.sqrt(1 / width), "bout": np.zeros(1)}
    params["alpha"] = 1 / np.sqrt(depth) if residual else None      # residual branch scale (keeps the signal size)
    return params


def mlp_forward(p, X):
    h = np.maximum(X @ p["Win"].T + p["bin"], 0)
    cache = [(X, h)]
    hs = [h]
    for W, b in zip(p["W"], p["b"]):
        z = h @ W.T + b
        a = np.maximum(z, 0)
        h = h + p["alpha"] * a if p["alpha"] is not None else a
        cache.append((z, a)); hs.append(h)
    logits = (h @ p["Wout"].T + p["bout"]).ravel()
    return logits, cache, hs


def mlp_loss_grads(p, X, y):
    logits, cache, hs = mlp_forward(p, X)
    prob = 0.5 * (1 + np.tanh(0.5 * logits))
    loss = float(np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))))
    g = ((prob - y) / len(y))[:, None]                     # dL/dlogits, (N, 1)
    grads = {"W": [None] * len(p["W"]), "b": [None] * len(p["b"])}
    grads["Wout"] = g.T @ hs[-1]; grads["bout"] = g.sum(axis=0)
    gh = g @ p["Wout"]                                     # dL/dh_last
    for l in reversed(range(len(p["W"]))):
        z, a = cache[l + 1]
        ga = gh * p["alpha"] if p["alpha"] is not None else gh
        gz = ga * (z > 0)
        grads["W"][l] = gz.T @ hs[l]; grads["b"][l] = gz.sum(axis=0)
        gh = gz @ p["W"][l] + (gh if p["alpha"] is not None else 0)
    X0, h0 = cache[0]
    gz0 = gh * (h0 > 0)
    grads["Win"] = gz0.T @ X0; grads["bin"] = gz0.sum(axis=0)
    return loss, grads, (prob > 0.5).astype(float)


def train_mlp(depth, width, residual, X, y, steps, lr, seed=0):
    p = make_mlp(depth, width, residual, seed)
    vel = {k: (np.zeros_like(v) if not isinstance(v, list) else [np.zeros_like(a) for a in v])
           for k, v in p.items() if k != "alpha"}
    losses = []
    for _ in range(steps):
        loss, g, pred = mlp_loss_grads(p, X, y)
        losses.append(loss)
        for k in ("Win", "bin", "Wout", "bout"):
            vel[k] = 0.9 * vel[k] + g[k]; p[k] -= lr * vel[k]
        for l in range(depth):
            for k in ("W", "b"):
                vel[k][l] = 0.9 * vel[k][l] + g[k][l]; p[k][l] -= lr * vel[k][l]
    loss, _, pred = mlp_loss_grads(p, X, y)
    return np.array(losses), float((pred == y).mean())


def gradient_norms_at_init(depth, width, residual, X, y, seed=0):
    p = make_mlp(depth, width, residual, seed)
    _, g, _ = mlp_loss_grads(p, X, y)
    return [np.linalg.norm(g["W"][l]) for l in range(depth)]


def fig_plain_vs_residual_toy():
    print("\n=== fig_plain_vs_residual_toy ===")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = y.astype(float)
    width, steps, lr = 16, 300, 0.05
    runs = [("plain, 2 hidden layers", 2, False, SERIES[0]), ("plain, 24 hidden layers", 24, False, SERIES[1]),
            ("residual, 24 hidden layers", 24, True, SERIES[2])]
    results = {}
    seeds = (0, 1, 2)
    print(f"two moons (400 points, noise 0.15), width {width}, full-batch SGD momentum 0.9, lr {lr}, {steps} steps, {len(seeds)} seeds")
    print(f"{'network':28s} {'loss@50':>8s} {'loss@150':>9s} {'final loss':>11s} {'train acc':>10s}")
    for name, depth, res, col in runs:
        curves, accs = [], []
        for s in seeds:
            losses, acc = train_mlp(depth, width, res, X, y, steps, lr, seed=s)
            curves.append(losses); accs.append(acc)
        curves = np.array(curves); mean_curve = curves.mean(axis=0)
        results[name] = (mean_curve, col)
        print(f"{name:28s} {mean_curve[49]:>8.3f} {mean_curve[149]:>9.3f} {mean_curve[-1]:>11.3f} {100 * np.mean(accs):>9.1f}%")
    gn_plain = gradient_norms_at_init(24, width, False, X, y)
    gn_res = gradient_norms_at_init(24, width, True, X, y)
    print("gradient norm of the weight matrix at initialisation (seed 0), layer 1 / layer 12 / layer 24:")
    print(f"  plain-24   : {gn_plain[0]:.2e} / {gn_plain[11]:.2e} / {gn_plain[23]:.2e}  (ratio last/first {gn_plain[23] / gn_plain[0]:.1f})")
    print(f"  residual-24: {gn_res[0]:.2e} / {gn_res[11]:.2e} / {gn_res[23]:.2e}  (ratio last/first {gn_res[23] / gn_res[0]:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (curve, col) in results.items():
        axes[0].plot(np.arange(1, steps + 1), curve, color=col, label=name)
    axes[0].set(xlabel="training step", ylabel="training loss (binary cross-entropy)",
                title="Toy degradation: deeper plain MLP trains worse, residual does not")
    axes[0].legend()
    L = np.arange(1, 25)
    axes[1].plot(L, gn_plain, marker="o", markersize=4, color=SERIES[1], label="plain, 24 layers")
    axes[1].plot(L, gn_res, marker="s", markersize=4, color=SERIES[2], label="residual, 24 layers")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="hidden layer (1 = nearest the input)", ylabel="gradient norm at initialisation (log)",
                title="Gradient size per layer before training")
    axes[1].legend()
    fig.suptitle("Numpy toy on two moons (not the notebook's CIFAR-10 experiment)", fontsize=10, color=INK_2, y=1.02)
    save(fig, NB, "plain_vs_residual_toy")


# ----------------------------------------------------------------------------------------------
# Section 7: the architecture tour as two scatter plots
# ----------------------------------------------------------------------------------------------
def fig_architecture_tour():
    print("\n=== fig_architecture_tour ===")
    tour = [("AlexNet", 2012, 61.0, 0.71, 56.5), ("VGG-16", 2014, 138.0, 15.5, 71.6), ("GoogLeNet", 2014, 6.6, 1.5, 69.8),
            ("ResNet-50", 2015, 25.6, 4.1, 76.1), ("ResNeXt-50", 2016, 25.0, 4.2, 77.6), ("DenseNet-121", 2016, 8.0, 2.8, 74.4),
            ("MobileNetV2", 2018, 3.5, 0.3, 71.9), ("EfficientNet-B0", 2019, 5.3, 0.4, 77.7), ("ConvNeXt-T", 2022, 28.6, 4.5, 82.5)]
    print(f"{'architecture':16s} {'year':>5s} {'params (M)':>10s} {'GFLOPs':>7s} {'top-1 %':>8s}")
    for n, yr, p, f, a in tour:
        print(f"{n:16s} {yr:>5d} {p:>10.1f} {f:>7.2f} {a:>8.1f}")
    d = {n: (p, f, a) for n, _, p, f, a in tour}
    print(f"VGG-16 / MobileNetV2 GFLOPs: {d['VGG-16'][1] / d['MobileNetV2'][1]:.0f}x; ResNet-50 / EfficientNet-B0 GFLOPs: {d['ResNet-50'][1] / d['EfficientNet-B0'][1]:.1f}x")
    print(f"AlexNet / GoogLeNet params: {d['AlexNet'][0] / d['GoogLeNet'][0]:.1f}x; ConvNeXt-T - ResNet-50 top-1: {d['ConvNeXt-T'][2] - d['ResNet-50'][2]:.1f} points")
    print(f"accuracy per million parameters: VGG-16 {d['VGG-16'][2] / d['VGG-16'][0]:.2f}, MobileNetV2 {d['MobileNetV2'][2] / d['MobileNetV2'][0]:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    offsets = {"ResNet-50": (6, -12), "ResNeXt-50": (6, 2), "GoogLeNet": (6, -10), "MobileNetV2": (6, 4),
               "DenseNet-121": (6, 2), "EfficientNet-B0": (6, 4), "ConvNeXt-T": (6, 2), "VGG-16": (-42, 6), "AlexNet": (6, 2)}
    for ax, idx, xlabel in [(axes[0], 0, "parameters (millions, log scale)"), (axes[1], 1, "forward-pass GFLOPs at 224x224 (log scale)")]:
        for n, _, p, f, a in tour:
            xval = (p, f)[idx]
            ax.scatter(xval, a, s=42, color=SERIES[0], zorder=3)
            ax.annotate(n, (xval, a), textcoords="offset points", xytext=offsets[n], fontsize=8, color=INK_2)
        ax.set_xscale("log"); ax.set(xlabel=xlabel, ylabel="ImageNet top-1 accuracy (%)")
    axes[0].set_title("Accuracy versus parameters"); axes[1].set_title("Accuracy versus compute")
    axes[1].annotate("", xy=(0.3, 71.9), xytext=(15.5, 71.6), arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.2))
    axes[1].text(1.6, 70.3, "same accuracy, 50x fewer FLOPs", fontsize=8, color=SERIES[1], ha="center")
    save(fig, NB, "architecture_tour")


# ----------------------------------------------------------------------------------------------
# Section 8: occlusion sensitivity of a toy template-matching "classifier"
# ----------------------------------------------------------------------------------------------
def fig_occlusion_toy():
    print("\n=== fig_occlusion_toy ===")
    r = np.random.default_rng(3)
    n = 32
    img = 0.25 + 0.08 * r.standard_normal((n, n))                 # grey, slightly noisy background
    img[4:12, 20:28] += 0.25 * r.random((8, 8))                   # a distracting bright texture top-right
    cy, cx = 20, 11                                                # the "object": a plus sign, centred here
    img[cy - 4:cy + 5, cx - 1:cx + 2] = 1.0
    img[cy - 1:cy + 2, cx - 4:cx + 5] = 1.0
    template = np.zeros((9, 9)); template[:, 3:6] = 1; template[3:6, :] = 1
    template = template - template.mean()

    def score(x):
        """Toy model: best normalised match of the plus template anywhere in the image, squashed to (0, 1)."""
        resp = corr2d(x, template)
        win = sliding_window_view(x, (9, 9))
        patch_std = win.std(axis=(2, 3)) + 1e-6
        ncc = resp / (patch_std * np.sqrt((template ** 2).sum()) * 9)
        return float(1 / (1 + np.exp(-12 * (ncc.max() - 0.5))))

    p_full = score(img)
    patch, stride, fill = 8, 2, 0.25
    rows = list(range(0, n - patch + 1, stride))
    occ = np.zeros((len(rows), len(rows)))
    for a, i in enumerate(rows):
        for b, j in enumerate(rows):
            x = img.copy(); x[i:i + patch, j:j + patch] = fill
            occ[a, b] = score(x)
    amin = np.unravel_index(occ.argmin(), occ.shape)
    print(f"image {n}x{n}, object (plus sign) centred at row {cy}, col {cx}; distractor texture at rows 4-12, cols 20-28")
    print(f"score with nothing hidden: {p_full:.3f}")
    print(f"occlusion grid: patch {patch}, stride {stride} -> {len(rows)}x{len(rows)} = {occ.size} positions")
    print(f"lowest score {occ.min():.3f} when the patch starts at row {rows[amin[0]]}, col {rows[amin[1]]} (covers the plus sign)")
    print(f"score when the patch covers the distractor (row 4, col 20): {occ[rows.index(4), rows.index(20)]:.3f}")
    print(f"score when the patch covers the plain background (row 0, col 0): {occ[0, 0]:.3f}")
    print(f"fraction of positions where the score drops below 0.5: {(occ < 0.5).mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0].set_title(f"toy image, score = {p_full:.2f}", fontsize=10)
    axes[0].add_patch(Rectangle((rows[amin[1]] - 0.5, rows[amin[0]] - 0.5), patch, patch, fill=False, edgecolor=SERIES[1], linewidth=2))
    im = axes[1].imshow(occ, cmap=SEQ_CMAP, vmin=0, vmax=1, extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    axes[1].set_title("score with an 8x8 patch hidden at each position", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="score")
    drop = p_full - occ
    im2 = axes[2].imshow(drop, cmap=SEQ_CMAP, vmin=0, vmax=max(drop.max(), 1e-3), extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    axes[2].set_title("drop in score (the evidence the model uses)", fontsize=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="score drop")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Occlusion sensitivity of a toy template-matching model (not a trained CNN)", fontsize=11, fontweight="bold")
    save(fig, NB, "occlusion_toy")


def main():
    fig_dense_vs_conv_params()
    fig_translation_equivariance()
    fig_cross_correlation_worked()
    fig_conv_arithmetic()
    fig_hand_designed_kernels()
    demo_im2col()
    fig_pooling()
    demo_downsampling_costs()
    fig_separable_ratio()
    fig_lenet_layers()
    fig_vgg_stage_costs()
    fig_crop_flip_augmentation()
    fig_plain_vs_residual_toy()
    fig_architecture_tour()
    fig_occlusion_toy()


if __name__ == "__main__":
    main()
