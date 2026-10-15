"""Figures for the companion guide of notebook 15 (deployment and model optimization).

Everything here is numpy + matplotlib only (no torch, no downloads). Each fig_* function builds its
data, prints every number quoted in the guide, and saves one PNG under notebooks/figures/15/.
Where a chart is a model or a schematic rather than a measurement, its title says so.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

import hashlib, math

NB = "15"


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# Shared toy pieces
# --------------------------------------------------------------------------------------
A_MS, C_MS = 0.30, 0.012          # latency model l(b) = a + c * b (ms); shape of a small CNN on a CPU


def latency_model(b):
    return A_MS + C_MS * np.asarray(b, dtype=float)


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def quantize_affine(x, bits=8, symmetric=True, per_channel_dim=None):
    """numpy twin of the notebook's quantize_affine: returns (q, scale, zero_point)."""
    x = np.asarray(x, dtype=float)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin, qmax = 0, 2 ** bits - 1
    if per_channel_dim is None:
        x_min, x_max = x.min(), x.max()
    else:
        dims = tuple(d for d in range(x.ndim) if d != per_channel_dim)
        x_min, x_max = x.min(axis=dims, keepdims=True), x.max(axis=dims, keepdims=True)
    if symmetric:
        scale = np.maximum(np.abs(x_max), np.abs(x_min)) / qmax
        zero_point = np.zeros_like(scale)
    else:
        x_min = np.minimum(x_min, 0.0)
        x_max = np.maximum(x_max, 0.0)
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - np.round(x_min / np.maximum(scale, 1e-12))
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(x / scale) + zero_point, qmin, qmax)
    return q, scale, zero_point


def dequantize_affine(q, scale, zero_point):
    return (q - zero_point) * scale


# --------------------------------------------------------------------------------------
# Section 1: latency percentiles of a simulated request stream
# --------------------------------------------------------------------------------------
def fig_latency_percentiles():
    banner("Section 1: latency percentiles (simulated 2,000 requests)")
    g = np.random.default_rng(1)
    base = g.lognormal(mean=np.log(2.0), sigma=0.15, size=2000)          # typical ~2 ms call
    spikes = g.random(2000) < 0.03                                       # 3 % of calls hit a GC pause / cache miss
    lat = base + spikes * g.uniform(3.0, 9.0, size=2000)
    print("  first 8 simulated latencies (ms):", np.round(lat[:8], 2).tolist())
    mean, p50, p90, p99 = lat.mean(), *np.percentile(lat, [50, 90, 99])
    print(f"  mean {mean:.2f} ms | p50 {p50:.2f} | p90 {p90:.2f} | p99 {p99:.2f} | max {lat.max():.2f}")
    print(f"  share of requests slower than 2x the median: {100 * (lat > 2 * p50).mean():.1f} %")
    print(f"  the mean sits at the {100 * (lat <= mean).mean():.0f}th percentile of the distribution")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(lat, bins=np.linspace(1, 12, 67), color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    top = ax.get_ylim()[1]
    for v, lab, col, frac, dx, ha in [(p50, f"p50 = {p50:.2f} ms", INK_2, 0.80, -4, "right"), (p90, f"p90 = {p90:.2f} ms", ORANGE, 0.92, 4, "left"),
                                      (p99, f"p99 = {p99:.2f} ms", RED, 0.92, 4, "left")]:
        ax.axvline(v, color=col, lw=1.2, ls="--")
        ax.annotate(lab, (v, top * frac), xytext=(dx, 0), textcoords="offset points", color=col, fontsize=9, ha=ha)
    ax.set_xlabel("latency of one request (ms)")
    ax.set_ylabel("requests")
    ax.set_title("A mean hides the tail: p50 vs p99 of 2,000 simulated requests")
    save(fig, NB, "latency_percentiles")


# --------------------------------------------------------------------------------------
# Section 1: latency and throughput versus batch size (linear latency model)
# --------------------------------------------------------------------------------------
def fig_latency_throughput_vs_batch():
    banner(f"Section 1: latency model l(b) = {A_MS} ms + {C_MS} ms x b")
    bs = np.array([1, 4, 16, 64, 256])
    lat = latency_model(bs)
    per_img = lat / bs * 1e3
    thr = bs / lat * 1e3
    print(f"  {'batch':>5s} {'latency ms':>10s} {'per image us':>12s} {'img/s':>9s}")
    for b, l, p, t in zip(bs, lat, per_img, thr):
        print(f"  {b:5d} {l:10.3f} {p:12.1f} {t:9.0f}")
    knee = A_MS / C_MS
    print(f"  asymptotic throughput 1/c = {1e3 / C_MS:,.0f} img/s; batch a/c = {knee:.0f} reaches half of it")
    print(f"  per-image cost at batch 1 is {per_img[0] / (C_MS * 1e3):.0f}x the arithmetic cost c")

    b_fine = np.logspace(0, 8, 200, base=2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(b_fine, latency_model(b_fine), color=BLUE)
    axes[0].plot(bs, lat, "o", color=BLUE)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size"); axes[0].set_ylabel("latency per call (ms)")
    axes[0].set_title("Latency: flat, then linear (model)")
    axes[0].annotate(f"overhead a = {A_MS} ms", (1, A_MS), xytext=(12, 34), textcoords="offset points", fontsize=9, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].plot(b_fine, b_fine / latency_model(b_fine) * 1e3, color=ORANGE)
    axes[1].plot(bs, thr, "o", color=ORANGE)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("batch size"); axes[1].set_ylabel("throughput (images / s)")
    axes[1].set_title("Throughput: steep, then saturating (model)")
    axes[1].axvline(knee, color=AXIS, ls=":", lw=1)
    axes[1].annotate(f"knee near a/c = {knee:.0f}", (knee, thr.max() * 0.35), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK_2)
    save(fig, NB, "latency_throughput_vs_batch")


# --------------------------------------------------------------------------------------
# Section 2: what is inside the bundle (bytes per tensor) and a checksum
# --------------------------------------------------------------------------------------
def fig_bundle_size_breakdown():
    banner("Section 2: size of the MNISTCNN state_dict, tensor by tensor (float32)")
    c1, c2, hidden = 16, 32, 128
    tensors = [("conv1.weight", (c1, 1, 3, 3)), ("conv1.bias", (c1,)), ("conv2.weight", (c2, c1, 3, 3)), ("conv2.bias", (c2,)),
               ("fc1.weight", (hidden, c2 * 7 * 7)), ("fc1.bias", (hidden,)), ("fc2.weight", (10, hidden)), ("fc2.bias", (10,))]
    counts = [int(np.prod(s)) for _, s in tensors]
    total = sum(counts)
    print(f"  {'tensor':14s} {'shape':>16s} {'values':>8s} {'bytes fp32':>10s} {'share':>6s}")
    for (n, s), c in zip(tensors, counts):
        print(f"  {n:14s} {str(s):>16s} {c:8,d} {4 * c:10,d} {100 * c / total:5.1f}%")
    print(f"  total {total:,d} parameters = {4 * total / 1024:.1f} KB as float32; linear layers hold "
          f"{100 * (counts[4] + counts[5] + counts[6] + counts[7]) / total:.1f} %")
    int8_bytes = 4 * (counts[0] + counts[1] + counts[2] + counts[3] + counts[5] + counts[7]) + (counts[4] + counts[6]) + 2 * 8
    print(f"  with int8 linear weights (biases and convolutions float32): {int8_bytes / 1024:.1f} KB, "
          f"{4 * total / int8_bytes:.2f}x smaller")
    manifest = b"mnist_cnn 1.0.0 mean=0.1307 std=0.3081"
    print(f"  sha256 of the manifest line {manifest!r}: {hashlib.sha256(manifest).hexdigest()[:16]}...")
    manifest2 = b"mnist_cnn 1.0.0 mean=0.1307 std=0.3080"
    print(f"  sha256 after changing one digit of std      : {hashlib.sha256(manifest2).hexdigest()[:16]}...")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [n for n, _ in tensors]
    kb = np.array(counts) * 4 / 1024
    y = np.arange(len(names))
    ax.barh(y, kb, color=BLUE, edgecolor=SURFACE, linewidth=1.2, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("size in the state_dict (KB, float32, log scale)")
    ax.set_title("fc1.weight is 97 % of the file: that is what int8 shrinks")
    for yi, v in zip(y, kb):
        ax.annotate(f"{v:,.1f} KB" if v >= 1 else f"{v:.2f} KB", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0.01, 3000)
    save(fig, NB, "bundle_size_breakdown")


# --------------------------------------------------------------------------------------
# Section 3: verifying an export numerically (float64 reference vs float32 "runtime" with another
# summation order), at several batch sizes
# --------------------------------------------------------------------------------------
def fig_export_verification():
    banner("Section 3: export verification on a toy 784 -> 64 -> 10 softmax network")
    g = np.random.default_rng(3)
    W1 = g.normal(0, 0.05, (784, 64)); b1 = g.normal(0, 0.05, 64)
    W2 = g.normal(0, 0.2, (64, 10)); b2 = g.normal(0, 0.05, 10)

    def reference(x):                               # "eager": float64, straightforward order
        h = np.maximum(x @ W1 + b1, 0)
        z = h @ W2 + b2
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def runtime(x):                                 # "exported": float32 and a chunked accumulation order
        x32 = x.astype(np.float32); W1f, W2f = W1.astype(np.float32), W2.astype(np.float32)
        h = np.zeros((x.shape[0], 64), np.float32)
        for s in range(0, 784, 112):                # 7 chunks, summed in a different order than one big dot
            h += x32[:, s:s + 112] @ W1f[s:s + 112]
        h = np.maximum(h + b1.astype(np.float32), 0)
        z = h @ W2f + b2.astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    batches = [1, 3, 8, 64, 256]
    diffs, argmax_agree = [], []
    for bsz in batches:
        x = g.random((bsz, 784))
        pr, pe = reference(x), runtime(x)
        diffs.append(np.abs(pe - pr).max())
        argmax_agree.append(np.mean(pe.argmax(1) == pr.argmax(1)))
        print(f"  batch {bsz:3d}: max |runtime - reference| = {diffs[-1]:.2e} | argmax agreement {100 * argmax_agree[-1]:.0f} % | "
              f"rows sum to {pe.sum(1).min():.6f} .. {pe.sum(1).max():.6f}")
    # a broken export: wrong normalization constant, argmax can still agree
    x = g.random((64, 784))
    pr = reference(x); pb = reference(x * 0.9)
    print(f"  a 'broken' export (inputs scaled by 0.9): max |diff| = {np.abs(pb - pr).max():.2e} "
          f"but argmax agreement still {100 * np.mean(pb.argmax(1) == pr.argmax(1)):.0f} %")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(batches, diffs, "o-", color=BLUE, label="correct export (float32 reorder)")
    ax.axhline(np.abs(pb - pr).max(), color=RED, ls="--", lw=1.2, label="broken export (wrong preprocessing)")
    ax.axhline(1e-5, color=AXIS, ls=":", lw=1)
    ax.annotate("atol = 1e-5 of the notebook's assert_allclose", (batches[0], 1e-5), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size used for the check"); ax.set_ylabel("max |exported - eager| (probabilities)")
    ax.set_title("Verify an export on probabilities, at several batch sizes")
    ax.legend(loc="center right")
    save(fig, NB, "export_verification")


# --------------------------------------------------------------------------------------
# Section 4: affine quantization worked example, error vs bits, straight-through estimator
# --------------------------------------------------------------------------------------
def fig_quantization_error_vs_bits():
    banner("Section 4: affine quantization of 8 weights, then error vs bits")
    w8 = np.array([-0.62, -0.31, -0.05, 0.00, 0.08, 0.27, 0.45, 0.93])
    for symmetric in (True, False):
        q, s, z = quantize_affine(w8, 8, symmetric=symmetric)
        w_hat = dequantize_affine(q, s, z)
        name = "symmetric int8 [-128, 127]" if symmetric else "asymmetric uint8 [0, 255]"
        print(f"  {name}: scale s = {s:.6f}, zero-point z = {int(z)}")
        print(f"    {'x':>7s} {'q':>5s} {'x_hat':>9s} {'error':>9s}")
        for x, qi, xh in zip(w8, q, w_hat):
            print(f"    {x:7.3f} {int(qi):5d} {xh:9.5f} {xh - x:9.5f}")
        err = w_hat - w8
        print(f"    max |error| {np.abs(err).max():.5f} (s/2 = {s / 2:.5f}) | rms error {np.sqrt((err ** 2).mean()):.5f} (s/sqrt(12) = {s / np.sqrt(12):.5f})")
    # a 4-bit symmetric pass on the same 8 weights
    q4, s4, z4 = quantize_affine(w8, 4, symmetric=True)
    print(f"  4-bit symmetric: s = {s4:.4f}, q = {q4.astype(int).tolist()}, x_hat = {np.round(dequantize_affine(q4, s4, z4), 3).tolist()}")

    # error vs bits on a Gaussian weight matrix with channel-dependent spread
    g = np.random.default_rng(4)
    W = g.normal(0, 0.05, (64, 256)) * g.uniform(0.25, 1.0, (64, 1))
    W[0, 0] = 0.35                                               # one outlier sets the per-tensor range
    bits = np.arange(2, 9)
    err_pt, err_pc = [], []
    print(f"  Gaussian weights 64 x 256, std 0.05 scaled per channel by 0.25..1.0, one outlier 0.35; relative RMS error")
    print(f"  {'bits':>4s} {'per-tensor':>10s} {'per-channel':>11s}")
    for b in bits:
        for store, kw in ((err_pt, {}), (err_pc, {"per_channel_dim": 0})):
            q, s, z = quantize_affine(W, int(b), symmetric=True, **kw)
            store.append(np.linalg.norm(dequantize_affine(q, s, z) - W) / np.linalg.norm(W))
        print(f"  {b:4d} {err_pt[-1]:10.4f} {err_pc[-1]:11.4f}")
    ratios = np.array(err_pt[:-1]) / np.array(err_pt[1:])
    print(f"  per-tensor error ratio from b to b+1 bits: {np.round(ratios, 2).tolist()} (about 2 each, i.e. one bit halves the error)")

    # straight-through estimator picture: 3-bit fake quantization of x in [-1, 1]
    xs = np.linspace(-1, 1, 801)
    q3, s3, z3 = quantize_affine(xs, 3, symmetric=True)
    xhat3 = dequantize_affine(q3, s3, z3)
    print(f"  3-bit symmetric fake quantization of [-1, 1]: step s = {s3:.4f}, levels = {np.round(np.unique(xhat3), 3).tolist()}")
    print("  forward uses the staircase, backward uses the slope 1 of the dashed identity (straight-through estimator)")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(bits, err_pt, "o-", color=BLUE, label="per-tensor scale")
    axes[0].plot(bits, err_pc, "s-", color=ORANGE, label="per-channel scales")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bits"); axes[0].set_ylabel("relative RMS error  ||w_hat - w|| / ||w||")
    axes[0].set_title("One more bit halves the error; per-channel helps")
    axes[0].legend()
    axes[0].annotate(f"8 bits, per-channel: {err_pc[-1]:.3f}", (8, err_pc[-1]), xytext=(-120, 14), textcoords="offset points", fontsize=9, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[1].plot(xs, xs, "--", color=MUTED, lw=1.2, label="identity (used by the backward pass)")
    axes[1].plot(xs, xhat3, color=BLUE, label="3-bit fake quantization (forward)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("fakequant(x)")
    axes[1].set_title("Straight-through estimator")
    axes[1].legend(loc="upper left", fontsize=8)
    save(fig, NB, "quantization_error_vs_bits")


# --------------------------------------------------------------------------------------
# Section 5: magnitude pruning threshold and an accuracy-vs-sparsity schematic
# --------------------------------------------------------------------------------------
def fig_pruning():
    banner("Section 5: magnitude pruning on a toy weight vector, global vs per-layer")
    g = np.random.default_rng(5)
    w_fc = g.normal(0, 0.02, 2000)            # a big linear layer with small weights
    w_conv = g.normal(0, 0.12, 200)           # a small convolution with large weights
    w_all = np.concatenate([w_fc, w_conv])
    target = 0.8
    thr = np.quantile(np.abs(w_all), target)
    mask = np.abs(w_all) > thr
    print(f"  weights: fc layer 2,000 values (std 0.02), conv layer 200 values (std 0.12)")
    print(f"  global 80 % magnitude pruning: threshold |w| <= {thr:.4f} is zeroed; {int((~mask).sum())} of {len(w_all)} weights removed")
    print(f"  resulting sparsity: fc {100 * (~mask[:2000]).mean():.1f} %, conv {100 * (~mask[2000:]).mean():.1f} %, overall {100 * (~mask).mean():.1f} %")
    thr_fc, thr_conv = np.quantile(np.abs(w_fc), target), np.quantile(np.abs(w_conv), target)
    print(f"  per-layer 80 % instead would use thresholds fc {thr_fc:.4f} and conv {thr_conv:.4f}")
    first = np.round(w_all[:8], 4)
    print(f"  first 8 fc weights {first.tolist()} -> kept: {(np.abs(w_all[:8]) > thr).astype(int).tolist()}")
    # iterative relative amounts (Exercise 3)
    cur = 0.0
    for tgt in (0.5, 0.8, 0.95):
        rel = (tgt - cur) / (1 - cur)
        print(f"  iterative schedule: from {cur:.0%} to {tgt:.0%} sparsity removes {rel:.1%} of the remaining weights")
        cur = tgt
    # accuracy-vs-sparsity schematic (logistic cliff), labelled as a model
    s = np.linspace(0, 1, 201)
    acc_oneshot = 99.0 - 88.0 / (1 + np.exp(-(s - 0.86) / 0.035))
    acc_ft = 99.0 - 88.0 / (1 + np.exp(-(s - 0.965) / 0.012))
    print("  schematic curves (not measured): acc_oneshot(s) = 99 - 88 / (1 + exp(-(s - 0.86)/0.035)), fine-tuned cliff at 0.965")
    for sv in (0.5, 0.8, 0.9, 0.95):
        i = int(round(sv * 200))
        print(f"    sparsity {sv:.0%}: one-shot {acc_oneshot[i]:.1f} %, after fine-tuning {acc_ft[i]:.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(w_all, bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    axes[0].axvspan(-thr, thr, color=ORANGE, alpha=0.25, lw=0)
    axes[0].annotate(f"|w| <= {thr:.3f}: pruned\n({target:.0%} of the weights)", (thr, axes[0].get_ylim()[1] * 0.7), xytext=(18, 0),
                     textcoords="offset points", fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[0].set_xlabel("weight value"); axes[0].set_ylabel("count")
    axes[0].set_title("Magnitude pruning: zero the smallest 80 %")
    axes[1].plot(100 * s, acc_oneshot, color=BLUE, label="one-shot, no fine-tuning")
    axes[1].plot(100 * s, acc_ft, color=ORANGE, label="with fine-tuning")
    axes[1].set_xlabel("global weight sparsity (%)"); axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_title("Accuracy vs sparsity (schematic)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower left")
    save(fig, NB, "pruning_threshold_and_sparsity_curve")


# --------------------------------------------------------------------------------------
# Section 6: soft targets at several temperatures and the distillation loss
# --------------------------------------------------------------------------------------
def fig_soft_targets():
    banner("Section 6: teacher logits softened at T = 1, 2, 5, 10 and the distillation loss")
    z_t = np.array([6.0, 3.0, 1.0, 0.0, -1.0])        # teacher logits for 5 classes, class 0 is right
    temps = [1, 2, 5, 10]
    probs = {T: softmax(z_t / T) for T in temps}
    print(f"  teacher logits {z_t.tolist()}")
    for T in temps:
        print(f"  T = {T:2d}: probabilities {np.round(probs[T], 3).tolist()} | top-1 {probs[T].max():.3f} | runner-up {np.sort(probs[T])[-2]:.3f}")
    z_s = np.array([4.0, 3.5, 0.5, 0.0, -0.5])        # a student that confuses classes 0 and 1
    T, alpha = 4.0, 0.7
    y = 0
    p_t, p_s = softmax(z_t / T), softmax(z_s / T)
    kl = np.sum(p_t * (np.log(p_t) - np.log(p_s)))
    soft = T * T * kl
    hard = -np.log(softmax(z_s)[y])
    total = alpha * soft + (1 - alpha) * hard
    print(f"  student logits {z_s.tolist()}, T = {T:.0f}, alpha = {alpha}")
    print(f"  softened teacher {np.round(p_t, 3).tolist()} | softened student {np.round(p_s, 3).tolist()}")
    print(f"  KL(teacher || student) at T = {kl:.4f}; x T^2 = {soft:.4f} | CE(hard label) = {hard:.4f}")
    print(f"  loss = {alpha} x {soft:.4f} + {1 - alpha:.1f} x {hard:.4f} = {total:.4f}")
    grad = T * (p_s - p_t)
    print(f"  gradient of the T^2-scaled soft term w.r.t. the student logits: T (p_s - p_t) = {np.round(grad, 4).tolist()}")

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, T in zip(axes, temps):
        ax.bar(range(5), probs[T], color=BLUE, edgecolor=SURFACE, linewidth=1.2, width=0.65)
        ax.set_title(f"T = {T}")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"c{i}" for i in range(5)])
        ax.set_ylim(0, 1)
        ax.grid(False)
        ax.annotate(f"{probs[T].max():.2f}", (0, probs[T].max()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    axes[0].set_ylabel("probability")
    fig.suptitle("Soft targets: the same logits at increasing temperature", fontweight="bold")
    save(fig, NB, "soft_targets_temperature")


# --------------------------------------------------------------------------------------
# Section 6 (second chart): the compression menu, size vs accuracy (toy, labelled)
# --------------------------------------------------------------------------------------
def fig_compression_menu():
    banner("Section 6: compression menu, size vs accuracy vs latency (illustrative numbers)")
    options = [  # name, size KB, accuracy %, relative CPU latency at batch 1
        ("CNN float32", 808.0, 99.1, 1.00),
        ("CNN int8 dynamic", 217.0, 99.1, 0.98),
        ("CNN 90 % pruned (sparse storage)", 162.0, 98.6, 1.00),
        ("CNN structured, half the channels", 420.0, 98.5, 0.70),
        ("distilled MLP float32", 199.0, 97.9, 0.25),
        ("distilled MLP int8", 52.0, 97.9, 0.15),
    ]
    print("  illustrative, not measured; shaped like the notebook's results")
    print(f"  {'option':36s} {'size KB':>8s} {'acc %':>6s} {'rel. latency':>12s}")
    for n, kb, acc, lat in options:
        print(f"  {n:36s} {kb:8.0f} {acc:6.1f} {lat:12.2f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (n, kb, acc, lat) in enumerate(options):
        ax.scatter(kb, acc, s=60 + 500 * lat, color=SERIES[i], edgecolor=SURFACE, linewidth=1.0, zorder=3, label=n)
    ax.set_xscale("log")
    ax.set_xlabel("model size (KB, log scale)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("The compression menu (illustrative): marker area = relative latency")
    ax.set_ylim(97.5, 99.4)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, NB, "compression_menu_size_vs_accuracy")


# --------------------------------------------------------------------------------------
# Section 7: the request-batching simulation of Exercise 5 on the latency model
# --------------------------------------------------------------------------------------
def simulate_batching(rate_per_ms, max_batch, max_wait_ms, latency_fn, n_requests=4000, seed=0):
    g = np.random.default_rng(seed)
    arrivals = np.cumsum(g.exponential(1.0 / rate_per_ms, n_requests))
    t_free, i, latencies = 0.0, 0, []
    while i < n_requests:
        start = max(t_free, arrivals[i])
        deadline = start + max_wait_ms
        j = i + 1
        while j < n_requests and j - i < max_batch and arrivals[j] <= deadline:
            j += 1
        full = (j - i == max_batch)
        launch = max(start, arrivals[j - 1]) if full else deadline
        finish = launch + latency_fn(j - i)
        latencies.extend((finish - arrivals[i:j]).tolist())
        t_free, i = finish, j
    lat = np.array(latencies)
    return {"throughput": n_requests / (t_free - arrivals[0]) * 1000.0, "mean": lat.mean(), "p99": np.percentile(lat, 99)}


def fig_batching_simulation():
    banner(f"Section 7: dynamic batching simulation (l(b) = {A_MS} + {C_MS} b ms, max_wait = 2 ms)")
    cap = 1000.0 / A_MS
    print(f"  unbatched capacity ~ 1000 / a = {cap:,.0f} req/s")
    loads = {"low load (0.5x capacity)": 0.5 / A_MS, "high load (3x capacity)": 3.0 / A_MS}
    max_batches = [1, 2, 4, 8, 16, 32, 64]
    results = {}
    for label, rate in loads.items():
        results[label] = [simulate_batching(rate, mb, 2.0, latency_model) for mb in max_batches]
        print(f"  {label}: {rate * 1000:,.0f} req/s")
        for mb, r in zip(max_batches, results[label]):
            print(f"    max_batch {mb:3d}: throughput {r['throughput']:8,.0f} req/s | mean {r['mean']:8.2f} ms | p99 {r['p99']:8.2f} ms")
    # a toy hand trace of one batch
    arr = np.array([0.00, 0.40, 0.90, 2.60])
    print("  hand trace, max_batch 4, max_wait 2 ms: arrivals", arr.tolist(), "ms; server free at 0")
    print(f"    deadline 2.00 ms collects the first three; launch at 2.00, batch of 3 takes l(3) = {latency_model(3):.3f} ms, "
          f"finish {2.0 + latency_model(3):.3f}; latencies {np.round(2.0 + latency_model(3) - arr[:3], 3).tolist()} ms")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for k, (label, rs) in enumerate(results.items()):
        axes[0].plot(max_batches, [r["p99"] for r in rs], "o-", color=SERIES[k], label=label)
        axes[1].plot(max_batches, [r["throughput"] for r in rs], "o-", color=SERIES[k], label=label)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("max_batch"); axes[0].set_ylabel("p99 latency (ms)"); axes[0].set_title("Latency vs batching (simulation)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("max_batch"); axes[1].set_ylabel("throughput (requests / s)"); axes[1].set_title("Throughput vs batching (simulation)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8, loc="center right")
    save(fig, NB, "batching_simulation")


# --------------------------------------------------------------------------------------
# Section 8: PSI worked example
# --------------------------------------------------------------------------------------
def fig_psi_drift():
    banner("Section 8: population stability index, reference vs production")
    g = np.random.default_rng(8)
    ref = g.normal(0.13, 0.03, 5000)                    # e.g. mean image intensity at training time
    prod_same = g.normal(0.13, 0.03, 5000)
    prod_shift = g.normal(0.145, 0.036, 5000)           # dimmer-contrast camera: mean +0.015, spread +20 %
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-6

    def psi_table(prod):
        p = np.histogram(ref, edges)[0] / len(ref) + eps
        q = np.histogram(prod, edges)[0] / len(prod) + eps
        return p, q, (q - p) * np.log(q / p)

    for name, prod in [("unchanged", prod_same), ("shifted", prod_shift)]:
        p, q, contrib = psi_table(prod)
        print(f"  scenario '{name}': PSI = {contrib.sum():.4f}")
        if name == "shifted":
            print(f"    {'bin':>3s} {'ref p_k':>8s} {'prod q_k':>9s} {'q-p':>7s} {'ln(q/p)':>8s} {'contrib':>8s}")
            for k in range(10):
                print(f"    {k + 1:3d} {p[k]:8.3f} {q[k]:9.3f} {q[k] - p[k]:7.3f} {np.log(q[k] / p[k]):8.3f} {contrib[k]:8.4f}")
            print(f"    sum = {contrib.sum():.4f} -> {'stable' if contrib.sum() < 0.1 else 'moderate shift, investigate' if contrib.sum() < 0.25 else 'significant shift, act'}")
    p, q, contrib = psi_table(prod_shift)
    kl_qp = np.sum(q * np.log(q / p)); kl_pq = np.sum(p * np.log(p / q))
    print(f"  check: KL(q||p) + KL(p||q) = {kl_qp + kl_pq:.4f} = PSI")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(1, 11)
    axes[0].bar(x - 0.2, p, width=0.4, color=BLUE, edgecolor=SURFACE, label="reference (training time)")
    axes[0].bar(x + 0.2, q, width=0.4, color=ORANGE, edgecolor=SURFACE, label="production window")
    axes[0].set_xticks(x); axes[0].set_xlabel("decile bin of the reference"); axes[0].set_ylabel("fraction of samples")
    axes[0].set_title("Reference deciles vs production histogram")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, contrib, color=[BLUE if c >= 0 else BLUE for c in contrib], edgecolor=SURFACE)
    axes[1].set_xticks(x); axes[1].set_xlabel("decile bin"); axes[1].set_ylabel("(q_k - p_k) ln(q_k / p_k)")
    axes[1].set_title(f"Per-bin contributions, PSI = {contrib.sum():.3f}")
    axes[1].annotate(f"largest: bin {int(np.argmax(contrib)) + 1}", (np.argmax(contrib) + 1, contrib.max()), xytext=(-70, 2),
                     textcoords="offset points", fontsize=9, color=INK_2)
    save(fig, NB, "psi_drift_histograms")


# --------------------------------------------------------------------------------------
# Section 8: cost of serving vs load, batch 1 vs batch 64
# --------------------------------------------------------------------------------------
def fig_cost_vs_load():
    banner("Section 8: workers and monthly cost from the latency model (2x headroom, 0.05 USD per worker-hour)")
    lat_b1, lat_b64 = latency_model(1), latency_model(64)
    thr = {1: 1000.0 / lat_b1, 64: 64 * 1000.0 / lat_b64}
    print(f"  one worker: {thr[1]:,.0f} req/s at batch 1 ({lat_b1:.3f} ms) | {thr[64]:,.0f} req/s at batch 64 ({lat_b64:.3f} ms per batch)")
    print(f"  {'load req/s':>10s} {'batch':>5s} {'workers':>7s} {'USD/month':>9s}")
    rows = {}
    for load in (100, 2_000, 50_000):
        for bs in (1, 64):
            workers = math.ceil(2 * load / thr[bs])
            rows[(load, bs)] = (workers, workers * 0.05 * 24 * 30)
            print(f"  {load:10,d} {bs:5d} {workers:7d} {rows[(load, bs)][1]:9,.0f}")
    loads = np.logspace(2, 5.3, 60)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, bs in enumerate((1, 64)):
        cost = np.ceil(2 * loads / thr[bs]) * 0.05 * 24 * 30
        ax.plot(loads, cost, color=SERIES[k], label=f"batch {bs} ({thr[bs]:,.0f} req/s per worker)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("load (requests / s)"); ax.set_ylabel("monthly cost (USD, 0.05 USD per worker-hour)")
    ax.set_title("Serving cost vs load: batching cuts the bill at high load (model)")
    ax.legend(loc="upper left")
    w50k = rows[(50_000, 1)][0], rows[(50_000, 64)][0]
    ax.annotate(f"50,000 req/s: {w50k[0]} vs {w50k[1]} workers", (50_000, rows[(50_000, 64)][1]), xytext=(-230, 45),
                textcoords="offset points", fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    save(fig, NB, "cost_vs_load")


def main():
    fig_latency_percentiles()
    fig_latency_throughput_vs_batch()
    fig_bundle_size_breakdown()
    fig_export_verification()
    fig_quantization_error_vs_bits()
    fig_pruning()
    fig_soft_targets()
    fig_compression_menu()
    fig_batching_simulation()
    fig_psi_drift()
    fig_cost_vs_load()


if __name__ == "__main__":
    main()
