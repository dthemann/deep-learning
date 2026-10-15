"""Figures for the companion guide of Notebook 09 · Attention and Transformers.

Everything here is numpy + matplotlib. Attention, masks, sinusoidal and rotary encodings, the
parameter / FLOP formulas, the learning-rate schedule and the decoding filters are re-implemented
in numpy so that every number in the guide can be reproduced without PyTorch.

Run:  python3 notebooks/figures/scripts/make_figures_09.py
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

NB = "09"


# ----------------------------------------------------------------------------------------------
# numpy building blocks
# ----------------------------------------------------------------------------------------------
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention; mask is boolean, True = may attend. Returns (output, weights)."""
    scores = q @ np.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores)
    return w @ v, w


def heatmap(ax, M, title, xlabels=None, ylabels=None, vmin=0, vmax=1, cmap=SEQ_CMAP, annotate=False, fmt="{:.2f}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.grid(False)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                txt = "0" if (val == 0 and vmin == 0) else fmt.format(val)
                colour = SURFACE if (vmax - vmin) and (val - vmin) / (vmax - vmin) > 0.55 else INK
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=colour)
    return im


# ----------------------------------------------------------------------------------------------
# Section 1: a worked attention example with 3 tokens and d_k = 2
# ----------------------------------------------------------------------------------------------
def fig_attention_worked_example():
    tokens = ["the", "cat", "sat"]
    Q = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    raw = Q @ K.T
    scaled = raw / math.sqrt(2)
    W = softmax(scaled)
    out = W @ V
    print("\n[fig_attention_worked_example] tokens:", tokens)
    print("Q =\n", Q); print("K =\n", K); print("V =\n", V)
    print("raw scores Q K^T =\n", raw)
    print("scaled scores (/ sqrt 2) =\n", scaled.round(3))
    print("softmax weights =\n", W.round(3), "\nrow sums:", W.sum(1).round(6))
    print("output = weights @ V =\n", out.round(3))
    print("hard lookup check, query x4 for 'the':", softmax((4 * Q[0]) @ K.T / math.sqrt(2)).round(3))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    heatmap(axes[0], scaled, "scaled scores $QK^\\top/\\sqrt{2}$", tokens, tokens, vmin=0, vmax=scaled.max(), annotate=True)
    heatmap(axes[1], W, "softmax weights (rows sum to 1)", tokens, tokens, annotate=True)
    heatmap(axes[2], out, "output = weights @ V", ["dim 0", "dim 1"], tokens, vmin=0, vmax=1, annotate=True)
    for ax in axes[:2]:
        ax.set_xlabel("key (read from)")
    axes[0].set_ylabel("query (reading)")
    save(fig, NB, "attention_worked_example")


# ----------------------------------------------------------------------------------------------
# Section 1.3: variance of q.k with and without the 1/sqrt(d_k) scaling
# ----------------------------------------------------------------------------------------------
def fig_score_variance():
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_keys, n_trials = 16, 4000
    var_raw, var_scaled, max_raw, max_scaled = [], [], [], []
    print("\n[fig_score_variance] 16 random keys, 4000 trials per d_k")
    print(f"{'d_k':>5} {'var(raw)':>9} {'var(scaled)':>12} {'max w raw':>10} {'max w scaled':>13}")
    for d in dims:
        q = rng.standard_normal((n_trials, d))
        k = rng.standard_normal((n_trials, n_keys, d))
        raw = np.einsum("nd,nkd->nk", q, k)
        scaled = raw / math.sqrt(d)
        var_raw.append(raw.var()); var_scaled.append(scaled.var())
        max_raw.append(softmax(raw).max(-1).mean()); max_scaled.append(softmax(scaled).max(-1).mean())
        print(f"{d:5d} {var_raw[-1]:9.1f} {var_scaled[-1]:12.3f} {max_raw[-1]:10.3f} {max_scaled[-1]:13.3f}")
    print("std of raw scores at d_k=256:", round(math.sqrt(var_raw[dims.index(256)]), 2), "| exp(16) =", f"{math.exp(16):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(dims, var_raw, marker="o", color=SERIES[0], label="unscaled $q^\\top k$")
    ax.plot(dims, var_scaled, marker="s", color=SERIES[1], label="scaled $q^\\top k/\\sqrt{d_k}$")
    ax.plot(dims, dims, linestyle=":", color=MUTED, linewidth=1.2, label="reference: var $= d_k$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("$d_k$ (key dimension)"); ax.set_ylabel("variance of the scores")
    ax.set_title("Score variance grows like $d_k$ unless scaled"); ax.legend()
    ax = axes[1]
    ax.plot(dims, max_raw, marker="o", color=SERIES[0], label="unscaled")
    ax.plot(dims, max_scaled, marker="s", color=SERIES[1], label="scaled")
    ax.axhline(1 / n_keys, linestyle=":", color=MUTED, linewidth=1.2, label="uniform (1/16)")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.02)
    ax.set_xlabel("$d_k$ (key dimension)"); ax.set_ylabel("mean largest attention weight")
    ax.set_title("Softmax saturation (16 random keys)"); ax.legend(loc="center right")
    ax.annotate(f"{max_raw[-1]:.2f} at $d_k$=1024", (dims[-1], max_raw[-1]), xytext=(-70, -18), textcoords="offset points", fontsize=8, color=INK_2)
    save(fig, NB, "score_variance_vs_dk")


# ----------------------------------------------------------------------------------------------
# Section 2: padding and causal masks
# ----------------------------------------------------------------------------------------------
def fig_masks():
    T, length = 6, 4
    causal = np.tril(np.ones((T, T), dtype=bool))
    pad = np.arange(T) < length                       # keys that are real
    combined = causal & pad[None, :]
    x = rng.standard_normal((T, 8))
    _, w_pad = attention(x, x, x, pad[None, :])
    _, w_causal = attention(x, x, x, causal)
    _, w_both = attention(x, x, x, combined)
    print("\n[fig_masks] T = 6, real length = 4")
    print("padding mask (1 = may attend):", pad.astype(int))
    print("combined mask (rows = queries):\n", combined.astype(int))
    print("weights with both masks, rounded:\n", w_both.round(2))
    print("row sums:", w_both.sum(1).round(6), "| masked entries exactly zero:", bool((w_both[~combined] == 0).all()))
    print("count of allowed keys per query:", combined.sum(1))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    labels = [str(i) for i in range(T)]
    heatmap(axes[0], w_pad, "padding mask (length 4 of 6)", labels, labels, annotate=True)
    heatmap(axes[1], w_causal, "causal mask", labels, labels, annotate=True)
    heatmap(axes[2], w_both, "padding AND causal", labels, labels, annotate=True)
    for ax in axes:
        ax.set_xlabel("key position")
    axes[0].set_ylabel("query position")
    save(fig, NB, "padding_and_causal_masks")


# ----------------------------------------------------------------------------------------------
# Section 3: multi-head attention as a reshape, and the memory of the score matrix
# ----------------------------------------------------------------------------------------------
def fig_multihead():
    tokens = ["she", "gave", "him", "the", "book"]
    T, d, h = len(tokens), 8, 2
    d_h = d // h
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv = (rng.standard_normal((d, d)) / math.sqrt(d) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print("\n[fig_multihead] T =", T, "d =", d, "heads =", h, "d_head =", d_h)
    print("projection output shape (T, d):", Q.shape, "-> view (T, h, d_h):", Q.reshape(T, h, d_h).shape,
          "-> transpose (h, T, d_h):", Q.reshape(T, h, d_h).transpose(1, 0, 2).shape)
    Qh = Q.reshape(T, h, d_h).transpose(1, 0, 2)
    Kh = K.reshape(T, h, d_h).transpose(1, 0, 2)
    Vh = V.reshape(T, h, d_h).transpose(1, 0, 2)
    out, W = attention(Qh, Kh, Vh, np.tril(np.ones((T, T), dtype=bool)))
    merged = out.transpose(1, 0, 2).reshape(T, d)
    # loop check
    loop = np.concatenate([attention(Q[:, i * d_h:(i + 1) * d_h], K[:, i * d_h:(i + 1) * d_h], V[:, i * d_h:(i + 1) * d_h],
                                     np.tril(np.ones((T, T), dtype=bool)))[0] for i in range(h)], axis=-1)
    print("weights shape (h, T, T):", W.shape, "| merged output shape:", merged.shape, "| reshape == per-head loop:", np.allclose(merged, loop))
    print("head 0 row for 'book':", W[0, -1].round(2), "\nhead 1 row for 'book':", W[1, -1].round(2))
    print("parameters of MHA with d = 32: 4 (d^2 + d) =", 4 * (32 * 32 + 32))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for i in range(h):
        heatmap(axes[i], W[i], f"head {i} (causal, $d_h$ = {d_h})", tokens, tokens, annotate=True)
        axes[i].set_xlabel("key")
    axes[0].set_ylabel("query")
    save(fig, NB, "multihead_two_heads")


def fig_attention_memory():
    Ts = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    heads, d_h = 32, 128                              # a 4096-wide model with 32 heads
    scores_gib = heads * Ts.astype(float) ** 2 * 2 / 2 ** 30           # float16 T x T per head
    kv_gib = 2 * Ts.astype(float) * heads * d_h * 2 / 2 ** 30          # K and V for one layer, float16
    print("\n[fig_attention_memory] one layer, 32 heads, d_head 128, float16")
    for T, s, k in zip(Ts, scores_gib, kv_gib):
        print(f"T={T:7d}  score matrices {s:10.3f} GiB   K+V (FlashAttention keeps only these) {k:8.4f} GiB")
    print("4096 tokens, 32 heads, float32 score matrix:", 4096 ** 2 * 32 * 4 / 2 ** 30, "GiB")

    fig, ax = plt.subplots()
    ax.plot(Ts, scores_gib, marker="o", color=SERIES[0], label="materialised $T\\times T$ scores (quadratic)")
    ax.plot(Ts, kv_gib, marker="s", color=SERIES[1], label="keys and values only (linear)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length T (tokens)"); ax.set_ylabel("memory per layer and sequence (GiB, float16)")
    ax.set_title("Why FlashAttention never writes the score matrix")
    ax.annotate(f"{scores_gib[-1]:.0f} GiB at 128k", (Ts[-1], scores_gib[-1]), xytext=(-85, -4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.legend(loc="upper left")
    save(fig, NB, "attention_memory_vs_length")


# ----------------------------------------------------------------------------------------------
# Section 4: sinusoidal encodings and RoPE
# ----------------------------------------------------------------------------------------------
def sinusoidal(max_len, d, base=10000.0):
    pos = np.arange(max_len)[:, None].astype(float)
    i = np.arange(0, d, 2).astype(float)
    ang = pos / base ** (i / d)
    pe = np.zeros((max_len, d))
    pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
    return pe


def fig_sinusoidal():
    T, d = 64, 32
    pe = sinusoidal(T, d)
    small = sinusoidal(4, 4)
    print("\n[fig_sinusoidal] d = 4 toy table (rows = positions 0..3, columns = dims 0..3):\n", small.round(3))
    print("frequencies for d = 4: omega_0 = 1, omega_1 = 10000^(-2/4) =", 10000 ** (-2 / 4))
    omega = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    t0 = 32
    sims = pe @ pe[t0]
    print("d = 32: dot product p_32 . p_t for offsets 0, 1, 2, 5, 10:", [round(float(sims[t0 + k]), 3) for k in [0, 1, 2, 5, 10]])
    print("symmetric: p_32.p_37 =", round(float(sims[t0 + 5]), 4), " p_32.p_27 =", round(float(sims[t0 - 5]), 4))
    print("same for another anchor: p_10.p_15 =", round(float(pe[10] @ pe[15]), 4), "| formula sum_i cos(omega_i 5) =", round(float(np.cos(omega * 5).sum()), 4))
    # shift matrix check
    k = 7
    M = np.zeros((d, d))
    for j, w in enumerate(omega):
        c, s = math.cos(w * k), math.sin(w * k)
        M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[c, s], [-s, c]]
    print("pe[t+7] == M_7 pe[t] for all t:", np.allclose(pe[:-k] @ M.T, pe[k:]))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    im = axes[0].imshow(pe.T, aspect="auto", cmap=DIV_CMAP, vmin=-1, vmax=1, origin="lower")
    axes[0].set_xlabel("position t"); axes[0].set_ylabel("encoding dimension"); axes[0].grid(False)
    axes[0].set_title("Sinusoidal encoding (d = 32)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    axes[1].plot(np.arange(T) - t0, sims, color=SERIES[0])
    axes[1].axvline(0, color=MUTED, linestyle=":", linewidth=1)
    axes[1].set_xlabel("offset k = t - 32"); axes[1].set_ylabel("$p_{32}^\\top p_t$")
    axes[1].set_title("Similarity depends on the offset only")
    axes[1].annotate(f"peak {sims[t0]:.0f} = d/2 at k = 0", (0, sims[t0]), xytext=(8, -4), textcoords="offset points", fontsize=8, color=INK_2)
    save(fig, NB, "sinusoidal_encoding")


def rope_rotate(x, theta):
    """Rotate the pairs (x[0::2], x[1::2]) by the angles theta (one per pair)."""
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(theta) - x2 * np.sin(theta)
    out[1::2] = x1 * np.sin(theta) + x2 * np.cos(theta)
    return out


def fig_rope():
    d = 4
    inv_freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)       # [1, 0.01]
    q = np.array([1.0, 0.0, 1.0, 0.0])
    k = np.array([0.8, 0.6, 0.0, 1.0])
    print("\n[fig_rope] d = 4, frequencies:", inv_freq, "q =", q, "k =", k)
    positions = np.arange(8)
    rotated = np.array([rope_rotate(q, m * inv_freq) for m in positions])
    print("pair 0 of q (frequency 1) at positions 0..7:\n", rotated[:, :2].round(3))
    print("norm of rotated q at each position:", np.linalg.norm(rotated, axis=1).round(6))

    def score(m, n):
        return float(rope_rotate(q, m * inv_freq) @ rope_rotate(k, n * inv_freq))
    pairs = [(3, 0), (5, 2), (13, 10), (40, 37)]
    print("offset 3 scores:", [(m, n, round(score(m, n), 4)) for m, n in pairs])
    print("offset 2 scores:", [(m, n, round(score(m, n), 4)) for m, n in [(2, 0), (5, 3), (40, 38)]])
    print("raw q.k without RoPE:", float(q @ k))
    ms = np.arange(3, 60)
    s3 = [score(m, m - 3) for m in ms]
    s2 = [score(m, m - 2) for m in ms]
    s0 = [score(m, m) for m in ms]
    print("spread of score(m, m-3) over m = 3..59:", round(max(s3) - min(s3), 12))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color=GRID, linewidth=1)
    for m in positions:
        x, y = rotated[m, 0], rotated[m, 1]
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SEQ_CMAP(0.2 + 0.8 * m / 7), lw=1.6))
        ax.text(1.12 * x, 1.12 * y, f"t={m}", ha="center", va="center", fontsize=8, color=INK_2)
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_title("RoPE: pair 0 of q rotates by t radians"); ax.set_xlabel("$q_0$"); ax.set_ylabel("$q_1$")
    ax = axes[1]
    ax.plot(ms, s0, color=SERIES[0], label="offset 0 (key at m)")
    ax.plot(ms, s2, color=SERIES[1], label="offset 2 (key at m - 2)")
    ax.plot(ms, s3, color=SERIES[2], label="offset 3 (key at m - 3)")
    ax.set_xlabel("absolute query position m"); ax.set_ylabel("score $(R_m q)^\\top(R_n k)$")
    ax.set_title("Score depends on the offset, not on m"); ax.legend(loc="center right")
    save(fig, NB, "rope_rotation_and_scores")


# ----------------------------------------------------------------------------------------------
# Section 5: parameters and FLOPs
# ----------------------------------------------------------------------------------------------
def block_params(d):
    return 12 * d * d + 13 * d


def gpt_params(L, d, V, T_max):
    return L * block_params(d) + V * d + T_max * d + 2 * d


def flops_forward(L, d, V, T):
    return L * (24 * d * d + 4 * T * d) + 2 * V * d


def fig_block_params():
    d = 64
    comps = {"attention W_Q, W_K, W_V, W_O": 4 * d * d + 4 * d, "MLP W_1, W_2": 8 * d * d + 5 * d, "two LayerNorms": 4 * d}
    print("\n[fig_block_params] block with d = 64:", comps, "| total", sum(comps.values()), "| formula 12d^2+13d =", block_params(d))
    print("MLP share of a block:", round(comps["MLP W_1, W_2"] / sum(comps.values()), 3))
    family = [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024), ("GPT-2 large", 36, 1280), ("GPT-2 XL", 48, 1600)]
    V, T = 50257, 1024
    blocks, embeds, totals = [], [], []
    for name, L, dm in family:
        b = L * block_params(dm); e = V * dm + T * dm + 2 * dm
        blocks.append(b / 1e6); embeds.append(e / 1e6); totals.append((b + e) / 1e6)
        print(f"{name:>13}: L={L:2d} d={dm:4d} blocks {b/1e6:7.1f}M + embeddings/final LN {e/1e6:5.1f}M = {(b+e)/1e6:7.1f}M")
    print("GPT-2 small exact:", gpt_params(12, 768, V, T), "(published 124,439,808)")
    print("notebook FAST GPT (L=2, d=64, V=65, T=64):", gpt_params(2, 64, 65, 64), "| full (L=6, d=256, V=65, T=256):", gpt_params(6, 256, 65, 256))

    fig, ax = plt.subplots()
    names = [f[0] for f in family]
    ax.bar(names, blocks, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, label="Transformer blocks: $L(12d^2+13d)$")
    ax.bar(names, embeds, width=0.6, bottom=blocks, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, label="token + position embeddings")
    for i, t in enumerate(totals):
        ax.annotate(f"{t:,.0f}M", (i, t), ha="center", va="bottom", fontsize=8.5, color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_ylabel("parameters (millions)"); ax.set_title("Parameter count of the GPT-2 family by the notebook's formula")
    ax.legend(loc="upper left")
    save(fig, NB, "gpt2_parameter_count")


def fig_flops_vs_length():
    L, d, V = 12, 768, 50257
    Ts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    weights = np.full(len(Ts), (L * 24 * d * d + 2 * V * d) / 1e9)
    attn = L * 4 * Ts * d / 1e9
    share = attn / (weights + attn)
    print("\n[fig_flops_vs_length] GPT-2 small, forward FLOPs per token")
    print(f"weight matrices (constant in T): {weights[0]:.3f} GFLOP/token")
    for T, a, s in zip(Ts, attn, share):
        print(f"T={T:7d}  attention products {a:8.3f} GFLOP/token  share {s:6.1%}")
    print("ratio attention/weights = T/(6d) at T=1024:", 1024 / (6 * 768), "| exact (no embedding term):", (4 * 1024 * d) / (24 * d * d))
    N = gpt_params(L, d, V, 1024)
    print("6N per token =", 6 * N / 1e9, "GFLOP | 3 x forward at T=1024 =", 3 * flops_forward(L, d, V, 1024) / 1e9, "GFLOP")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(Ts, weights, marker="s", color=SERIES[0], label="weight matrices ($24d^2$ per block + head)")
    ax.plot(Ts, attn, marker="o", color=SERIES[1], label="attention products ($4Td$ per block)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length T (tokens)"); ax.set_ylabel("forward GFLOP per token")
    ax.set_title("GPT-2 small: cost per token vs context"); ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(Ts, 100 * share, marker="o", color=SERIES[1])
    ax.set_xscale("log", base=2); ax.set_ylim(0, 100)
    ax.set_xlabel("context length T (tokens)"); ax.set_ylabel("attention share of forward FLOPs (%)")
    ax.set_title("Attention share: 13% at 1k, 95% at 128k")
    ax.annotate(f"{100*share[2]:.0f}%", (Ts[2], 100 * share[2]), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.annotate(f"{100*share[-1]:.0f}%", (Ts[-1], 100 * share[-1]), xytext=(-24, -14), textcoords="offset points", fontsize=8.5, color=INK_2)
    save(fig, NB, "flops_attention_vs_weights")


# ----------------------------------------------------------------------------------------------
# Section 6: data windows and the learning-rate schedule
# ----------------------------------------------------------------------------------------------
def lr_at(step, max_lr, warmup, total_steps, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def fig_lr_schedule():
    text = "To be, or not to be"
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]
    print("\n[fig_lr_schedule] toy corpus:", repr(text), "| vocabulary", len(chars), chars)
    print("ids:", ids)
    block = 5
    for start in [0, 7]:
        print(f"window at {start}: x = {repr(text[start:start+block])} -> y = {repr(text[start+1:start+block+1])} | ids x {ids[start:start+block]} y {ids[start+1:start+block+1]}")
    print("uniform-guess loss for V = 65: ln 65 =", round(math.log(65), 3))
    max_lr, warmup, steps, min_lr = 3e-3, 50, 500, 3e-4
    lrs = np.array([lr_at(s, max_lr, warmup, steps, min_lr) for s in range(steps)])
    for s in [0, 24, 49, 50, 162, 275, 387, 499]:
        print(f"lr at step {s:3d}: {lrs[s]:.2e}")
    print("peak", lrs.max(), "at step", int(lrs.argmax()), "| final", f"{lrs[-1]:.2e}", "| min_lr/max_lr =", min_lr / max_lr)

    fig, ax = plt.subplots()
    ax.plot(np.arange(steps), lrs * 1e3, color=SERIES[0])
    ax.axvline(warmup, color=MUTED, linestyle=":", linewidth=1)
    ax.text(warmup + 6, 0.12, "warmup ends at step 50\n(peak 3.0e-3)", fontsize=8.5, color=INK_2, va="bottom")
    ax.annotate(f"cosine floor {min_lr*1e3:.1f}e-3 at step 499", (steps - 1, min_lr * 1e3), xytext=(-235, -12), textcoords="offset points", fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate (x 1e-3)")
    ax.set_title("Linear warmup then cosine decay (FAST_RUN settings)")
    save(fig, NB, "lr_warmup_cosine")


# ----------------------------------------------------------------------------------------------
# Section 7: decoding filters and the KV-cache
# ----------------------------------------------------------------------------------------------
def fig_decoding():
    tokens = ["the", "a", "his", "my", "an", "one", "this", "zzz"]
    p = np.array([0.40, 0.25, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02])
    logits = np.log(p)
    print("\n[fig_decoding] toy next-token distribution:", {t: float(v) for t, v in zip(tokens, p)})
    t_cool, t_hot = softmax(logits / 0.5), softmax(logits / 2.0)
    print("temperature 0.5:", t_cool.round(3)); print("temperature 2.0:", t_hot.round(3))
    k = 3
    keep_k = np.zeros(8, bool); keep_k[np.argsort(-p)[:k]] = True
    p_k = np.where(keep_k, p, 0); p_k = p_k / p_k.sum()
    print("top-k (k=3) kept:", [t for t, m in zip(tokens, keep_k) if m], "renormalised:", p_k[keep_k].round(3))
    cum = np.cumsum(p)
    remove = (cum - p) > 0.8
    keep_p = ~remove
    p_p = np.where(keep_p, p, 0); p_p = p_p / p_p.sum()
    print("top-p (p=0.8): cumulative", cum.round(2), "| mass before each token", (cum - p).round(2))
    print("top-p kept:", [t for t, m in zip(tokens, keep_p) if m], "renormalised:", p_p[keep_p].round(3))
    print("greedy picks:", tokens[int(np.argmax(p))])
    print("naive rule 'cumulative > p' on a 0.95 top token with p = 0.9 keeps:", int((np.cumsum([0.95, 0.03, 0.02]) <= 0.9).sum()), "tokens")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey=True)
    panels = [("model probabilities (temperature 1)", p, np.ones(8, bool)),
              ("temperature 0.5 (sharper)", t_cool, np.ones(8, bool)),
              ("top-k, k = 3 (renormalised)", p_k, keep_k),
              ("top-p, p = 0.8 (renormalised)", p_p, keep_p)]
    for ax, (title, probs, keep) in zip(axes.ravel(), panels):
        colours = [SERIES[0] if m else AXIS for m in keep]
        ax.bar(tokens, np.where(keep, probs, p), width=0.6, color=colours, edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(title); ax.set_ylim(0, 0.9)
        for i, (pr, m) in enumerate(zip(probs, keep)):
            if m and pr >= 0.2:
                ax.annotate(f"{pr:.2f}", (i, pr), ha="center", va="bottom", fontsize=8, color=INK_2, xytext=(0, 2), textcoords="offset points")
    axes[0, 0].set_ylabel("probability"); axes[1, 0].set_ylabel("probability")
    fig.suptitle("Decoding filters on a toy distribution (grey bars = removed, original height)", fontsize=11, fontweight="bold")
    save(fig, NB, "decoding_top_k_top_p")


def fig_kv_cache():
    T0, n = 8, 56
    steps = np.arange(n)
    naive = T0 + steps
    cached = np.where(steps == 0, T0, 1)
    print("\n[fig_kv_cache] prompt T0 =", T0, "new tokens n =", n)
    print("tokens processed per step, naive, first 5:", naive[:5], "... last:", naive[-1])
    print("tokens processed per step, cached, first 5:", cached[:5], "... last:", cached[-1])
    print("totals: naive", int(naive.sum()), "= n*T0 + n(n-1)/2 =", n * T0 + n * (n - 1) // 2, "| cached", int(cached.sum()), "= T0 + n - 1 | ratio", round(naive.sum() / cached.sum(), 1))
    L, h, d_h = 6, 8, 32
    print("notebook full model cache for", T0 + n, "tokens, float32:", 2 * L * h * (T0 + n) * d_h * 4 / 1024, "KiB")
    print("70B model (L=80, d=8192) cache at 8192 tokens, float16:", 2 * 80 * 8192 * 8192 * 2 / 2 ** 30, "GiB | weights fp16:", 70e9 * 2 / 1e9, "GB")
    print("Llama-2-70B-like at 4096 tokens: MHA", 2 * 80 * 4096 * 8192 * 2 / 2 ** 30, "GiB | GQA with 8 of 64 heads", 2 * 80 * 4096 * 8192 * 2 / 8 / 2 ** 30, "GiB")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(steps, naive, color=SERIES[0], marker="o", markevery=8, label="naive: whole context each step")
    ax.plot(steps, cached, color=SERIES[1], marker="s", markevery=8, label="KV-cache: prefill, then 1 token")
    ax.set_xlabel("generation step"); ax.set_ylabel("token-positions processed in the step")
    ax.set_title("Work per generated token"); ax.legend(loc="center left")
    ax = axes[1]
    ax.plot(steps, np.cumsum(naive), color=SERIES[0], label="naive (quadratic)")
    ax.plot(steps, np.cumsum(cached), color=SERIES[1], label="KV-cache (linear)")
    ax.set_xlabel("generation step"); ax.set_ylabel("cumulative token-positions")
    ax.set_title(f"Total work: {int(naive.sum()):,} vs {int(cached.sum())}"); ax.legend(loc="upper left")
    save(fig, NB, "kv_cache_tokens_processed")


# ----------------------------------------------------------------------------------------------
# Section 8: scaling
# ----------------------------------------------------------------------------------------------
def fig_scaling():
    configs = [("notebook full", 6, 256, 65, 256), ("GPT-2 small", 12, 768, 50257, 1024), ("GPT-2 XL", 48, 1600, 50257, 1024),
               ("GPT-3", 96, 12288, 50257, 2048), ("Chinchilla-like", 80, 8192, 32000, 2048)]
    Ns, Cs = [], []
    print("\n[fig_scaling] compute-optimal budgets, 20 tokens per parameter, C = 6 N D, 1 PFLOP/s at 30% utilisation")
    for name, L, d, V, T in configs:
        N = gpt_params(L, d, V, T); D = 20 * N; C = 6 * N * D; days = C / (1e15 * 0.3) / 86400
        Ns.append(N); Cs.append(C)
        print(f"{name:>15}: N = {N/1e9:9.4f} B | D = {D/1e9:9.1f} B tokens | C = {C:.2e} FLOPs | {days:10.1f} days")
    print("x10 parameters -> compute factor:", (10 ** 2), "(C = 120 N^2)")
    alpha, Nc = 0.076, 8.8e13
    for N in [1e7, 1e8, 1e9, 1e10, 1e11]:
        print(f"Kaplan law L(N) = (Nc/N)^0.076 at N = {N:.0e}: {(Nc / N) ** alpha:.3f} nats")
    print("GPT-3 tokens per parameter: 300e9 / 175e9 =", round(300 / 175, 2), "| Chinchilla: 1.4e12 / 70e9 =", 1.4e12 / 70e9)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    Ns_a, Cs_a = np.array(Ns), np.array(Cs)
    ax.plot(Ns_a, Cs_a, color=SERIES[0], marker="o", linewidth=1.5)
    for (name, *_), N, C in zip(configs, Ns, Cs):
        ax.annotate(name, (N, C), xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters N"); ax.set_ylabel("training FLOPs C = 6 N (20 N)")
    ax.set_title("Compute-optimal cost grows as $N^2$")
    ax = axes[1]
    Ngrid = np.logspace(6, 12, 200)
    ax.plot(Ngrid, (Nc / Ngrid) ** alpha, color=SERIES[0])
    ax.set_xscale("log")
    ax.set_xlabel("parameters N (data and compute not limiting)"); ax.set_ylabel("test loss (nats per token)")
    ax.set_title("Kaplan power law $L(N) = (N_c/N)^{0.076}$")
    for N in [1e8, 1e10]:
        ax.annotate(f"{(Nc / N) ** alpha:.2f}", (N, (Nc / N) ** alpha), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK_2)
        ax.plot([N], [(Nc / N) ** alpha], marker="o", color=SERIES[0])
    save(fig, NB, "scaling_compute_and_power_law")


# ----------------------------------------------------------------------------------------------
# Section 9: the three attention patterns and BERT masking
# ----------------------------------------------------------------------------------------------
def fig_family():
    T_src, T_tgt = 6, 6
    enc = np.ones((T_src, T_src), bool)
    dec = np.tril(np.ones((T_tgt, T_tgt), bool))
    cross = np.ones((T_tgt, T_src), bool)
    print("\n[fig_family] visible keys per query: encoder", enc.sum(1), "| decoder (causal)", dec.sum(1), "| cross-attention", cross.sum(1))
    text = "the quick brown fox jumps over the lazy dog, and the cat sleeps."
    r = np.random.default_rng(31)
    selected = r.random(len(text)) < 0.15
    u = r.random(len(text))
    corrupted = list(text)
    n_mask = n_rand = n_keep = 0
    for i in np.where(selected)[0]:
        if u[i] < 0.8:
            corrupted[i] = "_"; n_mask += 1
        elif u[i] < 0.9:
            corrupted[i] = chr(97 + r.integers(26)); n_rand += 1
        else:
            n_keep += 1
    print("BERT masking on", len(text), "characters: expected 15% =", round(0.15 * len(text), 1), "selected; got", int(selected.sum()),
          f"(masked {n_mask}, random {n_rand}, kept {n_keep})")
    print("original :", repr(text)); print("corrupted:", repr("".join(corrupted)))
    print("nn.TransformerEncoderLayer(d=64, nhead=4, dim_feedforward=256) parameters:",
          (3 * 64 * 64 + 3 * 64) + (64 * 64 + 64) + (64 * 256 + 256) + (256 * 64 + 64) + 2 * 2 * 64, "= block formula", block_params(64), "| two layers:", 2 * block_params(64))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    heatmap(axes[0], enc.astype(float), "encoder (BERT): every key visible", [str(i) for i in range(T_src)], [str(i) for i in range(T_src)])
    heatmap(axes[1], dec.astype(float), "decoder (GPT): causal", [str(i) for i in range(T_tgt)], [str(i) for i in range(T_tgt)])
    heatmap(axes[2], cross.astype(float), "cross-attention (T5 decoder)", [f"s{i}" for i in range(T_src)], [f"t{i}" for i in range(T_tgt)])
    axes[0].set_xlabel("key (dark = may attend)"); axes[1].set_xlabel("key (dark = may attend)"); axes[2].set_xlabel("encoder key (source token)")
    axes[0].set_ylabel("query"); axes[2].set_ylabel("decoder query (target token)")
    save(fig, NB, "family_attention_patterns")


def main():
    fig_attention_worked_example()
    fig_score_variance()
    fig_masks()
    fig_multihead()
    fig_attention_memory()
    fig_sinusoidal()
    fig_rope()
    fig_block_params()
    fig_flops_vs_length()
    fig_lr_schedule()
    fig_decoding()
    fig_kv_cache()
    fig_scaling()
    fig_family()


if __name__ == "__main__":
    main()
