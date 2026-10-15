"""Figures for the companion guide of notebook 14 (efficient training and scaling).

numpy + matplotlib only. Each fig_* function builds its data, prints every number quoted in the
guide (clearly labelled), and saves one PNG under notebooks/figures/14/. Charts that show a model or
a toy simulation rather than a measurement say so in their title.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

NB = "14"
MB, GB = 2 ** 20, 2 ** 30


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# Section 1: timing a GPU correctly (model) and Amdahl's law for a training step
# --------------------------------------------------------------------------------------
def fig_timing_and_amdahl():
    banner("Section 1: naive vs synchronized timing (model) and Amdahl's law for one step")
    # (a) a model of a GPU: 10 TFLOP/s sustained in float32, kernel launch 20 us
    peak_flops, launch_us = 10e12, 20.0
    sizes = np.array([256, 512, 1024, 2048, 4096, 8192])
    compute_ms = 2 * sizes.astype(float) ** 3 / peak_flops * 1e3
    naive_ms = np.full_like(compute_ms, launch_us / 1e3)
    print("  model GPU: 10 TFLOP/s sustained, 20 us per kernel launch")
    print(f"  {'n':>6s} {'GFLOP':>9s} {'synchronized ms':>16s} {'naive ms':>9s} {'ratio':>8s}")
    for n, c, nv in zip(sizes, compute_ms, naive_ms):
        print(f"  {n:6d} {2 * n ** 3 / 1e9:9.2f} {c:16.3f} {nv:9.3f} {c / nv:8.1f}")

    # (b) Amdahl: a hypothetical 10 ms step
    stages = ["data loading", "host-to-device copy", "forward", "backward", "optimizer step"]
    ms = np.array([4.0, 0.5, 2.0, 3.0, 0.5])
    total = ms.sum()
    print(f"\n  hypothetical step: {dict(zip(stages, ms))} -> total {total:.1f} ms")
    for i, s in enumerate(stages):
        new = ms.copy(); new[i] = ms[i] / 10
        print(f"  make '{s}' 10x faster: step {new.sum():.2f} ms, speed-up {total / new.sum():.2f}x "
              f"(stage share {100 * ms[i] / total:.0f} %, Amdahl limit {total / (total - ms[i]):.2f}x)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(sizes, compute_ms, marker="o", color=SERIES[0], label="synchronized (real duration)")
    ax.plot(sizes, naive_ms, marker="s", color=SERIES[1], label="naive timer (launch only)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("matrix size n (n x n matmul)"); ax.set_ylabel("measured time (ms)")
    ax.set_title("Model of GPU timing: the naive timer never moves")
    ax.annotate(f"{compute_ms[-1]:.0f} ms", (sizes[-1], compute_ms[-1]), xytext=(-40, -14),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.legend(loc="upper left")
    ax = axes[1]
    left = 0.0
    for s, v, c in zip(stages, ms, SERIES[:5]):
        ax.barh(["step"], [v], left=left, color=c, edgecolor=SURFACE, linewidth=1.5, label=f"{s} ({v:.1f} ms)")
        left += v
    ax.set_xlim(0, total); ax.set_xlabel("time within one training step (ms)")
    ax.set_title("Hypothetical 10 ms step: only the big stage is worth optimizing")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2)
    ax.set_yticks([])
    save(fig, NB, "timing_and_amdahl")


# --------------------------------------------------------------------------------------
# Section 2: memory arithmetic for GPT-style models, fp32 vs mixed precision
# --------------------------------------------------------------------------------------
def transformer_params(L, d, V, s):
    return L * (12 * d ** 2 + 13 * d) + V * d + s * d


def transformer_activation_bytes(L, d, a, s, b, bytes_per_elem=2):
    return L * s * b * d * (34 + 5 * a * s / d) * bytes_per_elem / 2


def fig_memory_breakdown():
    banner("Section 2: training memory of three GPT-style models (batch 1 x 2048 tokens)")
    V, s, b = 50257, 2048, 1
    configs = [("125M", 12, 768, 12), ("1.3B", 24, 2048, 24), ("7B", 32, 4096, 32)]
    rows = []
    for name, L, d, a in configs:
        P = transformer_params(L, d, V, s)
        fp32 = {"weights": 4 * P, "gradients": 4 * P, "Adam states": 8 * P,
                "activations": transformer_activation_bytes(L, d, a, s, b, 4)}
        mixed = {"weights": 2 * P + 4 * P, "gradients": 2 * P, "Adam states": 8 * P,
                 "activations": transformer_activation_bytes(L, d, a, s, b, 2)}
        rows.append((name, P, fp32, mixed))
        print(f"  {name}: L={L} d={d} heads={a} -> P = {P / 1e6:,.0f}M parameters")
        for lab, dct in (("fp32", fp32), ("mixed", mixed)):
            parts = "  ".join(f"{k} {v / GB:6.2f}" for k, v in dct.items())
            print(f"     {lab:5s}: {parts}  | total {sum(dct.values()) / GB:7.2f} GiB "
                  f"(state {sum(v for k, v in dct.items() if k != 'activations') / P:.0f} B/param)")
    # the notebook's small models, from the leaf-output rule of thumb
    mlp_P = 784 * 256 + 256 + 256 * 256 + 256 + 256 * 10 + 10
    cnn_P = 9 * 32 + 32 + 64 + 9 * 32 * 64 + 64 + 128 + 3136 * 128 + 128 + 1290
    mlp_act, cnn_act = 256 * 6 + 10, 25088 * 3 + 6272 + 12544 * 3 + 3136 * 2 + 128 * 2 + 10
    print(f"\n  notebook models: MLP P={mlp_P:,} act/sample={mlp_act:,} | SmallCNN P={cnn_P:,} act/sample={cnn_act:,}")
    for B in (32, 128, 512):
        print(f"   B={B:3d}: MLP state {mlp_P * 16 / MB:5.1f} MB, activations {mlp_act * B * 4 / MB:5.1f} MB | "
              f"CNN state {cnn_P * 16 / MB:5.1f} MB, activations {cnn_act * B * 4 / MB:6.1f} MB")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    keys = ["weights", "gradients", "Adam states", "activations"]
    x = np.arange(len(rows))
    w = 0.36
    for j, (lab, off) in enumerate((("fp32", -w / 2), ("mixed", w / 2))):
        bottom = np.zeros(len(rows))
        for k, c in zip(keys, SERIES[:4]):
            vals = np.array([(r[2] if lab == "fp32" else r[3])[k] / GB for r in rows])
            ax.bar(x + off, vals, width=w, bottom=bottom, color=c, edgecolor=SURFACE, linewidth=1.0,
                   label=k if j == 0 else None)
            bottom += vals
        for xi, tot in zip(x + off, bottom):
            ax.annotate(f"{tot:.0f}", (xi, tot), ha="center", va="bottom", fontsize=8, color=INK_2,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\nfp32 | mixed" for r in rows])
    ax.set_ylabel("memory (GiB)"); ax.set_title("Training memory: state is 16 bytes/param either way, activations halve")
    ax.legend(loc="upper left")
    save(fig, NB, "memory_breakdown_models")


# --------------------------------------------------------------------------------------
# Section 3: float formats on a number line, and loss scaling against fp16 underflow
# --------------------------------------------------------------------------------------
def fig_float_formats():
    banner("Section 3.1: float32 / float16 / bfloat16 constants")
    fmts = {
        "float32": dict(exp=8, mant=23), "float16": dict(exp=5, mant=10), "bfloat16": dict(exp=8, mant=7)}
    table = {}
    for name, f in fmts.items():
        emax = 2 ** (f["exp"] - 1) - 1
        eps = 2.0 ** -f["mant"]
        vmax = (2 - 2.0 ** -f["mant"]) * 2.0 ** emax
        tiny = 2.0 ** (1 - emax)
        sub = tiny * 2.0 ** -f["mant"]
        table[name] = dict(eps=eps, max=vmax, tiny=tiny, sub=sub)
        print(f"  {name:9s} exp {f['exp']} mant {f['mant']} | eps {eps:.2e} | max {vmax:.3e} | "
              f"smallest normal {tiny:.2e} | smallest subnormal {sub:.2e}")
    for name, dt in (("float32", np.float32), ("float16", np.float16)):
        fi = np.finfo(dt)
        print(f"  numpy check {name}: eps {fi.eps:.2e} max {fi.max:.3e} tiny {fi.tiny:.2e}")
    # rounding examples
    print(f"  fp16: 1.0003 -> {float(np.float16(1.0003))}, 1.003 -> {float(np.float16(1.003))}, "
          f"70000 -> {float(np.float16(70000.0))}")
    def to_bf16(x):
        m, e = np.frexp(np.float64(x))
        return float(np.ldexp(np.round(m * 2 ** 8) / 2 ** 8, e))
    print(f"  bf16: 1.003 -> {to_bf16(1.003)}, 1.0078125 -> {to_bf16(1.0078125)}, 65504 -> {to_bf16(65504.0)}")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = list(fmts.keys())
    for i, (name, c) in enumerate(zip(names, SERIES[:3])):
        t = table[name]
        ax.plot([t["sub"], t["max"]], [i, i], color=c, lw=6, alpha=0.25, solid_capstyle="butt")
        ax.plot([t["tiny"], t["max"]], [i, i], color=c, lw=6, solid_capstyle="butt", label=name)
        ax.plot([t["eps"]], [i], marker="D", ms=7, color=INK, zorder=5)
        ax.annotate(f"eps {t['eps']:.1e}", (t["eps"], i), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        ax.annotate(f"max {t['max']:.3g}", (t["max"], i), xytext=(0, -14), textcoords="offset points",
                    ha="right", fontsize=8, color=INK_2)
        ax.annotate(f"min normal {t['tiny']:.1e}", (t["tiny"], i), xytext=(0, -14), textcoords="offset points",
                    ha="left", fontsize=8, color=INK_2)
    ax.set_xscale("log"); ax.set_xlim(1e-46, 1e40)
    ax.set_yticks(range(3)); ax.set_yticklabels(names); ax.set_ylim(-0.8, 2.8)
    ax.set_xlabel("magnitude (log scale); faint = subnormal range, diamond = machine epsilon")
    ax.set_title("Three float formats: range (bar) versus precision (epsilon)")
    ax.legend(loc="upper left", ncol=3)
    save(fig, NB, "float_formats_number_line")


def fig_loss_scaling():
    banner("Section 3.2: a toy gradient distribution against the fp16 thresholds, before and after scaling")
    g = np.exp(rng.normal(np.log(1e-6), 2.0, size=200_000))          # log-normal gradient magnitudes
    zero_thr, sub_thr, max_thr = 2.0 ** -24, 2.0 ** -14, 65504.0
    S = 2.0 ** 16
    def report(label, x):
        print(f"  {label:26s} median {np.median(x):.2e} | flushed to zero (<{zero_thr:.1e}): {100 * np.mean(x < zero_thr):5.2f} % "
              f"| subnormal: {100 * np.mean((x >= zero_thr) & (x < sub_thr)):5.2f} % | overflow (>{max_thr:.0f}): {100 * np.mean(x > max_thr):.3f} %")
    report("unscaled gradients", g)
    report(f"scaled by S = 2^16", g * S)
    print(f"  example: 1e-8 * 2^16 = {1e-8 * S:.3e}")

    bins = np.logspace(-12, 6, 73)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(g, bins=bins, color=SERIES[0], alpha=0.85, label="gradient magnitudes (toy, log-normal)")
    ax.hist(g * S, bins=bins, color=SERIES[1], alpha=0.75, label="after multiplying the loss by S = 65 536")
    for v, lab in ((zero_thr, "fp16 flush-to-zero 6e-8"), (sub_thr, "fp16 smallest normal 6e-5"), (max_thr, "fp16 max 65 504")):
        ax.axvline(v, color=INK_2, lw=0.9, ls="--")
        ax.annotate(lab, (v, ax.get_ylim()[1] * 0.97), rotation=90, va="top", ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log"); ax.set_xlabel("gradient magnitude"); ax.set_ylabel("count")
    ax.set_title("Loss scaling shifts the whole gradient distribution into the fp16 range (toy data)")
    ax.legend(loc="upper left")
    save(fig, NB, "loss_scaling_histogram")


# --------------------------------------------------------------------------------------
# Section 4: gradient accumulation equivalence and the checkpointing trade-off
# --------------------------------------------------------------------------------------
def fig_grad_accumulation():
    banner("Section 4.1: per-sample gradients, micro-batch means and the big-batch mean")
    g = np.array([0.8, -0.2, 0.5, 0.1, -0.4, 0.6, 0.3, -0.1])      # per-sample gradients of one weight
    K, b = 2, 4
    micro = g.reshape(K, b).mean(axis=1)
    print("  per-sample gradients:", g.tolist())
    print(f"  micro-batch means (b={b}): {micro.tolist()} -> mean of means {micro.mean():.3f}")
    print(f"  big-batch mean over all {len(g)} samples: {g.mean():.3f}")
    print(f"  accumulate loss/K: {np.sum(micro / K):.3f} (correct) | forgot the 1/K: {np.sum(micro):.3f} = {np.sum(micro) / g.mean():.0f}x too large")
    print(f"  SGD, lr 0.05: correct step {-0.05 * g.mean():.4f}, buggy step {-0.05 * micro.sum():.4f} (= lr x K)")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(g))
    colors = [SERIES[0]] * b + [SERIES[1]] * b
    ax.bar(x, g, color=colors, edgecolor=SURFACE, linewidth=1.5)
    for k in range(K):
        ax.hlines(micro[k], k * b - 0.4, k * b + b - 1 + 0.4, color=SERIES[k], lw=2.5,
                  label=f"micro-batch {k + 1} mean = {micro[k]:.2f}")
    ax.axhline(g.mean(), color=INK, lw=1.5, ls="--", label=f"big-batch mean = {g.mean():.2f} = mean of the two means")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"s{i + 1}" for i in x])
    ax.set_xlabel("sample (two micro-batches of 4)"); ax.set_ylabel("gradient of one weight")
    ax.set_title("Gradient accumulation: averaging micro-batch means gives the big-batch mean")
    ax.legend(loc="upper right")
    save(fig, NB, "grad_accumulation_equivalence")


def fig_checkpointing_tradeoff():
    banner("Section 4.2: activation memory vs number of checkpoint segments (L = 64 layers)")
    L = 64
    ks = np.array([1, 2, 4, 8, 16, 32, 64])
    mem = ks + L / ks            # segment inputs kept + the inside of the one segment being recomputed
    print(f"  no checkpointing: {L} layer activations kept, compute 3.0 units (fwd 1 + bwd 2)")
    print(f"  {'segments k':>10s} {'kept = k + L/k':>15s} {'reduction':>10s} {'compute':>8s}")
    for k, m in zip(ks, mem):
        comp = 3.0 if k == L else 4.0
        print(f"  {k:10d} {m:15.1f} {L / m:9.1f}x {comp:8.1f}")
    kopt = np.sqrt(L)
    print(f"  optimum k = sqrt(L) = {kopt:.0f}: {2 * kopt:.0f} layer activations instead of {L} ({L / (2 * kopt):.0f}x less), +{100 / 3:.0f} % compute")
    # the notebook's DeepMLP: 8 blocks of 4 saved tensors, B=256, d=1024
    B, d, nblk = 256, 1024, 8
    one = B * d * 4 / MB
    print(f"  notebook DeepMLP (B={B}, d={d}): one B x d tensor = {one:.1f} MB; without ckpt {4 * nblk + 1} tensors = {(4 * nblk + 1) * one:.0f} MB; "
          f"with ckpt {nblk + 2} tensors = {(nblk + 2) * one:.0f} MB (ratio {(4 * nblk + 1) / (nblk + 2):.1f}x)")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ks, mem, marker="o", color=SERIES[0], label="activations kept = k + L/k (layer-activations)")
    ax.axhline(L, color=SERIES[1], ls="--", lw=1.5, label="no checkpointing: L = 64")
    ax.plot([kopt], [2 * kopt], marker="*", ms=14, color=SERIES[3], ls="none", label="k = sqrt(L) = 8: 16 kept, +33 % compute")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("number of checkpoint segments k"); ax.set_ylabel("layer activations held at the peak")
    ax.set_ylim(0, 72)
    ax.set_title("Activation checkpointing: memory minimum at sqrt(L) segments (L = 64)")
    ax.legend(loc="upper center")
    save(fig, NB, "checkpointing_memory_vs_segments")


# --------------------------------------------------------------------------------------
# Section 5: torch.compile break-even and TF32 / bf16 rounding error of a matmul
# --------------------------------------------------------------------------------------
def fig_compile_breakeven():
    banner("Section 5.1: break-even of compilation (illustrative scenarios)")
    scen = [("small model on a GPU", 25.0, 3.0, 1.8), ("large model on a GPU", 90.0, 400.0, 360.0),
            ("small model on a laptop CPU", 40.0, 5.0, 4.5)]
    steps = np.logspace(2, 6, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, comp_s, eager_ms, comp_ms), c in zip(scen, SERIES[:3]):
        be = comp_s / ((eager_ms - comp_ms) / 1e3)
        print(f"  {name:28s} compile {comp_s:5.0f} s | eager {eager_ms:6.1f} ms | compiled {comp_ms:6.1f} ms "
              f"| saving {eager_ms - comp_ms:5.1f} ms/step | break-even {be:,.0f} steps")
        eager_total = steps * eager_ms / 1e3
        comp_total = comp_s + steps * comp_ms / 1e3
        ax.plot(steps, comp_total / eager_total, color=c, label=f"{name} (break-even {be:,.0f} steps)")
        ax.plot([be], [1.0], marker="o", color=c, ms=7)
    ax.axhline(1.0, color=INK_2, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training steps in the run"); ax.set_ylabel("compiled total time / eager total time")
    ax.set_title("torch.compile pays off only past the break-even step count (illustrative)")
    ax.legend(loc="upper right")
    save(fig, NB, "compile_breakeven")


def round_mantissa(x, bits):
    """Round float64 values to `bits` explicit mantissa bits (bits=10 emulates TF32/fp16 inputs, 7 = bf16)."""
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 2.0 ** (bits + 1)) / 2.0 ** (bits + 1), e)


def fig_tf32_rounding_error():
    banner("Section 5.2: relative error of a 1024 x 1024 matmul with inputs rounded to fewer mantissa bits")
    n = 1024
    A = rng.standard_normal((n, n)); Bm = rng.standard_normal((n, n))
    ref = A @ Bm
    res = {}
    for bits, lab in ((23, "float32 inputs"), (10, "TF32 / float16 inputs (10 bits)"), (7, "bfloat16 inputs (7 bits)")):
        out = round_mantissa(A, bits) @ round_mantissa(Bm, bits)
        res[lab] = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        print(f"  {lab:34s} relative error {res[lab]:.2e}   (half-ulp of the inputs: {2.0 ** -(bits + 1):.1e})")
    fp32_acc = (A.astype(np.float32) @ Bm.astype(np.float32)).astype(np.float64)
    print(f"  {'float32 inputs and accumulation':34s} relative error {np.linalg.norm(fp32_acc - ref) / np.linalg.norm(ref):.2e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labs = list(res.keys()); vals = [res[k] for k in labs]
    bar(ax, [l.replace(" inputs", "").replace(" (", "\n(") for l in labs], vals, color=BLUE, label_fmt="{:.1e}")
    ax.set_yscale("log"); ax.set_ylim(1e-8, 1e-1)
    ax.set_ylabel("relative error of A @ B (Frobenius norm)")
    ax.set_title("Rounding the inputs of a 1024 x 1024 matmul: TF32 costs about 1e-4")
    save(fig, NB, "tf32_rounding_error")


# --------------------------------------------------------------------------------------
# Section 6: random search vs a TPE-like sampler (toy), and median pruning (toy)
# --------------------------------------------------------------------------------------
def toy_objective(x, y):
    """Toy validation accuracy over x = log10(lr) in [-4, -1] and y = dropout in [0, 0.5]."""
    return 0.90 - 0.08 * (x + 2.3) ** 2 - 0.30 * (y - 0.1) ** 2 - 0.02 * np.abs(x + 2.3) * y


def tpe_like(n_trials, n_startup=5, n_cand=24, gamma=0.25, rs=None):
    xs, ys, vs = [], [], []
    for t in range(n_trials):
        if t < n_startup:
            x, y = rs.uniform(-4, -1), rs.uniform(0, 0.5)
        else:
            order = np.argsort(vs)[::-1]
            n_good = max(1, int(np.ceil(gamma * len(vs))))
            good, bad = order[:n_good], order[n_good:]
            pts = np.array(list(zip(xs, ys)))
            bw = np.array([0.3, 0.05])
            def dens(c, idx):
                d = (c[None, :] - pts[idx]) / bw
                return np.mean(np.exp(-0.5 * np.sum(d ** 2, axis=1))) + 1e-12
            base = pts[good[rs.integers(len(good), size=n_cand)]]
            cands = base + rs.normal(size=(n_cand, 2)) * bw
            cands[:, 0] = np.clip(cands[:, 0], -4, -1); cands[:, 1] = np.clip(cands[:, 1], 0, 0.5)
            scores = [dens(c, good) / dens(c, bad) if len(bad) else dens(c, good) for c in cands]
            x, y = cands[int(np.argmax(scores))]
        xs.append(x); ys.append(y); vs.append(toy_objective(x, y))
    return np.maximum.accumulate(vs)


def fig_search_and_pruning():
    banner("Section 6: random search vs TPE-like search (toy objective), and median pruning (toy curves)")
    n_trials, n_rep = 40, 200
    best = toy_objective(-2.3, 0.1)
    rand_curves, tpe_curves = [], []
    for r in range(n_rep):
        rs = np.random.default_rng(1000 + r)
        xr, yr = rs.uniform(-4, -1, n_trials), rs.uniform(0, 0.5, n_trials)
        rand_curves.append(np.maximum.accumulate(toy_objective(xr, yr)))
        tpe_curves.append(tpe_like(n_trials, rs=np.random.default_rng(5000 + r)))
    rand_curves, tpe_curves = np.array(rand_curves), np.array(tpe_curves)
    print(f"  toy optimum: accuracy {best:.3f} at log10(lr) = -2.3, dropout = 0.1 ({n_rep} repeats)")
    for t in (5, 10, 20, 40):
        print(f"  after {t:2d} trials: median best-so-far random {np.median(rand_curves[:, t - 1]):.4f} | "
              f"TPE-like {np.median(tpe_curves[:, t - 1]):.4f} | gap to optimum random {best - np.median(rand_curves[:, t - 1]):.4f}, "
              f"TPE-like {best - np.median(tpe_curves[:, t - 1]):.4f}")

    # median pruning on toy learning curves
    n_tr, n_steps, n_startup = 10, 12, 3
    prs = np.random.default_rng(7)
    plateau = prs.uniform(0.80, 0.90, n_tr); tau = prs.uniform(1.5, 5.0, n_tr)
    steps = np.arange(1, n_steps + 1)
    curves = plateau[:, None] * (1 - np.exp(-steps[None, :] / tau[:, None])) + prs.normal(0, 0.004, (n_tr, n_steps))
    history = {s: [] for s in steps}
    pruned_at = [None] * n_tr
    for i in range(n_tr):
        for s in steps:
            v = curves[i, s - 1]
            if i >= n_startup and len(history[s]) >= n_startup and v < np.median(history[s]):
                pruned_at[i] = s
                history[s].append(v)
                break
            history[s].append(v)
    used = sum(n_steps if p is None else p for p in pruned_at)
    print(f"\n  pruning toy: {n_tr} trials x {n_steps} steps; startup trials {n_startup}")
    for i in range(n_tr):
        print(f"   trial {i}: plateau {plateau[i]:.3f}, " + (f"pruned at step {pruned_at[i]}" if pruned_at[i] else f"completed, final {curves[i, -1]:.3f}"))
    print(f"  pruned {sum(p is not None for p in pruned_at)} of {n_tr}; steps used {used} of {n_tr * n_steps} ({100 * (1 - used / (n_tr * n_steps)):.0f} % saved)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    t = np.arange(1, n_trials + 1)
    for curves_, c, lab in ((rand_curves, SERIES[0], "random search"), (tpe_curves, SERIES[1], "TPE-like model-based search")):
        med = np.median(curves_, axis=0); lo, hi = np.percentile(curves_, [25, 75], axis=0)
        ax.plot(t, med, color=c, label=f"{lab} (median of {n_rep} runs)")
        ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
    ax.axhline(best, color=INK_2, ls="--", lw=0.9)
    ax.annotate(f"optimum {best:.2f}", (n_trials, best), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("trial"); ax.set_ylabel("best validation accuracy so far (toy objective)")
    ax.set_ylim(0.82, 0.905); ax.set_title("Toy 2-D search: model-based sampling converges faster")
    ax.legend(loc="lower right")
    ax = axes[1]
    for i in range(n_tr):
        end = n_steps if pruned_at[i] is None else pruned_at[i]
        c = SERIES[0] if pruned_at[i] is None else MUTED
        ax.plot(steps[:end], curves[i, :end], color=c, lw=1.4, alpha=0.9)
        if pruned_at[i] is not None:
            ax.plot(end, curves[i, end - 1], marker="x", ms=8, color=SERIES[1], mew=2)
    ax.plot([], [], color=SERIES[0], label="completed trial"); ax.plot([], [], color=MUTED, label="pruned trial")
    ax.plot([], [], marker="x", ls="none", color=SERIES[1], ms=8, mew=2, label="pruned here (below median)")
    ax.set_xlabel("report step"); ax.set_ylabel("intermediate validation accuracy")
    ax.set_title(f"Median pruning (toy): {used} of {n_tr * n_steps} steps used")
    ax.legend(loc="lower right")
    save(fig, NB, "search_and_pruning_toy")


# --------------------------------------------------------------------------------------
# Section 7: ring all-reduce traffic and ZeRO memory per device
# --------------------------------------------------------------------------------------
def fig_ddp_and_zero():
    banner("Section 7: ring all-reduce traffic per device and ZeRO bytes per parameter per device")
    models = [("notebook MLP", 269_322), ("ResNet-50", 25.6e6), ("GPT-2 small", 124.4e6), ("1B model", 1e9)]
    Ps = np.array([2, 4, 8, 16, 32, 64])
    print(f"  ring all-reduce: traffic per device = 2 (P-1)/P x gradient bytes (float32 gradients)")
    for name, n in models:
        M = n * 4
        tr8 = 2 * (8 - 1) / 8 * M
        print(f"  {name:13s} gradient {M / MB:8.2f} MB | traffic at P=8 {tr8 / MB:8.2f} MB -> {tr8 / 10e9 * 1e3:7.2f} ms at 10 GB/s, "
              f"{tr8 / 100e9 * 1e3:6.3f} ms at 100 GB/s")
    print(f"  factor 2(P-1)/P for P = {Ps.tolist()}: {np.round(2 * (Ps - 1) / Ps, 3).tolist()}")
    N7 = 7e9
    print("\n  ZeRO bytes per parameter per device (mixed precision baseline 16 B):")
    for Nd in (1, 2, 8, 64):
        s0, s1, s2, s3 = 16.0, 4 + 12 / Nd, 2 + 14 / Nd, 16 / Nd
        print(f"   N={Nd:2d}: DDP {s0:5.2f} | ZeRO-1 {s1:5.2f} | ZeRO-2 {s2:5.2f} | ZeRO-3 {s3:5.2f} B/param "
              f"-> 7B model per GPU: {s0 * N7 / 1e9:6.1f} / {s1 * N7 / 1e9:5.1f} / {s2 * N7 / 1e9:5.1f} / {s3 * N7 / 1e9:5.1f} GB")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for (name, n), c in zip(models, SERIES[:4]):
        traffic_ms = 2 * (Ps - 1) / Ps * n * 4 / 10e9 * 1e3
        ax.plot(Ps, traffic_ms, marker="o", color=c, label=f"{name} ({n / 1e6:.1f}M params)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(Ps); ax.set_xticklabels([str(p) for p in Ps])
    ax.set_xlabel("number of devices P"); ax.set_ylabel("all-reduce time per step at 10 GB/s (ms)")
    ax.set_title("Ring all-reduce: traffic per device saturates at 2x the gradient")
    ax.legend(loc="center right", fontsize=8)
    ax = axes[1]
    Nd = np.array([1, 2, 4, 8, 16, 32, 64])
    for vals, c, lab in ((np.full_like(Nd, 16.0, dtype=float), SERIES[0], "DDP (replicated): 16"),
                         (4 + 12 / Nd, SERIES[1], "ZeRO-1: 4 + 12/N"),
                         (2 + 14 / Nd, SERIES[2], "ZeRO-2: 2 + 14/N"),
                         (16 / Nd, SERIES[3], "ZeRO-3 / FSDP: 16/N")):
        ax.plot(Nd, vals, marker="o", color=c, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(Nd); ax.set_xticklabels([str(n) for n in Nd])
    ax.set_xlabel("number of devices N"); ax.set_ylabel("training-state bytes per parameter per device")
    ax.set_title("Sharding the training state (mixed precision + Adam)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, NB, "ddp_allreduce_and_zero")


# --------------------------------------------------------------------------------------
# Section 8: 6ND compute, the scaling-law surface with the compute-optimal frontier, seeds
# --------------------------------------------------------------------------------------
def training_hours(flops, n_gpus, peak=312e12, mfu=0.4):
    return flops / (n_gpus * peak * mfu) / 3600


def fig_six_nd():
    banner("Section 8.1: C = 6 N D for well-known models (A100 bf16 peak 312 TFLOP/s, MFU 40 %)")
    models = [("GPT-2 small", 124e6, 9e9), ("Llama-2 7B", 7e9, 2e12), ("Chinchilla 70B", 70e9, 1.4e12),
              ("GPT-3 175B", 175e9, 300e9)]
    # the notebook's FLOP-counter check on MLP(512, depth 2), batch 256
    N1, N2, N3, B = 784 * 512, 512 * 512, 512 * 10, 256
    N = N1 + N2 + N3
    print(f"  MLP(512, depth 2): N = {N:,} weights; forward 2NB = {2 * N * B:,.0f} FLOPs; "
          f"forward+backward = 6NB - 2 N1 B = {6 * N * B - 2 * N1 * B:,.0f} -> ratio to 6NB {(6 * N * B - 2 * N1 * B) / (6 * N * B):.3f}")
    print(f"  {'model':15s} {'N':>8s} {'D':>8s} {'C=6ND':>9s} {'A100-hours':>11s} {'days on 256':>12s} {'tokens/param':>12s}")
    rows = []
    for name, n, d in models:
        C = 6 * n * d
        rows.append((name, C, training_hours(C, 1)))
        print(f"  {name:15s} {n / 1e9:7.3f}B {d / 1e9:7.0f}B {C:9.2e} {training_hours(C, 1):11,.0f} {training_hours(C, 256) / 24:12.1f} {d / n:12.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [r[0] for r in rows]; hours = [r[2] for r in rows]
    bar(ax, names, hours, color=BLUE, label_fmt="{:,.0f} h")
    ax.set_yscale("log"); ax.set_ylim(1, 1e7)
    ax.set_ylabel("A100-hours at 40 % MFU (log scale)")
    ax.set_title("Training compute C = 6ND converted to GPU-hours")
    save(fig, NB, "six_nd_compute")


def fig_scaling_law_surface():
    banner("Section 8.2: parametric loss L(N, D) with the notebook's illustrative constants")
    E, A, Bc, al, be = 1.7, 400.0, 1140.0, 0.35, 0.35
    L = lambda N, D: E + A / N ** al + Bc / D ** be
    ratio = (Bc / A) ** (1 / al)
    print(f"  constants E={E} A={A} B={Bc} alpha={al} beta={be} -> optimal D/N = (B/A)^(1/alpha) = {ratio:.1f}")
    print(f"  {'C':>8s} {'N_opt':>9s} {'D_opt':>9s} {'L_opt':>7s} {'L at 2 N_opt':>13s} {'L at N_opt/2':>13s}")
    for C in (1e19, 1e21, 1e23, 5.76e23):
        Nopt = np.sqrt(C / 120); Dopt = 20 * Nopt
        print(f"  {C:8.2e} {Nopt / 1e9:8.2f}B {Dopt / 1e9:8.0f}B {L(Nopt, Dopt):7.4f} {L(2 * Nopt, Dopt / 2):13.4f} {L(Nopt / 2, 2 * Dopt):13.4f}")
    gpt3_N, gpt3_D = 175e9, 300e9
    C3 = 6 * gpt3_N * gpt3_D; N3 = np.sqrt(C3 / 120)
    print(f"  GPT-3 budget C={C3:.2e}: actual L(175B, 300B) = {L(gpt3_N, gpt3_D):.4f} | compute-optimal N={N3 / 1e9:.1f}B, D={20 * N3 / 1e9:.0f}B, L = {L(N3, 20 * N3):.4f}")

    Ns = np.logspace(8, 12, 300); Ds = np.logspace(9, 13, 300)
    NN, DD = np.meshgrid(Ns, Ds)
    LL = L(NN, DD)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pcm = ax.pcolormesh(NN, DD, LL, cmap=SEQ_CMAP, shading="auto", vmin=1.75, vmax=2.8)
    fig.colorbar(pcm, ax=ax, label="loss L(N, D) (parametric fit, illustrative constants)")
    for C, c in zip((1e20, 1e22, 1e24), (SERIES[1], SERIES[3], SERIES[4])):
        ax.plot(Ns, C / (6 * Ns), color=c, lw=1.5, label=f"iso-compute C = {C:.0e}")
    ax.plot(Ns, 20 * Ns, color=INK, lw=2.0, ls="--", label="compute-optimal frontier D = 20 N")
    ax.plot([gpt3_N], [gpt3_D], marker="o", ms=8, color=SERIES[7], ls="none", label="GPT-3 (175B, 300B tokens)")
    ax.plot([70e9], [1.4e12], marker="s", ms=8, color=SERIES[2], ls="none", label="Chinchilla (70B, 1.4T)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e8, 1e12); ax.set_ylim(1e9, 1e13)
    ax.set_xlabel("parameters N"); ax.set_ylabel("training tokens D")
    ax.set_title("Loss surface L(N, D): follow the frontier along an iso-compute line")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, NB, "scaling_law_surface")


def fig_ablation_seeds():
    banner("Section 8.3: one seed versus three seeds for a four-way ablation (simulated)")
    names = ["baseline", "lr_x2", "wide_512", "adamw_1e-3"]
    true = np.array([0.882, 0.886, 0.884, 0.879])
    sd = 0.004
    srs = np.random.default_rng(11)
    one = true + srs.normal(0, sd, 4)
    three = true[:, None] + srs.normal(0, sd, (4, 3))
    print(f"  true means {true.tolist()}, seed-to-seed std {sd}")
    print(f"  single-seed results: " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, one)))
    print(f"  single-seed ranking: {[names[i] for i in np.argsort(-one)]}")
    print(f"  three-seed means +- std: " + ", ".join(f"{n} {m:.4f}+-{s:.4f}" for n, m, s in zip(names, three.mean(1), three.std(1, ddof=1))))
    print(f"  three-seed ranking: {[names[i] for i in np.argsort(-three.mean(1))]} | true ranking: {[names[i] for i in np.argsort(-true)]}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(4)
    ax.bar(x - 0.2, one, width=0.4, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, label="one seed")
    ax.bar(x + 0.2, three.mean(1), width=0.4, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5,
           yerr=three.std(1, ddof=1), capsize=4, ecolor=INK_2, label="mean of three seeds (error bar = std)")
    ax.plot(x, true, marker="D", ms=7, color=INK, ls="none", label="true mean (simulation)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.86, 0.90); ax.set_ylabel("validation accuracy")
    ax.set_title("Ablations: a single seed can reorder configurations that differ by 0.4 points")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, NB, "ablation_seeds_error_bars")


def main():
    fig_timing_and_amdahl()
    fig_memory_breakdown()
    fig_float_formats()
    fig_loss_scaling()
    fig_grad_accumulation()
    fig_checkpointing_tradeoff()
    fig_compile_breakeven()
    fig_tf32_rounding_error()
    fig_search_and_pruning()
    fig_ddp_and_zero()
    fig_six_nd()
    fig_scaling_law_surface()
    fig_ablation_seeds()


if __name__ == "__main__":
    main()
