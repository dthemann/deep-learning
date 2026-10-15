"""Figures for the companion guide of Notebook 08 · Sequence Models: RNNs, LSTMs and GRUs.

Everything here is numpy + matplotlib. The vanilla RNN cell and the LSTM cell (PyTorch's gate order
i, f, g, o) are re-implemented in numpy together with their exact backward passes (backpropagation
through time), so that the gradient-versus-lag experiments and the delayed-recall experiment of the
notebook can be reproduced without PyTorch. A least-squares linear model and a tiny numpy MLP stand in
for the notebook's forecasters, and a toy "superposition memory" illustrates the seq2seq bottleneck.
Both backward passes are verified against finite differences when the script starts.

Run:  python3 notebooks/figures/scripts/make_figures_08.py
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
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Circle, FancyArrowPatch

NB = "08"
SEED = 42            # same seed as the notebook (the forecasting series below is the notebook's series)


def sigmoid(z):
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# ----------------------------------------------------------------------------------------------
# Vanilla RNN and LSTM cells in numpy, with exact backward passes (BPTT)
# ----------------------------------------------------------------------------------------------
def rnn_init(D, H, seed=SEED):
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b); PyTorch-style uniform init in +-1/sqrt(H), zero bias."""
    r = np.random.default_rng(seed)
    k = 1.0 / math.sqrt(H)
    return {"W_xh": r.uniform(-k, k, (H, D)), "W_hh": r.uniform(-k, k, (H, H)), "b": np.zeros(H)}


def rnn_forward(p, X):
    """X: (B, T, D). Returns the final state (B, H) and a per-step cache."""
    B, T, _ = X.shape
    H = p["W_hh"].shape[0]
    h, cache = np.zeros((B, H)), []
    for t in range(T):
        h_prev = h
        h = np.tanh(X[:, t] @ p["W_xh"].T + h_prev @ p["W_hh"].T + p["b"])
        cache.append((X[:, t], h_prev, h))
    return h, cache


def rnn_backward(p, cache, dh):
    """Gradients of a loss that depends on the final state only (dh = dL/dh_T, shape (B, H))."""
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dX = np.zeros((cache[0][0].shape[0], len(cache), p["W_xh"].shape[1]))
    for t in reversed(range(len(cache))):
        x_t, h_prev, h = cache[t]
        da = dh * (1.0 - h ** 2)                      # through tanh
        grads["W_xh"] += da.T @ x_t
        grads["W_hh"] += da.T @ h_prev
        grads["b"] += da.sum(axis=0)
        dX[:, t] = da @ p["W_xh"]
        dh = da @ p["W_hh"]                           # dL/dh_{t-1} = W_hh^T diag(1-h_t^2) dL/dh_t
    return grads, dX


def lstm_init(D, H, forget_bias=0.0, seed=SEED, uniform_bias=False):
    """PyTorch parametrisation: stacked gates (i, f, g, o), W_ih (4H, D), W_hh (4H, H), one bias b (4H).

    The forget-gate slice b[H:2H] is set to `forget_bias` (the notebook's set_forget_bias)."""
    r = np.random.default_rng(seed)
    k = 1.0 / math.sqrt(H)
    p = {"W_ih": r.uniform(-k, k, (4 * H, D)), "W_hh": r.uniform(-k, k, (4 * H, H)),
         "b": r.uniform(-k, k, 4 * H) + r.uniform(-k, k, 4 * H) if uniform_bias else np.zeros(4 * H)}
    p["b"][H:2 * H] = forget_bias
    return p


def lstm_forward(p, X):
    """X: (B, T, D). Returns (h_T, c_T) and a per-step cache."""
    B, T, _ = X.shape
    H = p["W_hh"].shape[1]
    h, c, cache = np.zeros((B, H)), np.zeros((B, H)), []
    for t in range(T):
        h_prev, c_prev = h, c
        gates = X[:, t] @ p["W_ih"].T + h_prev @ p["W_hh"].T + p["b"]
        i, f, g, o = np.split(gates, 4, axis=1)
        i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
        c = f * c_prev + i * g                        # additive memory update
        tc = np.tanh(c)
        h = o * tc                                    # gated exposure
        cache.append((X[:, t], h_prev, c_prev, i, f, g, o, tc))
    return (h, c), cache


def lstm_backward(p, cache, dh):
    """Exact BPTT for a loss that depends on the final hidden state only (dh = dL/dh_T).

    Returns parameter gradients, dL/dX and the list of ||dL/dc_t|| (total gradient, both the direct
    cell path and the paths through h_t) for every step t, per sample."""
    H = p["W_hh"].shape[1]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    B, T, D = cache[0][0].shape[0], len(cache), p["W_ih"].shape[1]
    dX, dc, c_norms = np.zeros((B, T, D)), np.zeros_like(dh), [None] * T
    for t in reversed(range(T)):
        x_t, h_prev, c_prev, i, f, g, o, tc = cache[t]
        dc = dc + dh * o * (1.0 - tc ** 2)            # dL/dc_t: from c_{t+1} (already in dc) and from h_t
        c_norms[t] = np.linalg.norm(dc, axis=1)
        do, di, dg, df = dh * tc, dc * g, dc * i, dc * c_prev
        da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
        grads["W_ih"] += da.T @ x_t
        grads["W_hh"] += da.T @ h_prev
        grads["b"] += da.sum(axis=0)
        dX[:, t] = da @ p["W_ih"]
        dh = da @ p["W_hh"]                           # into h_{t-1}
        dc = dc * f                                   # into c_{t-1} along the cell path: diag(f_t)
    return grads, dX, c_norms


def gradient_checks():
    """Verify both backward passes against central finite differences on a tiny problem."""
    r = np.random.default_rng(1)
    B, T, D, H = 2, 5, 3, 4
    X = r.standard_normal((B, T, D)); w = r.standard_normal(H)
    eps, report = 1e-6, {}
    for name, init, fwd, bwd in [("RNN", rnn_init, rnn_forward, rnn_backward),
                                 ("LSTM", lambda D, H: lstm_init(D, H, 1.0, uniform_bias=True), lstm_forward, lstm_backward)]:
        p = init(D, H)
        def loss(p):
            out = fwd(p, X)[0]
            h = out[0] if isinstance(out, tuple) else out
            return float(np.sum(h * w))
        out, cache = fwd(p, X)
        dh = np.tile(w, (B, 1))
        grads = bwd(p, cache, dh)[0]
        worst = 0.0
        for k in p:
            for idx in [np.unravel_index(j, p[k].shape) for j in r.choice(p[k].size, min(6, p[k].size), replace=False)]:
                p[k][idx] += eps; lp = loss(p); p[k][idx] -= 2 * eps; lm = loss(p); p[k][idx] += eps
                worst = max(worst, abs((lp - lm) / (2 * eps) - grads[k][idx]))
        report[name] = worst
    print(f"[check] BPTT gradients vs finite differences: RNN max abs error {report['RNN']:.1e}, LSTM {report['LSTM']:.1e}")
    # dL/dc_t by finite differences for the LSTM: perturb c_t and rerun the remaining steps
    p = lstm_init(D, H, 1.0, uniform_bias=True)
    (h, c), cache = lstm_forward(p, X[:1])
    _, _, c_norms = lstm_backward(p, cache, w[None, :])
    t0 = 1
    def rerun(c_pert):
        hh, cc = cache[t0][6] * np.tanh(c_pert), c_pert   # h_{t0} = o_{t0} * tanh(c_{t0}) follows the perturbation
        for t in range(t0 + 1, T):
            gates = X[:1, t] @ p["W_ih"].T + hh @ p["W_hh"].T + p["b"]
            i, f, g, o = np.split(gates, 4, axis=1)
            cc = sigmoid(f) * cc + sigmoid(i) * np.tanh(g); hh = sigmoid(o) * np.tanh(cc)
        return float(np.sum(hh * w))
    c_t0 = cache[t0 + 1][2]
    fd = np.array([(rerun(c_t0 + eps * np.eye(H)[j]) - rerun(c_t0 - eps * np.eye(H)[j])) / (2 * eps) for j in range(H)])
    print(f"[check] ||dL/dc_t|| analytic {c_norms[t0][0]:.6f} vs finite differences {np.linalg.norm(fd):.6f}")
    return report


# ----------------------------------------------------------------------------------------------
# Section 1: same values, different order
# ----------------------------------------------------------------------------------------------
def fig_order_matters():
    T = 8
    rising = np.round(np.sin(np.linspace(0, 2 * np.pi, T) / 4), 2)
    falling = rising[::-1]
    shuffled = rising[np.random.default_rng(SEED).permutation(T)]
    print("\n== Section 1: same values, different order ==")
    print("t       :", list(range(T)))
    print("rising  :", rising.tolist()); print("falling :", falling.tolist()); print("shuffled:", shuffled.tolist())
    for name, s in [("rising", rising), ("falling", falling), ("shuffled", shuffled)]:
        print(f"{name:8s}: mean {s.mean():.4f} | last value {s[-1]:+.2f} | first value {s[0]:+.2f} | sorted equal to rising sorted: {np.array_equal(np.sort(s), np.sort(rising))}")
    fig, ax = plt.subplots()
    for name, s, c, m in zip(["rising", "falling", "shuffled"], [rising, falling, shuffled], SERIES[:3], ["o", "s", "^"]):
        ax.plot(range(T), s, marker=m, color=c, label=name)
    ax.axhline(rising.mean(), color=MUTED, linewidth=0.9)
    ax.annotate(f"mean of all three = {rising.mean():.3f}", (T - 1, rising.mean()), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("time step t"); ax.set_ylabel("value"); ax.set_title("Same eight values, three orders")
    ax.legend(loc="center left")
    save(fig, NB, "same_values_different_order")


# ----------------------------------------------------------------------------------------------
# Section 2: a vanilla RNN with one hidden unit, three steps by hand
# ----------------------------------------------------------------------------------------------
def fig_rnn_forward_one_unit():
    w_xh, w_hh, b = 0.8, 0.5, -0.1
    x = np.array([1.0, 0.5, -1.0])

    def run(xs):
        h, rows = 0.0, [(0, None, None, 0.0)]
        for t, xt in enumerate(xs, 1):
            a = w_xh * xt + w_hh * h + b
            h = math.tanh(a)
            rows.append((t, xt, a, h))
        return rows

    print("\n== Section 2: one-unit RNN forward pass (w_xh=0.8, w_hh=0.5, b=-0.1) ==")
    runs = {"inputs 1.0, 0.5, -1.0": run(x), "same inputs reversed (-1.0, 0.5, 1.0)": run(x[::-1])}
    for name, rows in runs.items():
        print(name)
        for t, xt, a, h in rows:
            print(f"  t={t}: x_t={'-' if xt is None else f'{xt:+.2f}'}  a_t={'-' if a is None else f'{a:+.4f}'}  h_t={h:+.4f}")
    hs = [r[3] for r in runs["inputs 1.0, 0.5, -1.0"]]
    factors = [(1 - hs[t] ** 2) * w_hh for t in (2, 3)]
    print("BPTT by hand for the first run: dh_t/dh_(t-1) = (1 - h_t^2) * w_hh")
    print(f"  t=2: (1 - {hs[2]:+.4f}^2) * 0.5 = {factors[0]:.4f} | t=3: (1 - {hs[3]:+.4f}^2) * 0.5 = {factors[1]:.4f} | "
          f"dh_3/dh_1 = product = {factors[0] * factors[1]:.4f}; with w_hh = 1.5 instead: {factors[0] * 3 * factors[1] * 3:.4f}")
    print(f"  parameter count of the notebook's cell (H=8, D=5): H*D + H*H + H = {8 * 5 + 8 * 8 + 8}; nn.RNN has one more bias vector: {8 * 5 + 8 * 8 + 2 * 8}")
    fig, ax = plt.subplots()
    for (name, rows), c, m in zip(runs.items(), SERIES[:2], ["o", "s"]):
        hs = [r[3] for r in rows]
        ax.plot(range(len(hs)), hs, marker=m, color=c, label=name)
        ax.annotate(f"h_3 = {hs[-1]:+.3f}", (3, hs[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(range(4)); ax.set_xlim(-0.2, 3.9); ax.set_ylim(-1.1, 0.85)
    ax.set_xlabel("time step t"); ax.set_ylabel("hidden state h_t")
    ax.set_title("One hidden unit, three steps: the order of the inputs changes the state")
    ax.legend(loc="lower right")
    save(fig, NB, "rnn_forward_pass_one_unit")


# ----------------------------------------------------------------------------------------------
# Section 3: gradient norm versus lag (explicit Jacobian products) and clipping
# ----------------------------------------------------------------------------------------------
def rnn_grad_norm_vs_lag(gain, T=60, D=8, H=32, act="tanh", seed=SEED):
    """||dL/dh_t|| for a readout on the last state, W_hh = gain * Q (Q orthogonal), W_xh = 0.1 * N(0, 1)."""
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((H, H)))
    W_hh, W_xh = gain * Q, 0.1 * r.standard_normal((H, D))
    x, h, hs = r.standard_normal((T, D)), np.zeros(H), []
    for t in range(T):
        a = W_xh @ x[t] + W_hh @ h
        h = np.tanh(a) if act == "tanh" else a
        hs.append(h)
    w = r.standard_normal(H)
    g, norms = w.copy(), [np.linalg.norm(w)]               # dL/dh_{T-1} = w  (lag 0)
    for t in range(T - 1, 0, -1):                           # dL/dh_{t-1} = W_hh^T diag(tanh') dL/dh_t
        d = (1.0 - hs[t] ** 2) if act == "tanh" else np.ones(H)
        g = W_hh.T @ (d * g)
        norms.append(np.linalg.norm(g))
    return np.arange(T), np.array(norms)


def fig_gradient_norm_vs_lag():
    print("\n== Section 3.4: gradient norm versus lag, tanh RNN with W_hh = gain * Q (H=32, T=60) ==")
    fig, ax = plt.subplots()
    for gain, c in zip([0.5, 1.0, 1.5], SERIES[:3]):
        lags, norms = rnn_grad_norm_vs_lag(gain)
        slope = np.polyfit(lags, np.log(norms), 1)[0]
        print(f"gain {gain:.1f}: norm at lag 0 = {norms[0]:.3f}, at lag 30 = {norms[30]:.2e}, at lag 59 = {norms[59]:.2e} | "
              f"fitted slope {slope:+.3f} per step (factor {math.exp(slope):.3f})")
        ax.semilogy(lags, norms, color=c, label=f"tanh RNN, gain {gain}")
    lags, norms_id = rnn_grad_norm_vs_lag(1.0, act="identity")
    ratios = norms_id[1:] / norms_id[:-1]
    print(f"identity activation, gain 1.0: per-step ratio min {ratios.min():.6f} max {ratios.max():.6f} (exactly 1: nothing vanishes)")
    for gain in [0.5, 1.5]:
        _, n_id = rnn_grad_norm_vs_lag(gain, act="identity")
        rr = n_id[1:] / n_id[:-1]
        print(f"identity activation, gain {gain}: per-step ratio {rr.min():.4f} .. {rr.max():.4f} (claim: exactly {gain})")
    ax.semilogy(lags, norms_id, color=MUTED, linestyle="--", linewidth=1.2, label="no non-linearity, gain 1.0 (reference)")
    ax.set_xlabel("lag (steps between the state and the loss)"); ax.set_ylabel("gradient norm ||dL/dh_t||")
    ax.set_title("Gradient norm versus lag in a vanilla RNN (numpy Jacobian products)")
    ax.legend(loc="lower left")
    save(fig, NB, "gradient_norm_vs_lag_tanh_rnn")


def fig_clipping():
    g, tau, v = np.array([4.0, 1.0]), 1.0, 0.5
    g_norm = g * min(1.0, tau / np.linalg.norm(g))
    g_val = np.clip(g, -v, v)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("\n== Section 3.5: clipping a 2-D gradient g = (4.0, 1.0) ==")
    print(f"||g|| = {np.linalg.norm(g):.3f}")
    print(f"clip by norm (tau=1):   g -> ({g_norm[0]:.3f}, {g_norm[1]:.3f}), norm {np.linalg.norm(g_norm):.3f}, cosine with g {cos(g, g_norm):.4f}")
    print(f"clip by value (+-0.5):  g -> ({g_val[0]:.3f}, {g_val[1]:.3f}), norm {np.linalg.norm(g_val):.3f}, cosine with g {cos(g, g_val):.4f}")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.add_patch(Circle((0, 0), tau, fill=False, edgecolor=MUTED, linewidth=0.9))
    for vec, c, name in [(g, SERIES[0], "original gradient"), (g_norm, SERIES[1], "clipped by norm (same direction)"),
                         (g_val, SERIES[2], "clipped by value (rotated)")]:
        ax.add_patch(FancyArrowPatch((0, 0), tuple(vec), arrowstyle="-|>", mutation_scale=14, color=c, linewidth=2))
        ax.plot([], [], color=c, linewidth=2, label=name)
    ax.annotate("threshold circle, radius 1", (0.72, -0.72), fontsize=8, color=INK_2)
    ax.set_xlim(-0.3, 4.4); ax.set_ylim(-1.1, 1.6); ax.set_aspect("equal")
    ax.set_xlabel("gradient component 1"); ax.set_ylabel("gradient component 2")
    ax.set_title("Clipping by norm keeps the direction, clipping by value does not")
    ax.legend(loc="upper right")
    save(fig, NB, "clipping_norm_vs_value")


# ----------------------------------------------------------------------------------------------
# Section 4: LSTM gates by hand, and the cell-path gradient versus lag (exact BPTT)
# ----------------------------------------------------------------------------------------------
def fig_lstm_gate_example():
    units = {"A: remember": dict(a_i=-3.0, a_f=3.0, a_g=0.5, a_o=1.0, c_prev=0.8),
             "B: overwrite": dict(a_i=3.0, a_f=-3.0, a_g=-1.5, a_o=2.0, c_prev=0.8),
             "C: store but hide": dict(a_i=1.0, a_f=1.0, a_g=1.0, a_o=-4.0, c_prev=-0.5)}
    print("\n== Section 4.2: one LSTM step for three units (hand-typed gate pre-activations) ==")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for ax, (name, u) in zip(axes, units.items()):
        i, f, g, o = sigmoid(u["a_i"]), sigmoid(u["a_f"]), math.tanh(u["a_g"]), sigmoid(u["a_o"])
        keep, write = f * u["c_prev"], i * g
        c_new = keep + write
        h_new = o * math.tanh(c_new)
        print(f"unit {name}: pre-activations a_i={u['a_i']:+.1f} a_f={u['a_f']:+.1f} a_g={u['a_g']:+.1f} a_o={u['a_o']:+.1f}, c_prev={u['c_prev']:+.2f}")
        print(f"   i={i:.3f} f={f:.3f} g~={g:+.3f} o={o:.3f} | f*c_prev={keep:+.3f} i*g~={write:+.3f} c_new={c_new:+.3f} tanh(c_new)={math.tanh(c_new):+.3f} h_new={h_new:+.4f}")
        labels = ["c_prev", "f·c_prev", "i·g~", "c_new", "h_new"]
        values = [u["c_prev"], keep, write, c_new, h_new]
        bars = ax.bar(labels, values, width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        for b_, val in zip(bars, values):
            ax.annotate(f"{val:+.2f}", (b_.get_x() + b_.get_width() / 2, val), xytext=(0, 3 if val >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_title(f"unit {name}\nf={f:.2f}  i={i:.2f}  o={o:.2f}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("value"); axes[0].set_ylim(-1.1, 1.1)
    fig.suptitle("One LSTM step: the gates decide what is kept, written and exposed", fontweight="bold", fontsize=12)
    save(fig, NB, "lstm_gate_worked_example")


def fig_lstm_cell_gradient_vs_lag():
    T, D, H = 60, 8, 32
    print("\n== Section 4.5: ||dL/dc_t|| versus lag for an LSTM (exact numpy BPTT, H=32, T=60) ==")
    fig, ax = plt.subplots()
    lags, norms_rnn = rnn_grad_norm_vs_lag(1.0, T=T, D=D, H=H)
    ax.semilogy(lags, norms_rnn, color=INK_2, linestyle="--", linewidth=1.2, label="vanilla RNN, gain 1.0 (reference)")
    r = np.random.default_rng(SEED)
    X = r.standard_normal((1, T, D)); w = r.standard_normal(H)
    for fb, c in zip([0.0, 2.0, 5.0], SERIES[:3]):
        p = lstm_init(D, H, forget_bias=fb)
        _, cache = lstm_forward(p, X)
        _, _, c_norms = lstm_backward(p, cache, w[None, :])
        norms = np.array([c_norms[t][0] for t in range(T)])[::-1]      # index = lag = T-1-t
        slope = np.polyfit(lags, np.log(norms), 1)[0]
        f_mean = np.mean([cache[t][4].mean() for t in range(T)])
        print(f"forget bias {fb:.0f} (sigmoid = {sigmoid(fb):.3f}, mean forget gate over the run {f_mean:.3f}): "
              f"fitted slope {slope:+.4f} per step, factor {math.exp(slope):.3f}, norm at lag 0 {norms[0]:.3f}, lag 59 {norms[59]:.2e}; "
              f"ideal f^59 = {sigmoid(fb) ** 59:.2e}")
        ax.semilogy(lags, norms, color=c, label=f"LSTM cell path, forget bias {fb:.0f} (f = {sigmoid(fb):.2f})")
    print(f"0.99^100 = {0.99 ** 100:.3f} | 0.5^30 = {0.5 ** 30:.1e} | 0.88^30 = {0.88 ** 30:.3f} | 0.993^60 = {0.993 ** 60:.3f}")
    ax.set_xlabel("lag (steps between the cell state and the loss)"); ax.set_ylabel("gradient norm ||dL/dc_t||")
    ax.set_title("The forget gate sets the decay rate of the gradient along the cell path")
    ax.legend(loc="lower left")
    save(fig, NB, "cell_state_gradient_vs_lag")


# ----------------------------------------------------------------------------------------------
# Section 5: the delayed-recall experiment, trained with numpy BPTT and Adam
# ----------------------------------------------------------------------------------------------
class Adam:
    """torch.optim.Adam defaults (betas 0.9/0.999, eps 1e-8), on a dict of arrays."""
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_by_norm(grads, max_norm=1.0):
    total = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] *= max_norm / (total + 1e-6)
    return total


def train_recall(kind, T, steps=300, batch=64, lr=5e-3, gate_bias=2.0, n_test=1000, seed=SEED):
    """Delayed recall: key (0..3) at position 0, distractors (4..11) elsewhere; predict the key from h_T."""
    V, C, E, H = 12, 4, 16, 32
    r = np.random.default_rng(seed)
    params = {"emb": r.standard_normal((V, E)), "W_head": r.uniform(-1 / math.sqrt(H), 1 / math.sqrt(H), (C, H)), "b_head": np.zeros(C)}
    cell = rnn_init(E, H, seed) if kind == "RNN" else lstm_init(E, H, gate_bias, seed, uniform_bias=True)
    params.update(cell)
    fwd, bwd = (rnn_forward, rnn_backward) if kind == "RNN" else (lstm_forward, lstm_backward)
    opt = Adam(params, lr)

    def make_batch(n):
        key = r.integers(0, C, n)
        return np.concatenate([key[:, None], r.integers(C, V, (n, T - 1))], axis=1), key

    def logits_of(x):
        out, cache = fwd(params, params["emb"][x])
        h = out[0] if isinstance(out, tuple) else out
        return h @ params["W_head"].T + params["b_head"], h, cache

    for _ in range(steps):
        x, y = make_batch(batch)
        logits, h, cache = logits_of(x)
        pr = np.exp(logits - logits.max(axis=1, keepdims=True)); pr /= pr.sum(axis=1, keepdims=True)
        dlogits = pr.copy(); dlogits[np.arange(batch), y] -= 1.0; dlogits /= batch
        grads = {"W_head": dlogits.T @ h, "b_head": dlogits.sum(axis=0)}
        cell_grads, dX = bwd(params, cache, dlogits @ params["W_head"])[:2]
        grads.update(cell_grads)
        grads["emb"] = np.zeros_like(params["emb"]); np.add.at(grads["emb"], x.reshape(-1), dX.reshape(-1, E))
        clip_by_norm(grads, 1.0)
        opt.step(grads)
    x, y = make_batch(n_test)
    return float((logits_of(x)[0].argmax(axis=1) == y).mean())


def fig_delayed_recall():
    lengths, steps = [5, 15, 30], 300
    print(f"\n== Section 5: delayed recall, numpy RNN vs LSTM (emb 16, hidden 32, Adam 5e-3, clip 1, {steps} steps, batch 64) ==")
    r = np.random.default_rng(SEED)
    key = r.integers(0, 4, 2); noise = r.integers(4, 12, (2, 7))
    print("example sequences (T=8):", np.concatenate([key[:, None], noise], axis=1).tolist(), "-> keys:", key.tolist())
    results = {"RNN": [], "LSTM (forget bias 2)": []}
    for T in lengths:
        for name in results:
            acc = train_recall("RNN" if name == "RNN" else "LSTM", T, steps=steps)
            results[name].append(acc)
        print(f"T = {T:2d} | " + " | ".join(f"{k}: {v[-1]:.3f}" for k, v in results.items()))
    acc_fb0 = train_recall("LSTM", 30, steps=steps, gate_bias=0.0)
    print(f"T = 30 | LSTM with forget bias 0 (f = 0.5): {acc_fb0:.3f}")
    fig, ax = plt.subplots()
    for (name, accs), c, m in zip(results.items(), SERIES[:2], ["o", "s"]):
        ax.plot(lengths, accs, marker=m, color=c, markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
    ax.plot([30], [acc_fb0], marker="D", color=SERIES[2], markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none",
            label="LSTM with forget bias 0, T = 30")
    ax.axhline(0.25, color=MUTED, linewidth=0.9)
    ax.annotate("chance (25%)", (lengths[0], 0.25), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log"); ax.set_xticks(lengths); ax.set_xticklabels([str(l) for l in lengths])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.05); ax.set_xlabel("sequence length T (log scale)"); ax.set_ylabel("test accuracy (1,000 sequences)")
    ax.set_title(f"Delayed recall: accuracy versus length ({steps} training steps per run)")
    ax.legend(loc="center left")
    save(fig, NB, "delayed_recall_accuracy_vs_length")


# ----------------------------------------------------------------------------------------------
# Section 6: perplexity and temperature
# ----------------------------------------------------------------------------------------------
def fig_perplexity():
    probs = np.array([0.50, 0.25, 0.10, 0.80, 0.05, 0.30])     # probability the model gave each actual next character
    nll = -np.log(probs)
    print("\n== Section 6.4: perplexity of a six-character example ==")
    print("p(next char):", probs.tolist()); print("-log p      :", np.round(nll, 3).tolist())
    print(f"mean NLL = {nll.mean():.4f} nats -> perplexity exp(mean NLL) = {math.exp(nll.mean()):.3f}")
    print(f"uniform over 65 characters: NLL log(65) = {math.log(65):.3f} nats, perplexity 65 | 1.4 nats -> ppl {math.exp(1.4):.2f} | 2.5 nats -> ppl {math.exp(2.5):.2f}")
    ce = np.linspace(0, 4.5, 200)
    fig, ax = plt.subplots()
    ax.plot(ce, np.exp(ce), color=SERIES[0])
    for x_, label, off, ha in [(math.log(65), "uniform over 65 characters", (-8, 6), "right"),
                               (2.5, "character bigram statistics only (about 12)", (-8, 6), "right"),
                               (1.4, "good character LSTM (about 4)", (-8, 8), "right"),
                               (nll.mean(), f"six-character example ({math.exp(nll.mean()):.2f})", (10, -2), "left")]:
        ax.plot([x_], [math.exp(x_)], marker="o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(label, (x_, math.exp(x_)), xytext=off, textcoords="offset points", ha=ha, fontsize=8, color=INK_2)
    ax.set_xlabel("cross-entropy (nats per character)"); ax.set_ylabel("perplexity = exp(cross-entropy)")
    ax.set_title("Perplexity is the exponential of the mean next-character loss")
    save(fig, NB, "perplexity_vs_cross_entropy")


def fig_temperature():
    chars = ["e", "a", "i", "o", "r"]
    z = np.array([2.0, 1.2, 0.6, 0.3, -0.8])
    print("\n== Section 6.6: temperature on a softmax over five characters, logits", z.tolist(), "==")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    for ax, tau in zip(axes, [0.3, 1.0, 1.5]):
        p = np.exp(z / tau); p /= p.sum()
        print(f"tau = {tau}: " + ", ".join(f"{c}={pi:.3f}" for c, pi in zip(chars, p)) + f" | most likely {p.max():.3f}, least likely {p.min():.3f}")
        bar(ax, chars, p, color=BLUE, label_fmt="{:.2f}")
        ax.set_title(f"temperature {tau}"); ax.set_xlabel("next character")
    axes[0].set_ylabel("probability"); axes[0].set_ylim(0, 1.05)
    fig.suptitle("Dividing the logits by a temperature sharpens (T < 1) or flattens (T > 1) the distribution", fontweight="bold", fontsize=12)
    save(fig, NB, "temperature_softmax")


# ----------------------------------------------------------------------------------------------
# Section 7: forecasting with a least-squares linear model and a numpy MLP
# ----------------------------------------------------------------------------------------------
def make_series():
    """The notebook's synthetic series: two sinusoids (periods 50 and 13), a trend and noise, seed 42."""
    r = np.random.default_rng(SEED)
    N = 3000
    tt = np.arange(N)
    s = (np.sin(2 * np.pi * tt / 50) + 0.5 * np.sin(2 * np.pi * tt / 13 + 0.3) + 0.0002 * tt + 0.1 * r.standard_normal(N)).astype(np.float32)
    return tt, s.astype(np.float64)


def make_windows(x, W):
    X = np.stack([x[i:i + W] for i in range(len(x) - W)])
    return X, x[W:]


def train_mlp_regressor(X, y, hidden=64, lr=2e-3, epochs=30, batch=64, seed=SEED):
    r = np.random.default_rng(seed)
    n_in = X.shape[1]
    k1, k2 = 1 / math.sqrt(n_in), 1 / math.sqrt(hidden)
    p = {"W1": r.uniform(-k1, k1, (hidden, n_in)), "b1": r.uniform(-k1, k1, hidden), "W2": r.uniform(-k2, k2, (1, hidden)), "b2": r.uniform(-k2, k2, 1)}
    opt = Adam(p, lr)
    for _ in range(epochs):
        for idx in np.array_split(r.permutation(len(X)), max(1, len(X) // batch)):
            xb, yb = X[idx], y[idx]
            z1 = xb @ p["W1"].T + p["b1"]; a1 = np.maximum(z1, 0); out = (a1 @ p["W2"].T + p["b2"])[:, 0]
            dout = 2 * (out - yb) / len(idx)
            g = {"W2": dout[None, :] @ a1, "b2": np.array([dout.sum()])}
            da1 = dout[:, None] * p["W2"] * (z1 > 0)
            g["W1"] = da1.T @ xb; g["b1"] = da1.sum(axis=0)
            clip_by_norm(g, 1.0); opt.step(g)
    return p


def mlp_predict(p, X):
    return (np.maximum(X @ p["W1"].T + p["b1"], 0) @ p["W2"].T + p["b2"])[:, 0]


_FORECAST = {}


def forecasting_setup():
    if _FORECAST:
        return _FORECAST
    tt, series = make_series()
    N_TRAIN, N_VAL, W = 2100, 450, 48
    train_raw = series[:N_TRAIN]
    mu, sigma = train_raw.mean(), train_raw.std()
    mu_all, sigma_all = series.mean(), series.std()
    scaled = (series - mu) / sigma
    X_tr, y_tr = make_windows(scaled[:N_TRAIN], W)
    X_va, y_va = make_windows(scaled[N_TRAIN - W:N_TRAIN + N_VAL], W)
    X_te, y_te = make_windows(scaled[N_TRAIN + N_VAL - W:], W)
    A = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    coef = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    lin = lambda X: np.hstack([X, np.ones((len(X), 1))]) @ coef
    mlp = train_mlp_regressor(X_tr, y_tr)
    print("\n== Section 7: forecasting setup (the notebook's series, seed 42) ==")
    print(f"series: {len(series)} points | train {N_TRAIN} | val {N_VAL} | test {len(series) - N_TRAIN - N_VAL} | window {W} | lcm(50, 13) = {math.lcm(50, 13)}")
    print(f"first eight values: {np.round(series[:8], 3).tolist()}")
    print(f"scaler fitted on train only: mean {mu:+.3f}, std {sigma:.3f}")
    print(f"scaler fitted on all data  : mean {mu_all:+.3f}, std {sigma_all:.3f}   <- leaks the trend (difference in mean {mu_all - mu:+.3f})")
    print(f"windows: train {X_tr.shape} | val {X_va.shape} | test {X_te.shape}")
    mae = lambda p_: float(np.mean(np.abs(p_ - y_te)) * sigma)
    p_naive, p_lin, p_mlp = X_te[:, -1], lin(X_te), mlp_predict(mlp, X_te)
    print(f"test MAE in original units: naive {mae(p_naive):.4f} | linear AR(48) {mae(p_lin):.4f} | numpy MLP {mae(p_mlp):.4f} | noise floor 0.1*sqrt(2/pi) = {0.1 * math.sqrt(2 / math.pi):.4f}")
    _FORECAST.update(dict(tt=tt, series=series, N_TRAIN=N_TRAIN, N_VAL=N_VAL, W=W, mu=mu, sigma=sigma, scaled=scaled,
                          X_te=X_te, y_te=y_te, lin=lin, mlp=mlp, p_naive=p_naive, p_lin=p_lin, p_mlp=p_mlp))
    return _FORECAST


def fig_forecast_one_step():
    F = forecasting_setup()
    mu, sigma = F["mu"], F["sigma"]
    show = slice(0, 200)
    t_test = np.arange(F["N_TRAIN"] + F["N_VAL"], len(F["series"]))[show]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t_test, F["y_te"][show] * sigma + mu, color=INK, linewidth=1.2, label="truth")
    ax.plot(t_test, F["p_naive"][show] * sigma + mu, color=SERIES[0], linewidth=1.2, alpha=0.9, label="naive (repeat last value)")
    ax.plot(t_test, F["p_mlp"][show] * sigma + mu, color=SERIES[1], linewidth=1.2, label="numpy MLP on the 48-value window")
    ax.set_xlabel("time step"); ax.set_ylabel("value"); ax.set_title("One-step-ahead forecasts on the first 200 test points")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.8); ax.legend(loc="upper left", ncol=3)
    save(fig, NB, "forecast_one_step_test_period")


def fig_forecast_rollout():
    F = forecasting_setup()
    mu, sigma, W = F["mu"], F["sigma"], F["W"]
    start = F["N_TRAIN"] + F["N_VAL"]
    H = 100
    context = F["scaled"][start - W:start]
    truth = F["series"][start:start + H]

    def rollout(predict):
        window, preds = context.copy(), []
        for _ in range(H):
            nxt = float(predict(window[None, :])[0])
            preds.append(nxt); window = np.append(window[1:], nxt)
        return np.array(preds) * sigma + mu

    roll_lin, roll_mlp = rollout(F["lin"]), rollout(lambda X: mlp_predict(F["mlp"], X))
    print("\n== Section 7.4: 100-step autoregressive rollout from the last validation window ==")
    print(f"MAE over 100 steps: linear AR(48) {np.mean(np.abs(roll_lin - truth)):.3f} | numpy MLP {np.mean(np.abs(roll_mlp - truth)):.3f}")
    for h in [1, 10, 25, 50, 100]:
        print(f"horizon {h:3d}: |error| linear {abs(roll_lin[h - 1] - truth[h - 1]):.3f} | MLP {abs(roll_mlp[h - 1] - truth[h - 1]):.3f}")
    t_roll = np.arange(start, start + H)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t_roll, truth, color=INK, linewidth=1.2, label="truth")
    ax.plot(t_roll, roll_lin, color=SERIES[0], linewidth=1.2, label="linear AR(48) rollout")
    ax.plot(t_roll, roll_mlp, color=SERIES[1], linewidth=1.2, label="numpy MLP rollout")
    ax.set_xlabel("time step"); ax.set_ylabel("value"); ax.set_title("100-step forecasts: each prediction is fed back as the next input")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.8); ax.legend(loc="upper left", ncol=3)
    save(fig, NB, "forecast_rollout_100_steps")


# ----------------------------------------------------------------------------------------------
# Section 8: padding, the mask, and how pads change a recurrent state
# ----------------------------------------------------------------------------------------------
def fig_padding_mask():
    r = np.random.default_rng(SEED)
    lengths = [3, 8, 5, 6, 2]
    L_max = max(lengths)
    padded = np.zeros((len(lengths), L_max), dtype=int)
    for i, L in enumerate(lengths):
        padded[i, :L] = r.integers(1, 21, L)
    mask = np.arange(L_max)[None, :] < np.array(lengths)[:, None]
    print("\n== Section 8.1: a padded batch of five sequences (ids 1..20, 0 = PAD) ==")
    for i in range(len(lengths)):
        print(f"seq {i}: length {lengths[i]} | padded {padded[i].tolist()} | mask {mask[i].astype(int).tolist()}")
    print(f"valid positions {int(mask.sum())} of {mask.size} ({mask.mean():.0%}); mean over time must divide by the true lengths {lengths}, not by {L_max}")
    # How pads change the state: an LSTM (4 -> 6, PyTorch-like init) reading 5 real inputs, then k zero inputs.
    p = lstm_init(4, 6, forget_bias=0.0, seed=SEED, uniform_bias=True)
    X_real = r.standard_normal((1, 5, 4))
    drift = []
    for k in range(0, 11):
        X = np.concatenate([X_real, np.zeros((1, k, 4))], axis=1)
        (h, _), _ = lstm_forward(p, X)
        if k == 0:
            h_alone = h
        drift.append(float(np.abs(h - h_alone).max()))
    print("max |h(alone) - h(after k pads)| for k = 0..10:", [f"{d:.3f}" for d in drift])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.imshow(mask.astype(float), cmap=SEQ_CMAP, vmin=-0.6, vmax=1.0, aspect="auto")
    for i in range(len(lengths)):
        for j in range(L_max):
            ax.text(j, i, str(padded[i, j]) if mask[i, j] else "PAD", ha="center", va="center", fontsize=8,
                    color=SURFACE if mask[i, j] else INK_2)
    ax.set_xticks(range(L_max)); ax.set_yticks(range(len(lengths))); ax.set_yticklabels([f"seq {i} (len {L})" for i, L in enumerate(lengths)])
    ax.set_xlabel("time step"); ax.set_title("Padded batch: dark = real token, light = PAD (mask = 0)"); ax.grid(False)
    ax = axes[1]
    ax.plot(range(11), drift, marker="o", color=SERIES[0])
    ax.annotate(f"{drift[10]:.3f} after 10 pads", (10, drift[10]), xytext=(-6, -14), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("number of PAD steps after the true end"); ax.set_ylabel("max |h(alone) - h(padded)|")
    ax.set_title("Pads keep changing an LSTM's final state")
    save(fig, NB, "padding_mask_and_state_drift")


# ----------------------------------------------------------------------------------------------
# Section 9: the bottleneck, with a toy fixed-size "superposition" memory
# ----------------------------------------------------------------------------------------------
def toy_memory_accuracy(H, lengths, n=2000, seed=SEED):
    """Store L digits in one H-dimensional vector s = sum_j P_j e(d_j) (random orthogonal P_j, unit codes e),
    read position j back as argmax_d <P_j^T s, e(d)>, and report exact-match accuracy over n sequences."""
    r = np.random.default_rng(seed)
    E = r.standard_normal((10, H)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    P = [np.linalg.qr(r.standard_normal((H, H)))[0] for _ in range(max(lengths))]
    accs = []
    for L in lengths:
        digits = r.integers(0, 10, (n, L))
        S = sum(E[digits[:, j]] @ P[j].T for j in range(L))                     # (n, H)
        pred = np.stack([np.argmax((S @ P[j]) @ E.T, axis=1) for j in range(L)], axis=1)
        accs.append(float(np.all(pred == digits, axis=1).mean()))
    return accs


def fig_seq2seq_bottleneck():
    lengths = list(range(2, 21))
    print("\n== Section 9: exact-match recall from a fixed-size vector (toy superposition memory, 2,000 sequences per length) ==")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for H, c, m in zip([32, 64, 128], SERIES[:3], ["o", "s", "^"]):
        accs = toy_memory_accuracy(H, lengths)
        print(f"H = {H:3d}: " + ", ".join(f"L={L}: {a:.3f}" for L, a in zip(lengths, accs) if L in (2, 4, 6, 8, 10, 12, 16, 20)))
        ax.plot(lengths, accs, marker=m, markersize=5, color=c, label=f"memory size H = {H}")
    ax.set_ylim(0, 1.05); ax.set_xticks(range(2, 21, 2)); ax.set_xlabel("number of digits stored in one vector (sequence length L)"); ax.set_ylabel("exact-match recall accuracy")
    ax.set_title("A fixed-size vector holds only so many digits"); ax.legend(loc="upper right")
    ax = axes[1]
    L = 6
    src = [7, 2, 9, 4, 0, 5]
    print(f"reversal example: source {src} | target {src[::-1]} | decoder input ['<sos>'] + target = {['<sos>'] + src[::-1]} | "
          f"decoder target = target + ['<eos>'] = {src[::-1] + ['<eos>']}")
    A = np.full((L, L), 0.02); A[np.arange(L), L - 1 - np.arange(L)] = 0.9
    A /= A.sum(axis=1, keepdims=True)
    print(f"idealised attention for reversing {src}: output step t reads source position L+1-t (1-indexed); row sums {np.round(A.sum(axis=1), 2).tolist()}")
    im = ax.imshow(A, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(L)); ax.set_xticklabels([str(d) for d in src]); ax.set_yticks(range(L)); ax.set_yticklabels([str(d) for d in src[::-1]])
    ax.set_xlabel("source position (input digit)"); ax.set_ylabel("output step (digit written)")
    ax.set_title("Idealised attention for reversal: an anti-diagonal"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    save(fig, NB, "seq2seq_bottleneck_toy_memory")


def main():
    gradient_checks()
    fig_order_matters()
    fig_rnn_forward_one_unit()
    fig_gradient_norm_vs_lag()
    fig_clipping()
    fig_lstm_gate_example()
    fig_lstm_cell_gradient_vs_lag()
    fig_delayed_recall()
    fig_perplexity()
    fig_temperature()
    fig_forecast_one_step()
    fig_forecast_rollout()
    fig_padding_mask()
    fig_seq2seq_bottleneck()


if __name__ == "__main__":
    main()
