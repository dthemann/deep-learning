"""Figures for the companion guide of Notebook 13 · Reinforcement Learning.

Everything here is numpy + matplotlib: a hand-built 4x4 FrozenLake MDP (the same transition table as
Gymnasium's), a numpy re-implementation of the CartPole physics, tabular Q-learning / SARSA runs, a
tabular REINFORCE run on the notebook's three-state MDP, and closed-form illustrations of the Huber
loss, the max-overestimation bias, GAE weights, the PPO clip and the RLHF KL penalty.

Run:  python3 notebooks/figures/scripts/make_figures_13.py
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

NB = "13"
np.set_printoptions(precision=4, suppress=True)


def banner(title):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)


# ----------------------------------------------------------------------------------------------
# Shared helpers: discounted returns, the 4x4 FrozenLake model, value iteration, moving average
# ----------------------------------------------------------------------------------------------
def discounted_returns(rewards, gamma):
    G, running = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def moving_average(x, w):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - w + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


LAKE = ["SFFF", "FHFH", "FFFH", "HFFG"]
ACTION_NAMES = {0: "left", 1: "down", 2: "right", 3: "up"}
ARROWS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}      # (dx, dy) in image coordinates


def build_lake(slippery=True):
    """Transition table as fixed-width arrays: nxt[s, a, k], rew[s, a, k], done[s, a, k], prob[s, a, k]."""
    nxt = np.zeros((16, 4, 3), int)
    rew = np.zeros((16, 4, 3))
    done = np.zeros((16, 4, 3), bool)
    prob = np.zeros((16, 4, 3))

    def move(r, c, a):
        if a == 0: c = max(c - 1, 0)
        elif a == 1: r = min(r + 1, 3)
        elif a == 2: c = min(c + 1, 3)
        elif a == 3: r = max(r - 1, 0)
        return r, c

    for s in range(16):
        r, c = divmod(s, 4)
        letter = LAKE[r][c]
        for a in range(4):
            if letter in "GH":                       # absorbing terminal states
                nxt[s, a] = s; prob[s, a] = [1.0, 0.0, 0.0]; done[s, a] = True
                continue
            moves = [(a - 1) % 4, a, (a + 1) % 4] if slippery else [a, a, a]
            for k, b in enumerate(moves):
                r2, c2 = move(r, c, b)
                l2 = LAKE[r2][c2]
                nxt[s, a, k] = r2 * 4 + c2
                rew[s, a, k] = float(l2 == "G")
                done[s, a, k] = l2 in "GH"
                prob[s, a, k] = 1.0 / 3.0
    return nxt, rew, done, prob


def value_iteration(model, gamma, tol=1e-8, max_iter=100_000):
    nxt, rew, done, prob = model
    V, deltas = np.zeros(16), []
    for _ in range(max_iter):
        Q = (prob * (rew + gamma * np.where(done, 0.0, V[nxt]))).sum(-1)      # (16, 4)
        V_new = Q.max(1)
        deltas.append(np.abs(V_new - V).max())
        V = V_new
        if deltas[-1] < tol:
            break
    return V, Q, Q.argmax(1), np.array(deltas)


def success_probability(model, policy, horizon=100):
    """Exact probability that `policy` (array of actions) reaches G within `horizon` steps from state 0."""
    nxt, rew, done, prob = model
    dist, succ = np.zeros(16), 0.0
    dist[0] = 1.0
    for _ in range(horizon):
        new = np.zeros(16)
        for s in range(16):
            if dist[s] == 0 or LAKE[s // 4][s % 4] in "GH":
                continue
            a = policy[s]
            for k in range(3):
                new[nxt[s, a, k]] += dist[s] * prob[s, a, k]
                succ += dist[s] * prob[s, a, k] * rew[s, a, k]
        dist = new
    return succ


def draw_lake(ax, V, policy, title, show_values=True):
    """Heat-map of V (one hue) with the greedy action as an arrow in every non-terminal cell."""
    ax.imshow(V.reshape(4, 4), cmap=SEQ_CMAP, vmin=0.0, vmax=max(float(V.max()), 1e-6))
    for s in range(16):
        r, c = divmod(s, 4)
        letter = LAKE[r][c]
        if letter == "H":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=INK_2))
            ax.text(c, r, "H", ha="center", va="center", color=SURFACE, fontsize=11, fontweight="bold")
        elif letter == "G":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=NEUTRAL_MID))
            ax.text(c, r, "G", ha="center", va="center", color=INK, fontsize=11, fontweight="bold")
        else:
            dx, dy = ARROWS[int(policy[s])]
            ax.arrow(c - 0.2 * dx, r - 0.2 * dy, 0.36 * dx, 0.36 * dy, head_width=0.14,
                     color=ORANGE, length_includes_head=True, linewidth=1.5)
            if show_values:
                ax.text(c - 0.44, r + 0.42, f"{V[s]:.2f}", fontsize=7, color=INK if V[s] < 0.5 * V.max() else SURFACE)
            if letter == "S":
                ax.text(c - 0.44, r - 0.3, "S", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title)


# ----------------------------------------------------------------------------------------------
# Section 1 · discounting
# ----------------------------------------------------------------------------------------------
def fig_discount_weights():
    banner("Figure discount_weights: the weight gamma^t of a reward t steps ahead")
    t = np.arange(0, 101)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, g in enumerate([0.5, 0.9, 0.99]):
        ax.plot(t, g ** t, color=SERIES[i], label=f"gamma = {g}  (horizon 1/(1-gamma) = {1/(1-g):.0f} steps)")
        h = 1 / (1 - g)
        ax.plot([h], [g ** h], "o", color=SERIES[i], ms=6)
    ax.annotate("at t = 1/(1-gamma) the weight is\nabout 1/e = 0.37", xy=(100, 0.99 ** 100), xytext=(55, 0.62),
                fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xlabel("steps into the future t"); ax.set_ylabel("weight gamma^t of the reward")
    ax.set_title("The discount factor sets how far ahead the agent looks"); ax.legend()
    save(fig, NB, "discount_weights")

    print("weights gamma^t at t = 0, 1, 5, 10, 50, 100:")
    for g in [0.5, 0.9, 0.99]:
        print(f"  gamma {g:4}: " + "  ".join(f"t={k}: {g**k:.4f}" for k in [0, 1, 5, 10, 50, 100]) +
              f"  | weight at t = 1/(1-gamma): {g ** (1/(1-g)):.4f}")
    print("\nworked example: rewards [0, 0, 0, 0, 0, 1], gamma = 0.9")
    G = discounted_returns(np.array([0, 0, 0, 0, 0, 1.0]), 0.9)
    for k in range(6):
        print(f"  G_{k} = 0.9^{5-k} * 1 = {G[k]:.4f}")
    print("recursion check: G_4 = R_5 + 0.9 * G_5 =", 0 + 0.9 * G[5])
    print("\nsum of gamma^t for an episode of 200 steps with reward 1 at every step:")
    for g in [0.5, 0.9, 0.99]:
        print(f"  gamma {g}: G_0 = {discounted_returns(np.ones(200), g)[0]:.3f}   (limit 1/(1-gamma) = {1/(1-g):.1f})")


# ----------------------------------------------------------------------------------------------
# Section 1 · the three-state MDP and the Bellman expectation equation
# ----------------------------------------------------------------------------------------------
def fig_bellman_three_state():
    banner("Figure bellman_three_state: exact policy evaluation vs iterating the Bellman operator")
    gamma = 0.9
    P = np.zeros((2, 3, 3)); R = np.zeros((2, 3))
    P[0] = np.eye(3)
    P[1] = np.array([[0.2, 0.8, 0.0], [0.0, 0.2, 0.8], [1.0, 0.0, 0.0]])
    R[1, 2] = 1.0
    policies = {"always move": np.array([[0.0, 1.0]] * 3), "coin flip": np.array([[0.5, 0.5]] * 3)}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    results = {}
    for i, (name, pi) in enumerate(policies.items()):
        P_pi = np.einsum("sa,ask->sk", pi, P)
        r_pi = np.einsum("sa,as->s", pi, R)
        V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
        Vk, errs = np.zeros(3), []
        for _ in range(50):
            Vk = r_pi + gamma * P_pi @ Vk
            errs.append(np.abs(Vk - V).max())
        errs = np.array(errs)
        results[name] = (P_pi, r_pi, V, errs)
        ax.semilogy(np.arange(1, 51), errs, color=SERIES[i], label=f"{name}: max |V_k - V^pi|")
        print(f"{name}: P_pi =\n{P_pi}\n   r_pi = {r_pi}   V^pi = {V}")
        print(f"   error after sweep 1, 10, 50: {errs[0]:.4f}, {errs[9]:.4f}, {errs[49]:.4f}; "
              f"ratio error[k+1]/error[k] at k = 2, 10, 49: {errs[2]/errs[1]:.4f}, {errs[10]/errs[9]:.4f}, {errs[49]/errs[48]:.4f}")
    errs_move = results["always move"][3]
    ax.semilogy(np.arange(1, 51), errs_move[0] * gamma ** np.arange(50), ls="--", color=INK_2,
                label="contraction bound: error_1 * gamma^(k-1)")
    ax.set_xlabel("sweep k"); ax.set_ylabel("max-norm error to the exact solution")
    ax.set_title("Policy evaluation by iterating the Bellman operator (gamma = 0.9)"); ax.legend()
    save(fig, NB, "bellman_three_state")

    V = results["always move"][2]
    print("\nworked example (always move): the three Bellman equations")
    print(f"  V(2) = 1 + 0.9 * V(0)                 -> {1 + 0.9 * V[0]:.4f}  (V(2) = {V[2]:.4f})")
    print(f"  V(1) = 0.9 * (0.2 V(1) + 0.8 V(2))     -> {0.9 * (0.2 * V[1] + 0.8 * V[2]):.4f}  (V(1) = {V[1]:.4f})")
    print(f"  V(0) = 0.9 * (0.2 V(0) + 0.8 V(1))     -> {0.9 * (0.2 * V[0] + 0.8 * V[1]):.4f}  (V(0) = {V[0]:.4f})")
    print("  first three sweeps from V = 0:")
    P_pi, r_pi = results["always move"][0], results["always move"][1]
    Vk = np.zeros(3)
    for k in range(3):
        Vk = r_pi + gamma * P_pi @ Vk
        print(f"    V_{k+1} = {Vk}")
    Vc = results["coin flip"][2]
    print(f"coin flip values: {Vc}  (all lower than always move: {np.all(Vc < V)})")


# ----------------------------------------------------------------------------------------------
# Section 2 · CartPole physics in numpy, one episode and the random baseline
# ----------------------------------------------------------------------------------------------
def cartpole_step(state, action):
    g, mc, mp, l, F, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x, xd, th, thd = state
    force = F if action == 1 else -F
    tm, pml = mc + mp, mp * l
    c, s = math.cos(th), math.sin(th)
    temp = (force + pml * thd ** 2 * s) / tm
    thacc = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c ** 2 / tm))
    xacc = temp - pml * thacc * c / tm
    x, xd = x + tau * xd, xd + tau * xacc
    th, thd = th + tau * thd, thd + tau * thacc
    new = np.array([x, xd, th, thd])
    terminated = abs(x) > 2.4 or abs(th) > 12 * 2 * math.pi / 360
    return new, 1.0, terminated


def cartpole_random_episode(rng, max_steps=500, record=False):
    state = rng.uniform(-0.05, 0.05, size=4)
    trace = [state.copy()]
    log = []
    for t in range(max_steps):
        action = int(rng.integers(2))
        state, r, term = cartpole_step(state, action)
        trunc = (t + 1 >= max_steps) and not term
        if record:
            trace.append(state.copy()); log.append((t, action, r, term, trunc, state.copy()))
        if term or trunc:
            return t + 1, np.array(trace), log
    return max_steps, np.array(trace), log


def fig_cartpole_random_baseline():
    banner("Figure cartpole_random_baseline: one random episode and the return distribution of 300 episodes")
    r0 = np.random.default_rng(42)
    length, trace, log = cartpole_random_episode(r0, record=True)
    print(f"one random episode (seed 42): length {length} steps; first observation {trace[0]}")
    print("  t  action  reward  terminated  truncated  obs = [x, x_dot, theta, theta_dot]")
    for (t, a, r, term, trunc, obs) in log[:5]:
        print(f"  {t}  {a}       {r:.0f}       {term!s:5}       {trunc!s:5}      {np.round(obs, 3)}")
    print(f"  last step {log[-1][0]}: obs {np.round(log[-1][5], 3)}, terminated {log[-1][3]}  "
          f"(|theta| > {12 * 2 * math.pi / 360:.4f} rad or |x| > 2.4 ends the episode)")

    r1 = np.random.default_rng(0)
    returns = np.array([cartpole_random_episode(r1)[0] for _ in range(300)])
    print(f"\nrandom policy, 300 episodes: mean return {returns.mean():.1f} +- {returns.std():.1f}, "
          f"min {returns.min()}, max {returns.max()}, median {np.median(returns):.0f}, "
          f"fraction above 60: {(returns > 60).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    t = np.arange(len(trace))
    names = ["x (cart position, m)", "x_dot (m/s)", "theta (pole angle, rad)", "theta_dot (rad/s)"]
    for i in range(4):
        ax.plot(t, trace[:, i], color=SERIES[i], label=names[i])
    ax.axhline(12 * 2 * math.pi / 360, color=INK_2, ls=":", lw=1)
    ax.axhline(-12 * 2 * math.pi / 360, color=INK_2, ls=":", lw=1)
    ax.text(0.3, 12 * 2 * math.pi / 360 + 0.03, "+-12 degrees: pole has fallen", fontsize=8, color=INK_2)
    ax.set_xlabel("time step t (0.02 s each)"); ax.set_ylabel("observation value")
    ax.set_title(f"One random episode ({length} steps)"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(returns, bins=np.arange(5, 85, 4), color=BLUE, edgecolor=SURFACE)
    ax.axvline(returns.mean(), color=ORANGE, ls="--", lw=1.5)
    ax.text(returns.mean() + 2, ax.get_ylim()[1] * 0.9, f"mean {returns.mean():.1f}", color=ORANGE, fontsize=9)
    ax.set_xlabel("episode return (= steps the pole stayed up)"); ax.set_ylabel("episodes (of 300)")
    ax.set_title("Return distribution of a random policy")
    save(fig, NB, "cartpole_random_baseline")


# ----------------------------------------------------------------------------------------------
# Section 3 · value iteration on the slippery lake
# ----------------------------------------------------------------------------------------------
def fig_frozenlake_value_iteration():
    banner("Figure frozenlake_value_iteration: V* and the greedy policy on slippery FrozenLake, convergence")
    model = build_lake(True)
    nxt, rew, done, prob = model
    print("transitions from state 0 (corner) for action 2 (right):")
    for k in range(3):
        print(f"   prob {prob[0, 2, k]:.3f} -> state {nxt[0, 2, k]:2d}, reward {rew[0, 2, k]:.0f}, terminal {done[0, 2, k]}")
    print("transitions from state 14 for action 2 (right):")
    for k in range(3):
        print(f"   prob {prob[14, 2, k]:.3f} -> state {nxt[14, 2, k]:2d}, reward {rew[14, 2, k]:.0f}, terminal {done[14, 2, k]}")
    out = {}
    for gamma in [0.99, 1.0, 0.9]:
        V, Q, pi, deltas = value_iteration(model, gamma)
        out[gamma] = (V, Q, pi, deltas)
        arrows = "\n".join("   " + "".join({0: "<", 1: "v", 2: ">", 3: "^"}[pi[r * 4 + c]] if LAKE[r][c] in "SF" else LAKE[r][c]
                                            for c in range(4)) for r in range(4))
        print(f"\ngamma = {gamma}: converged in {len(deltas)} sweeps, V*(start) = {V[0]:.4f}")
        print("  V* =\n" + "\n".join("   " + " ".join(f"{v:.3f}" for v in V[r*4:(r+1)*4]) for r in range(4)))
        print("  greedy policy:\n" + arrows)
        print(f"  success probability of the greedy policy within 100 steps: {success_probability(model, pi, 100):.4f}; "
              f"within 10000 steps: {success_probability(model, pi, 10000):.4f}")
    V, Q, pi, deltas = out[0.99]
    print("\nworked example: one backup of state 14 with V* (gamma 0.99), action right:")
    terms = [f"1/3*({rew[14,2,k]:.0f} + {'0' if done[14,2,k] else f'0.99*{V[nxt[14,2,k]]:.4f}'})" for k in range(3)]
    print("   Q*(14, right) = " + " + ".join(terms) + f" = {Q[14, 2]:.4f}")
    print("   Q*(14, a) for a = left, down, right, up:", Q[14], "-> greedy action:", ACTION_NAMES[int(pi[14])])
    print("   Q*(0, a) for a = left, down, right, up: ", Q[0], "-> greedy action:", ACTION_NAMES[int(pi[0])])
    print(f"   error bound after the last sweep: gamma/(1-gamma) * delta = {0.99/0.01 * deltas[-1]:.2e}")
    rand_pi = None
    # random policy success probability (uniform over actions): mix the four actions
    dist, succ = np.zeros(16), 0.0
    dist[0] = 1.0
    for _ in range(100):
        new = np.zeros(16)
        for s in range(16):
            if dist[s] == 0 or LAKE[s // 4][s % 4] in "GH":
                continue
            for a in range(4):
                for k in range(3):
                    new[nxt[s, a, k]] += dist[s] * 0.25 * prob[s, a, k]
                    succ += dist[s] * 0.25 * prob[s, a, k] * rew[s, a, k]
        dist = new
    print(f"   uniformly random policy: success probability within 100 steps = {succ:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].semilogy(np.arange(1, len(deltas) + 1), deltas, color=BLUE)
    axes[0].set_xlabel("sweep k"); axes[0].set_ylabel("max |V_k+1 - V_k|")
    axes[0].set_title(f"Value iteration converges in {len(deltas)} sweeps (gamma = 0.99)")
    draw_lake(axes[1], V, pi, "V* and greedy policy (arrows), gamma = 0.99")
    save(fig, NB, "frozenlake_value_iteration")
    return out


# ----------------------------------------------------------------------------------------------
# Section 3 · Q-learning vs SARSA (real numpy runs, several seeds)
# ----------------------------------------------------------------------------------------------
def tabular_td(model, n_episodes, method, alpha=0.1, gamma=0.99, eps_fn=None, seed=0, max_steps=100):
    nxt, rew, done, prob = model
    cum = prob.cumsum(-1)
    r = np.random.default_rng(seed)
    Q = np.zeros((16, 4))
    rets = np.zeros(n_episodes)
    decay = max(1, n_episodes // 2)
    eps_fn = eps_fn or (lambda ep: max(0.05, 1.0 - 0.95 * ep / decay))
    for ep in range(n_episodes):
        eps = eps_fn(ep)
        u_eps, u_tr = r.random(max_steps + 1), r.random(max_steps + 1)
        rand_a = r.integers(4, size=max_steps + 1)
        tie = r.random((max_steps + 1, 4)) * 1e-9              # random tie-breaking among equal Q-values

        def act(s, i):
            if u_eps[i] < eps:
                return int(rand_a[i])
            return int(np.argmax(Q[s] + tie[i]))

        s = 0
        a = act(s, 0)
        for t in range(max_steps):
            k = min(int(np.searchsorted(cum[s, a], u_tr[t])), 2)
            s2, rw, d = int(nxt[s, a, k]), rew[s, a, k], done[s, a, k]
            a2 = act(s2, t + 1)
            if d:
                target = rw
            elif method == "q_learning":
                target = rw + gamma * Q[s2].max()
            else:
                target = rw + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            rets[ep] += rw
            s, a = s2, a2
            if d:
                break
    return Q, rets


def fig_frozenlake_q_vs_sarsa(vi_out):
    banner("Figure frozenlake_q_vs_sarsa: learning curves of Q-learning and SARSA (4 seeds, 6000 episodes)")
    model = build_lake(True)
    V_star, Q_star, pi_star, _ = vi_out[0.99]
    n_episodes, seeds = 6000, [0, 1, 2, 3]
    opt_succ = success_probability(model, pi_star, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    learned = {}
    for i, method in enumerate(["q_learning", "sarsa"]):
        curves, greedy_succ, Qs = [], [], []
        for sd in seeds:
            Q, rets = tabular_td(model, n_episodes, method, seed=sd)
            curves.append(moving_average(rets, 200))
            greedy_succ.append(success_probability(model, Q.argmax(1), 100))
            Qs.append(Q)
        curves = np.array(curves)
        learned[method] = (Qs[0], greedy_succ)
        ax.plot(np.arange(n_episodes), curves.mean(0), color=SERIES[i], label=f"{method} (mean of {len(seeds)} seeds)")
        ax.fill_between(np.arange(n_episodes), curves.min(0), curves.max(0), color=SERIES[i], alpha=0.15, lw=0)
        print(f"{method}: training success rate over the last 500 episodes per seed: "
              f"{[round(float(c[-500:].mean()), 3) for c in curves]}")
        print(f"   exact success probability of the greedy policy (100-step limit) per seed: {np.round(greedy_succ, 3)}; "
              f"mean {np.mean(greedy_succ):.3f}")
        print(f"   greedy policy of seed 0 differs from value iteration in "
              f"{int((Qs[0].argmax(1) != pi_star)[[s for s in range(16) if LAKE[s//4][s%4] in 'SF']].sum())} of 11 non-terminal states")
    ax.axhline(opt_succ, color=INK_2, ls="--", lw=1.2, label=f"optimal policy, exploration off ({opt_succ:.3f})")
    ax.set_xlabel("training episode"); ax.set_ylabel("success rate (moving average of 200 episodes)")
    ax.set_title("Q-learning vs SARSA on slippery FrozenLake (epsilon 1.0 -> 0.05 over the first half)")
    ax.legend(loc="upper left")
    save(fig, NB, "frozenlake_q_vs_sarsa")

    Q = learned["q_learning"][0]
    print("\nworked example: Q-table of the Q-learning run (seed 0), rows = states 13, 14; columns = left, down, right, up")
    print("   Q[13] =", Q[13], " Q[14] =", Q[14])
    alpha, gamma = 0.1, 0.99
    print("one terminal update: s = 14, a = right, slipped onto the goal (r = 1, terminal), old Q[14, right] = 0.0")
    print(f"   target = r = 1.0;  new Q = 0.0 + {alpha} * (1.0 - 0.0) = {alpha * 1.0:.3f}")
    print("one non-terminal update from the learned table: s = 13, a = right, landed in s' = 14 with r = 0")
    q_target = gamma * Q[14].max()
    a_next = int(np.argmin(Q[14]))      # suppose the exploring agent picked the worst action next
    s_target = gamma * Q[14, a_next]
    print(f"   Q-learning target = 0 + {gamma} * max_a Q[14, a] = {gamma} * {Q[14].max():.4f} = {q_target:.4f}")
    print(f"   SARSA target if the next (exploratory) action is {ACTION_NAMES[a_next]}: 0 + {gamma} * Q[14, {ACTION_NAMES[a_next]}] "
          f"= {gamma} * {Q[14, a_next]:.4f} = {s_target:.4f}")
    print(f"   old Q[13, right] = {Q[13, 2]:.4f}; Q-learning update -> {Q[13,2] + alpha*(q_target - Q[13,2]):.4f}; "
          f"SARSA update -> {Q[13,2] + alpha*(s_target - Q[13,2]):.4f}")
    return learned


# ----------------------------------------------------------------------------------------------
# Section 3 (and Exercise 2) · epsilon schedules
# ----------------------------------------------------------------------------------------------
def fig_epsilon_schedules():
    banner("Figure epsilon_schedules: four exploration schedules over 8000 episodes")
    N = 8000
    tau = N / 5
    ep = np.arange(N)
    schedules = {
        "linear 1.0 -> 0.05 over the first half": np.maximum(0.05, 1.0 - 0.95 * ep / (N // 2)),
        f"exponential exp(-k/tau), tau = N/5, floor 0.05": np.maximum(0.05, np.exp(-ep / tau)),
        "constant 0.1": np.full(N, 0.1),
        "constant 0.5": np.full(N, 0.5),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, eps) in enumerate(schedules.items()):
        ax.plot(ep, eps, color=SERIES[i], label=name)
        print(f"{name}: average epsilon over training = {eps.mean():.4f}; epsilon at episode 1000, 3200, 4800: "
              f"{eps[1000]:.3f}, {eps[3200]:.3f}, {eps[4800]:.3f}")
    ax.annotate("exp(-2) = 0.135 at 2 tau", xy=(2 * tau, math.exp(-2)), xytext=(3000, 0.45), fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xlabel("training episode k"); ax.set_ylabel("epsilon (probability of a random action)")
    ax.set_title("Exploration schedules compared (N = 8000 episodes)"); ax.legend()
    save(fig, NB, "epsilon_schedules")
    print(f"exp(-2) = {math.exp(-2):.4f}, exp(-3) = {math.exp(-3):.4f}; 2 tau = {2*tau:.0f} episodes = 40% of the budget")


# ----------------------------------------------------------------------------------------------
# Section 4 · REINFORCE on the three-state MDP: learning curves and gradient variance
# ----------------------------------------------------------------------------------------------
P3 = np.zeros((2, 3, 3)); R3 = np.zeros((2, 3))
P3[0] = np.eye(3)
P3[1] = np.array([[0.2, 0.8, 0.0], [0.0, 0.2, 0.8], [1.0, 0.0, 0.0]])
R3[1, 2] = 1.0


def softmax_rows(theta):
    e = np.exp(theta - theta.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def episode3(theta, r, T=20):
    """One T-step episode of the three-state MDP under the softmax policy theta[s, a]."""
    pi = softmax_rows(theta)
    s, S, A, Rw = 0, [], [], []
    u_a, u_s = r.random(T), r.random(T)
    for t in range(T):
        a = int(u_a[t] < pi[s, 1])
        S.append(s); A.append(a); Rw.append(R3[a, s])
        s = int(np.searchsorted(P3[a, s].cumsum(), u_s[t]))
    return np.array(S), np.array(A), np.array(Rw)


def policy_grad_episode(theta, S, A, w):
    pi = softmax_rows(theta)
    g = np.zeros_like(theta)
    for s, a, wt in zip(S, A, w):
        e = np.zeros(2); e[a] = 1.0
        g[s] += (e - pi[s]) * wt
    return g


def reinforce3(variant, n_ep=300, lr=0.1, gamma=0.9, seed=0, T=20):
    r = np.random.default_rng(seed)
    theta, base, cnt = np.zeros((3, 2)), np.zeros(3), np.zeros(3)
    out = np.zeros(n_ep)
    for ep in range(n_ep):
        S, A, Rw = episode3(theta, r, T)
        G = discounted_returns(Rw, gamma)
        if variant == "full return":
            w = np.full(T, G[0])
        elif variant == "reward-to-go":
            w = G
        else:
            w = G - base[S]
            for s, g in zip(S, G):
                cnt[s] += 1; base[s] += (g - base[s]) / cnt[s]
        theta += lr * policy_grad_episode(theta, S, A, w) / T
        out[ep] = Rw.sum()
    return out, theta


def fig_reinforce_estimators():
    banner("Figure reinforce_estimators: three REINFORCE estimators on the three-state MDP")
    gamma, T = 0.9, 20
    variants = ["full return", "reward-to-go", "reward-to-go + baseline"]
    # exact values of the stationary reward rate for the two reference policies
    for name, pi in [("always move", np.array([[0.0, 1.0]] * 3)), ("coin flip", np.array([[0.5, 0.5]] * 3))]:
        P_pi = np.einsum("sa,ask->sk", pi, P3)
        w, vecs = np.linalg.eig(P_pi.T)
        d = np.real(vecs[:, np.argmin(abs(w - 1))]); d = d / d.sum()
        rate = d @ np.einsum("sa,as->s", pi, R3)
        print(f"{name}: stationary distribution {d}, reward per step {rate:.4f}, i.e. about {rate*T:.2f} per {T}-step episode")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    for i, variant in enumerate(variants):
        curves = np.array([reinforce3(variant, seed=sd)[0] for sd in range(20)])
        m = moving_average(curves.mean(0), 10)
        ax.plot(np.arange(len(m)), m, color=SERIES[i], label=variant)
        print(f"{variant}: mean reward per episode (20 seeds), episodes 1-20: {curves[:, :20].mean():.2f}, "
              f"141-160: {curves[:, 140:160].mean():.2f}, 281-300: {curves[:, -20:].mean():.2f}")
    ax.set_xlabel("training episode"); ax.set_ylabel("reward collected per 20-step episode (mean of 20 seeds, MA 10)")
    ax.set_title("REINFORCE learning curves (tabular softmax policy)"); ax.legend(loc="lower right")

    # gradient variance at a fixed, half-trained policy; baseline = exact V^pi(s)
    theta = np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]])
    pi = softmax_rows(theta)
    P_pi = np.einsum("sa,ask->sk", pi, P3); r_pi = np.einsum("sa,as->s", pi, R3)
    V_pi = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    print(f"\nfixed probe policy pi(move | s) = {pi[0, 1]:.3f} in every state; exact V^pi = {V_pi}")
    r = np.random.default_rng(123)
    grads = {v: [] for v in variants}
    n_ep = 4000
    for _ in range(n_ep):
        S, A, Rw = episode3(theta, r, T)
        G = discounted_returns(Rw, gamma)
        ws = {"full return": np.full(T, G[0]), "reward-to-go": G, "reward-to-go + baseline": G - V_pi[S]}
        for v in variants:
            grads[v].append(policy_grad_episode(theta, S, A, ws[v]).ravel())
    stats = []
    for v in variants:
        g = np.array(grads[v])
        mean_norm, total_var = np.linalg.norm(g.mean(0)), g.var(0).sum()
        stats.append((mean_norm, total_var, math.sqrt(total_var) / mean_norm))
        print(f"{v:26s}: ||mean grad|| = {mean_norm:.4f}, total variance = {total_var:.3f}, noise/signal = {math.sqrt(total_var)/mean_norm:.2f}")
    ax = axes[1]
    bar(ax, ["full\nreturn", "reward-\nto-go", "reward-to-go\n+ baseline"], [s[1] for s in stats], color=BLUE, label_fmt="{:.2f}")
    ax.set_ylabel("total variance of the one-episode gradient estimate")
    ax.set_title(f"Gradient noise at a fixed policy ({n_ep} episodes)")
    save(fig, NB, "reinforce_estimators")
    return stats


# ----------------------------------------------------------------------------------------------
# Section 4 · baseline and variance on a three-arm bandit (closed form + Monte Carlo)
# ----------------------------------------------------------------------------------------------
def fig_baseline_variance_bandit():
    banner("Figure baseline_variance_bandit: variance of the REINFORCE estimate as a function of the baseline")
    theta = np.array([0.5, -0.2, 0.1]); q = np.array([1.0, 2.0, 0.5]); sigma = 1.0
    pi = np.exp(theta) / np.exp(theta).sum()
    grads = np.eye(3) - pi[None, :]                      # grad_theta log pi(a) for each arm a (rows)
    true_grad = (pi[:, None] * grads * q[:, None]).sum(0)
    V_pi = pi @ q
    print(f"policy pi = {pi}, mean rewards q = {q}, reward noise sd = {sigma}, V^pi = {V_pi:.4f}")
    print(f"true gradient = {true_grad}")

    def total_var(b):
        second = sum(pi[a] * np.sum(grads[a] ** 2) * ((q[a] - b) ** 2 + sigma ** 2) for a in range(3))
        return second - np.sum(true_grad ** 2)

    b_opt = sum(pi[a] * np.sum(grads[a] ** 2) * q[a] for a in range(3)) / sum(pi[a] * np.sum(grads[a] ** 2) for a in range(3))
    bs = np.linspace(-2, 5, 141)
    r = np.random.default_rng(0)
    N = 100_000
    a = r.choice(3, size=N, p=pi); Rw = q[a] + sigma * r.normal(size=N)
    mc_b = np.array([-1.0, 0.0, V_pi, 2.0, 3.0, 4.0])
    mc_var, mc_mean = [], []
    for b in mc_b:
        g = grads[a] * (Rw - b)[:, None]
        mc_var.append(g.var(0).sum()); mc_mean.append(g.mean(0))
    print("baseline b -> exact total variance (and Monte Carlo estimate from 100000 pulls, mean gradient):")
    for b, v, m in zip(mc_b, mc_var, mc_mean):
        print(f"   b = {b:6.3f}: exact {total_var(b):.4f}, MC {v:.4f}, MC mean gradient {m}")
    print(f"variance-optimal baseline b* = {b_opt:.4f} (close to V^pi = {V_pi:.4f}); variance at b = 0: {total_var(0):.4f}, "
          f"at b = V^pi: {total_var(V_pi):.4f}, at b*: {total_var(b_opt):.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(bs, [total_var(b) for b in bs], color=BLUE, label="exact variance")
    ax.plot(mc_b, mc_var, "o", color=ORANGE, ms=6, label="Monte Carlo (100000 pulls)")
    ax.axvline(V_pi, color=INK_2, ls="--", lw=1)
    ax.text(V_pi + 0.08, total_var(4.0) * 0.8, f"b = V^pi = {V_pi:.2f}", fontsize=8, color=INK_2)
    ax.set_xlabel("baseline b subtracted from the reward"); ax.set_ylabel("total variance of the gradient estimate")
    ax.set_title("A baseline changes the variance, not the mean (3-arm bandit)"); ax.legend()
    save(fig, NB, "baseline_variance_bandit")


# ----------------------------------------------------------------------------------------------
# Section 5 · Huber loss, and the overestimation bias of the max
# ----------------------------------------------------------------------------------------------
def fig_huber_loss():
    banner("Figure huber_loss: Huber (delta = 1) vs squared loss and their gradients")
    u = np.linspace(-4, 4, 401)
    sq = 0.5 * u ** 2
    hub = np.where(np.abs(u) <= 1, 0.5 * u ** 2, np.abs(u) - 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(u, sq, color=BLUE, label="squared error  u^2 / 2")
    axes[0].plot(u, hub, color=ORANGE, label="Huber, delta = 1")
    axes[0].set_xlabel("TD error u = target - Q(s, a)"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(u, u, color=BLUE, label="squared: gradient u")
    axes[1].plot(u, np.clip(u, -1, 1), color=ORANGE, label="Huber: gradient clipped to [-1, 1]")
    axes[1].set_xlabel("TD error u"); axes[1].set_ylabel("d loss / d u"); axes[1].set_title("Gradient"); axes[1].legend()
    save(fig, NB, "huber_loss")
    for val in [0.5, 1.0, 2.0, 4.0]:
        h = 0.5 * val ** 2 if abs(val) <= 1 else abs(val) - 0.5
        print(f"u = {val}: squared/2 = {0.5*val**2:.3f}, Huber = {h:.3f}, gradient squared = {val:.1f}, Huber = {np.clip(val, -1, 1):.1f}")
    print("PyTorch check: smooth_l1_loss(beta = 1) = 0.5 u^2 if |u| < 1 else |u| - 0.5, identical to Huber with delta = 1")

    print("\nworked example: DQN and Double DQN targets for three transitions (gamma = 0.99)")
    gamma = 0.99
    rewards = np.array([1.0, 1.0, 1.0]); dones = np.array([0.0, 0.0, 1.0])
    q_online_next = np.array([[31.2, 30.1], [24.9, 25.4], [3.0, 2.5]])
    q_target_next = np.array([[30.4, 30.9], [25.1, 24.6], [2.8, 2.9]])
    q_sa = np.array([30.0, 26.0, 1.4])
    print("   transition  r  done  Q_online(s',.)   Q_target(s',.)   Q(s,a)")
    for i in range(3):
        print(f"   {i+1}           {rewards[i]:.0f}  {dones[i]:.0f}     {q_online_next[i]}     {q_target_next[i]}     {q_sa[i]}")
    y_dqn = rewards + gamma * (1 - dones) * q_target_next.max(1)
    sel = q_online_next.argmax(1)
    y_ddqn = rewards + gamma * (1 - dones) * q_target_next[np.arange(3), sel]
    for i in range(3):
        print(f"   transition {i+1}: DQN target = {rewards[i]:.0f} + 0.99 * {1-dones[i]:.0f} * max {q_target_next[i]} = {y_dqn[i]:.4f}; "
              f"Double DQN: online argmax = action {sel[i]}, target-net value {q_target_next[i, sel[i]]} -> {y_ddqn[i]:.4f}")
    td = y_dqn - q_sa
    hub = np.where(np.abs(td) <= 1, 0.5 * td ** 2, np.abs(td) - 0.5)
    print(f"   DQN TD errors y - Q(s,a) = {td}; Huber losses = {hub}; mean loss = {hub.mean():.4f}")
    print(f"   target-network trick: the targets use Q_target, which is refreshed only every C = 500 steps")


def fig_max_overestimation():
    banner("Figure max_overestimation: E[max of noisy estimates] - true max, single vs double estimator")
    r = np.random.default_rng(0)
    n_actions_list = [2, 3, 5, 10, 20]
    sigma, n_mc = 1.0, 200_000
    bias_single, bias_double = [], []
    print(f"all true Q*(s', a) equal to 10, estimate noise sd = {sigma}, {n_mc} Monte Carlo draws")
    for nA in n_actions_list:
        q_true = np.full(nA, 10.0)
        noiseA = r.normal(0, sigma, size=(n_mc, nA))
        noiseB = r.normal(0, sigma, size=(n_mc, nA))
        single = (q_true + noiseA).max(1)                                  # choose and evaluate with the same noisy copy
        sel = (q_true + noiseA).argmax(1)
        double = (q_true + noiseB)[np.arange(n_mc), sel]                   # choose with A, evaluate with B
        bias_single.append(single.mean() - 10.0); bias_double.append(double.mean() - 10.0)
        print(f"   {nA:2d} actions: single-estimator bias {bias_single[-1]:+.4f}, double-estimator bias {bias_double[-1]:+.4f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(n_actions_list, bias_single, "o-", color=BLUE, ms=6, label="max over one noisy estimate (DQN target)")
    ax.plot(n_actions_list, bias_double, "o-", color=ORANGE, ms=6, label="select with one estimate, evaluate with another (Double DQN)")
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel("number of actions |A|"); ax.set_ylabel("E[estimated max] - true max (reward units)")
    ax.set_title("Why the max over-estimates, and why decoupling fixes it"); ax.legend()
    save(fig, NB, "max_overestimation")
    print(f"   two actions, sd 1: theory E[max of two N(0,1)] = 1/sqrt(pi) = {1/math.sqrt(math.pi):.4f}")


# ----------------------------------------------------------------------------------------------
# Section 6 · GAE weights, a worked GAE example, and the PPO clipped objective
# ----------------------------------------------------------------------------------------------
def fig_gae_weights():
    banner("Figure gae_weights: weight (gamma*lambda)^l on the TD error l steps ahead")
    gamma = 0.99
    l = np.arange(0, 41)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, lam in enumerate([0.0, 0.5, 0.95, 1.0]):
        w = (gamma * lam) ** l
        ax.plot(l, w, "o-", color=SERIES[i], ms=4, lw=1.5, label=f"lambda = {lam}")
        print(f"lambda = {lam}: weights at l = 0, 1, 2, 5, 10, 20, 40: " + ", ".join(f"{(gamma*lam)**k:.4f}" for k in [0, 1, 2, 5, 10, 20, 40]) +
              f"; sum over l < 40: {w.sum():.3f}" + ("" if lam == 0 else f"; effective length 1/(1-gamma*lambda) = {1/(1-gamma*lam):.1f}" if lam < 1 else ""))
    ax.set_xlabel("steps ahead l"); ax.set_ylabel("weight of delta_(t+l) in A_t")
    ax.set_title("GAE: from the one-step TD error (lambda = 0) to the Monte Carlo advantage (lambda = 1), gamma = 0.99")
    ax.legend()
    save(fig, NB, "gae_weights")

    print("\nworked example: a 4-step rollout, gamma = 0.99")
    rewards = np.array([1.0, 1.0, 1.0, 1.0]); values = np.array([20.0, 21.0, 19.0, 22.0]); last_value = 23.0
    dones = np.array([0.0, 0.0, 0.0, 0.0])
    print(f"   rewards {rewards}, V(s_t) {values}, V(s_4) = {last_value}, no episode end")
    for lam in [0.95, 0.0, 1.0]:
        T = len(rewards); adv = np.zeros(T); last = 0.0; deltas = np.zeros(T)
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else values[t + 1]
            deltas[t] = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
            last = deltas[t] + gamma * lam * (1 - dones[t]) * last
            adv[t] = last
        if lam == 0.95:
            print(f"   TD errors delta_t = r + 0.99 V(s_t+1) - V(s_t): {np.round(deltas, 4)}")
            print(f"   e.g. delta_3 = 1 + 0.99 * 23.0 - 22.0 = {1 + 0.99*23 - 22:.4f}; delta_2 = 1 + 0.99 * 22.0 - 19.0 = {1 + 0.99*22 - 19:.4f}")
        print(f"   lambda = {lam}: A_t = {np.round(adv, 4)}, value targets A_t + V(s_t) = {np.round(adv + values, 4)}")
        if lam == 0.95:
            print(f"      check A_2 = delta_2 + 0.99 * 0.95 * A_3 = {deltas[2]:.4f} + {0.99*0.95:.4f} * {adv[3]:.4f} = {deltas[2] + 0.9405*adv[3]:.4f}")
    print("   with a terminal flag after step 2 (dones = [0, 0, 1, 0]) the recursion restarts: ")
    dones2 = np.array([0.0, 0.0, 1.0, 0.0]); T = 4; adv = np.zeros(T); last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        d = rewards[t] + gamma * nv * (1 - dones2[t]) - values[t]
        last = d + gamma * 0.95 * (1 - dones2[t]) * last
        adv[t] = last
    print(f"      lambda = 0.95: A_t = {np.round(adv, 4)}  (A_2 = r_2 - V(s_2) = {1 - 19.0:.1f}: no bootstrap past a terminal state)")


def fig_ppo_clip_objective():
    banner("Figure ppo_clip_objective: the clipped surrogate as a function of the probability ratio")
    eps = 0.2
    rho = np.linspace(0.4, 1.6, 241)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, A in zip(axes, [1.0, -1.0]):
        unclipped = rho * A
        clipped = np.minimum(rho * A, np.clip(rho, 1 - eps, 1 + eps) * A)
        ax.plot(rho, unclipped, color=BLUE, ls=":", label="unclipped  rho * A")
        ax.plot(rho, clipped, color=ORANGE, label="PPO objective  min(rho A, clip(rho) A)")
        ax.axvline(1 - eps, color=AXIS, lw=1); ax.axvline(1 + eps, color=AXIS, lw=1)
        ax.set_xlabel("probability ratio rho = pi_new / pi_old"); ax.set_ylabel("objective per sample")
        ax.set_title(f"advantage A = {A:+.0f}: " + ("gradient is zero beyond rho = 1.2" if A > 0 else "gradient is zero below rho = 0.8"))
        ax.legend(loc="upper left" if A > 0 else "lower left")
    save(fig, NB, "ppo_clip_objective")
    print("objective values min(rho*A, clip(rho, 0.8, 1.2)*A):")
    for A in [1.0, -1.0]:
        print(f"   A = {A:+.0f}: " + ", ".join(f"rho={rh}: {min(rh*A, np.clip(rh, 0.8, 1.2)*A):+.2f}" for rh in [0.5, 0.8, 1.0, 1.2, 1.5]))
    print("\nworked example: one action with pi_old = 0.40; after an update pi_new = 0.52")
    old, new = 0.40, 0.52
    rho_ex = new / old
    print(f"   ratio = {new}/{old} = {rho_ex:.3f} (also exp(log {new} - log {old}) = {math.exp(math.log(new) - math.log(old)):.3f})")
    for A in [1.5, -1.5]:
        val = min(rho_ex * A, np.clip(rho_ex, 0.8, 1.2) * A)
        print(f"   A = {A:+.1f}: unclipped {rho_ex*A:+.3f}, clipped {np.clip(rho_ex, 0.8, 1.2)*A:+.3f}, objective = min = {val:+.3f}"
              + ("  (clip active: no further gradient)" if A > 0 else "  (ratio moved the wrong way: unclipped value kept, gradient still active)"))
    print("   entropy of a two-action policy: uniform (0.5, 0.5) -> ln 2 = %.4f nats; (0.9, 0.1) -> %.4f; (0.99, 0.01) -> %.4f"
          % (math.log(2), -(0.9 * math.log(0.9) + 0.1 * math.log(0.1)), -(0.99 * math.log(0.99) + 0.01 * math.log(0.01))))
    print(f"   entropy bonus per step with c_e = 0.1 at a uniform policy: 0.1 * ln 2 = {0.1*math.log(2):.4f}")


# ----------------------------------------------------------------------------------------------
# Section 7 · RLHF: Bradley-Terry reward model and the KL-regularized optimum
# ----------------------------------------------------------------------------------------------
def fig_rlhf_kl_beta():
    banner("Figure rlhf_kl_beta: Bradley-Terry preference probability, and the KL-regularized optimal policy")
    gap = np.linspace(-6, 6, 241)
    sig = 1 / (1 + np.exp(-gap))
    pi_ref = np.array([0.40, 0.30, 0.20, 0.10])
    reward = np.array([0.0, 1.0, 2.0, 3.0])
    labels = ["A (ref 0.40, r 0)", "B (ref 0.30, r 1)", "C (ref 0.20, r 2)", "D (ref 0.10, r 3)"]
    betas = [10.0, 1.0, 0.3]
    print("Bradley-Terry: P(y_w preferred) = sigmoid(r_w - r_l); gap 0 -> 0.5; gap 1 -> %.4f; gap 2 -> %.4f; gap -1 -> %.4f"
          % (1/(1+math.exp(-1)), 1/(1+math.exp(-2)), 1/(1+math.exp(1))))
    print("   reward-model loss -log sigmoid(gap): gap 1 -> %.4f, gap 2 -> %.4f, gap -1 -> %.4f"
          % (-math.log(1/(1+math.exp(-1))), -math.log(1/(1+math.exp(-2))), -math.log(1/(1+math.exp(1)))))
    print(f"\ntoy prompt with four candidate responses: pi_ref = {pi_ref}, reward-model scores r = {reward}")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    axes[0].plot(gap, sig, color=BLUE)
    axes[0].plot([1.0], [1/(1+math.exp(-1))], "o", color=BLUE, ms=6)
    axes[0].annotate("gap 1 -> 0.73", xy=(1.0, 1/(1+math.exp(-1))), xytext=(1.8, 0.45), fontsize=8, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=AXIS))
    axes[0].set_xlabel("reward gap r(y_w) - r(y_l)"); axes[0].set_ylabel("P(y_w is preferred)")
    axes[0].set_title("Bradley-Terry reward model")
    ax = axes[1]
    x = np.arange(4); width = 0.2
    ax.bar(x - 1.5 * width, pi_ref, width, color=INK_2, label="pi_ref", edgecolor=SURFACE)
    for i, beta in enumerate(betas):
        w = pi_ref * np.exp(reward / beta)
        pi_star = w / w.sum()
        kl = float(np.sum(pi_star * np.log(pi_star / pi_ref)))
        exp_r = float(pi_star @ reward)
        ax.bar(x + (i - 0.5) * width, pi_star, width, color=SERIES[i], label=f"beta = {beta}", edgecolor=SURFACE)
        print(f"   beta = {beta:4}: pi* = {pi_star}, expected reward {exp_r:.3f} (ref {pi_ref @ reward:.3f}), KL(pi* || pi_ref) = {kl:.4f} nats")
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_xlabel("candidate response (reward 0, 1, 2, 3)"); ax.set_ylabel("probability")
    ax.set_title("KL-regularized optimum pi* ~ pi_ref * exp(r / beta)"); ax.legend()
    save(fig, NB, "rlhf_kl_beta")
    print("   example: beta = 1, response D: 0.10 * exp(3) = %.4f before normalisation; normaliser = %.4f"
          % (0.10 * math.exp(3), float((pi_ref * np.exp(reward)).sum())))


def main():
    fig_discount_weights()
    fig_bellman_three_state()
    fig_cartpole_random_baseline()
    vi_out = fig_frozenlake_value_iteration()
    fig_frozenlake_q_vs_sarsa(vi_out)
    fig_epsilon_schedules()
    fig_reinforce_estimators()
    fig_baseline_variance_bandit()
    fig_huber_loss()
    fig_max_overestimation()
    fig_gae_weights()
    fig_ppo_clip_objective()
    fig_rlhf_kl_beta()


if __name__ == "__main__":
    main()
