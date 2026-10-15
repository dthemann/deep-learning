"""Figures for the companion guide of Notebook 10 · NLP with Pretrained Transformers (Hugging Face).

Everything here is numpy + matplotlib + pure Python. No PyTorch, no downloads, no Hugging Face
models: the Byte-Pair Encoding learner is the notebook's algorithm re-implemented in a few lines,
the skip-gram word2vec model with negative sampling is trained in numpy on a tiny synthetic corpus,
the fine-tuning memory table, the LoRA parameter counts, the DPO loss and the attention-memory
table are plain arithmetic, and the tabular DPO experiment of Section 6.3 is reproduced in numpy.

Run:  python3 notebooks/figures/scripts/make_figures_10.py
Every number quoted in the guide is printed by this script.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from figstyle import *   # SERIES, BLUE, ORANGE, ..., SEQ_CMAP, DIV_CMAP, apply_style, save, bar
apply_style()
rng = np.random.default_rng(0)

import math, re
from collections import Counter
from matplotlib.lines import Line2D

NB = "10"


def banner(title):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)


# ----------------------------------------------------------------------------------------------
# Byte-Pair Encoding, exactly as in the notebook (Section 1.2), in pure Python
# ----------------------------------------------------------------------------------------------
END = "</w>"


def bpe_init(text):
    words = re.findall(r"[a-z']+|[^\sa-z']", text.lower())
    return {tuple(w) + (END,): c for w, c in Counter(words).items()}


def bpe_pair_counts(word_freqs):
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def bpe_merge(pair, word_freqs):
    a, b = pair
    out = {}
    for symbols, freq in word_freqs.items():
        new, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(symbols[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out


def learn_bpe(text, num_merges):
    """Return (merges, vocab, history) with history = [(step, vocab size, tokens per word, count)]."""
    word_freqs = bpe_init(text)
    n_words = sum(word_freqs.values())
    vocab = set(s for w in word_freqs for s in w)
    merges, history = [], []
    for step in range(1, num_merges + 1):
        pairs = bpe_pair_counts(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_freqs = bpe_merge(best, word_freqs)
        merges.append(best)
        vocab.add(best[0] + best[1])
        tpw = sum(len(w) * f for w, f in word_freqs.items()) / n_words
        history.append((step, len(vocab), tpw, pairs[best]))
    return merges, vocab, history, word_freqs


def bpe_encode_word(word, ranks):
    symbols = list(word) + [END]
    while len(symbols) > 1:
        cands = [(ranks.get((a, b), math.inf), i) for i, (a, b) in enumerate(zip(symbols, symbols[1:]))]
        rank, i = min(cands)
        if rank == math.inf:
            break
        symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
    return symbols


def bpe_encode(text, ranks):
    toks = []
    for w in re.findall(r"[a-z']+|[^\sa-z']", text.lower()):
        toks.extend(bpe_encode_word(w, ranks))
    return toks


def bpe_decode(tokens):
    return "".join(tokens).replace(END, " ").strip()


# ----------------------------------------------------------------------------------------------
# Figure 1: BPE on the classic toy corpus (Sennrich et al. 2016 style)
# ----------------------------------------------------------------------------------------------
def fig_bpe_toy_merges():
    banner("Figure: bpe_toy_merges  (Section 1.2)")
    corpus = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
    wf = bpe_init(corpus)
    print("corpus word frequencies:", {"".join(k[:-1]): v for k, v in wf.items()})
    n_chars = len(set(s for w in wf for s in w))
    print(f"initial alphabet (distinct symbols incl. {END}): {n_chars}")
    pairs0 = bpe_pair_counts(wf)
    top0 = pairs0.most_common(6)
    print("pair counts before merge 1 (top 6):", [(a + "+" + b, c) for (a, b), c in top0])
    merges, vocab, hist, final_wf = learn_bpe(corpus, 12)
    print("\nstep | merge           | count | vocab | tokens/word")
    for (step, vs, tpw, cnt), m in zip(hist, merges):
        print(f"{step:4d} | {m[0]!r:7s}+{m[1]!r:8s} | {cnt:5d} | {vs:5d} | {tpw:.3f}")
    print(f"vocabulary after {len(merges)} merges: {len(vocab)} = {n_chars} + {len(merges)}:",
          len(vocab) == n_chars + len(merges))
    print("tokens/word: before any merge", f"{sum(len(w) * f for w, f in wf.items()) / sum(wf.values()):.3f}")
    for k in [0, 5, len(merges)]:
        ranks = {m: i for i, m in enumerate(merges[:k])}
        print(f"after {k:2d} merges: lowest -> {bpe_encode_word('lowest', ranks)}   "
              f"newest -> {bpe_encode_word('newest', ranks)}   widest -> {bpe_encode_word('widest', ranks)}")
    ranks = {m: i for i, m in enumerate(merges)}
    toks = bpe_encode("lower newest", ranks)
    print("encode('lower newest') ->", toks, "-> decode ->", repr(bpe_decode(toks)))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    labels = [a + b for (a, b), _ in top0]
    bar(axes[0], labels, [c for _, c in top0], color=BLUE, label_fmt="{:d}")
    axes[0].set_title("Pair counts before merge 1"); axes[0].set_ylabel("count (word-weighted)")
    axes[0].set_xlabel("adjacent symbol pair"); axes[0].tick_params(axis="x", labelsize=8)
    steps = [0] + [h[0] for h in hist]
    axes[1].plot(steps, [n_chars] + [h[1] for h in hist], marker="o", color=BLUE)
    axes[1].set_title("Vocabulary size"); axes[1].set_xlabel("number of merges"); axes[1].set_ylabel("symbols")
    tpw0 = sum(len(w) * f for w, f in wf.items()) / sum(wf.values())
    tpws = [tpw0] + [h[2] for h in hist]
    axes[2].plot(steps, tpws, marker="o", color=BLUE)
    axes[2].annotate(f"{tpws[0]:.2f}", (0, tpws[0]), xytext=(6, -2), textcoords="offset points", fontsize=8, color=INK_2)
    axes[2].annotate(f"{tpws[-1]:.2f}", (steps[-1], tpws[-1]), xytext=(-24, 8), textcoords="offset points", fontsize=8, color=INK_2)
    axes[2].set_title("Compression achieved"); axes[2].set_xlabel("number of merges"); axes[2].set_ylabel("tokens per word")
    save(fig, NB, "bpe_toy_merges")


# ----------------------------------------------------------------------------------------------
# Figure 2: characters vs words vs subwords on four sentences
# ----------------------------------------------------------------------------------------------
TOY_CORPUS = """
the model reads the text and the tokenizer splits the text into tokens .
a small model learns slowly and a large model learns quickly .
the tokenizer learned the merges from the training text .
we train the model on the training data and test the model on new data .
the reader reads the longest words slowly and the shortest words quickly .
learning the merges makes the longest words shorter .
the training data has many words and the tokens are shorter than words .
new words are split into pieces that the model has seen in training .
the model is trained and tested on text that the tokenizer splits .
tokens are the pieces of words that the merges produce .
"""


def fig_token_counts_by_scheme():
    banner("Figure: token_counts_by_scheme  (Section 1.1 / 1.5)")
    merges, vocab, hist, _ = learn_bpe(TOY_CORPUS, 60)
    ranks = {m: i for i, m in enumerate(merges)}
    print(f"toy BPE: {len(merges)} merges learned on {len(TOY_CORPUS.split())} words, vocabulary {len(vocab)}")
    sentences = ["the model reads the text",
                 "new words are split into pieces",
                 "the tokenizer learned the merges",
                 "retokenization is unpredictable"]
    chars, words, subs = [], [], []
    print(f"{'sentence':36s} {'chars':>5s} {'words':>5s} {'BPE':>4s}   BPE tokens")
    for s in sentences:
        toks = bpe_encode(s, ranks)
        chars.append(len(s)); words.append(len(s.split())); subs.append(len(toks))
        print(f"{s!r:36s} {len(s):5d} {len(s.split()):5d} {len(toks):4d}   {toks}")
    x = np.arange(len(sentences)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (vals, name, col) in enumerate(zip([chars, words, subs], ["characters", "words", "subwords (toy BPE, 60 merges)"], SERIES[:3])):
        bars = ax.bar(x + (k - 1) * w, vals, width=w, color=col, label=name, edgecolor=SURFACE, linewidth=1.0)
        for b, v in zip(bars, vals):
            ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8,
                        color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i + 1}" for i in range(len(sentences))])
    ax.set_ylabel("tokens per sentence"); ax.set_xlabel("sentence (see table in the guide)")
    ax.set_title("Sequence length under three tokenization schemes"); ax.legend()
    save(fig, NB, "token_counts_by_scheme")


# ----------------------------------------------------------------------------------------------
# A tiny skip-gram word2vec with negative sampling, in numpy (Section 2)
# ----------------------------------------------------------------------------------------------
TOPICS = {
    "royalty": (["king", "queen", "prince", "princess", "duke", "lord", "lady", "knight"], ["rules", "crowns", "greets", "honours"]),
    "animals": (["dog", "cat", "puppy", "kitten", "horse", "wolf", "rabbit", "fox"], ["chases", "bites", "sniffs", "follows"]),
    "food":    (["bread", "cheese", "apple", "soup", "butter", "honey", "pie", "stew"], ["flavours", "garnishes", "sweetens", "thickens"]),
}


def make_toy_corpus(n_per_topic=600, seed=0):
    r = np.random.default_rng(seed)
    sents = []
    for nouns, verbs in TOPICS.values():
        for _ in range(n_per_topic):
            a, b = r.choice(nouns, 2, replace=False)
            sents.append([a, r.choice(verbs), b])          # "the" is dropped: like the notebook's subsampling
    r.shuffle(sents)
    return sents


def logsigmoid(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    """torch.optim.Adam with default betas (0.9, 0.999) and eps 1e-8."""
    def __init__(self, shapes, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros(s) for s in shapes]; self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_skipgram(sents, dim=16, window=2, K=5, epochs=4, batch=128, lr=0.02, seed=0):
    r = np.random.default_rng(seed)
    vocab = sorted(set(w for s in sents for w in s))
    w2i = {w: i for i, w in enumerate(vocab)}
    ids = [[w2i[w] for w in s] for s in sents]
    counts = np.bincount(np.concatenate(ids), minlength=len(vocab))
    noise = counts ** 0.75; noise = noise / noise.sum()
    centers, contexts = [], []
    for s in ids:
        for i, c in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    centers.append(c); contexts.append(s[j])
    pairs = np.stack([centers, contexts], 1)
    V = len(vocab)
    v_in = r.uniform(-0.5 / dim, 0.5 / dim, (V, dim))      # center table (the final embedding)
    u_out = np.zeros((V, dim))                               # context table, zero as in word2vec.c
    opt = Adam([v_in.shape, u_out.shape], lr)
    log, step = [], 0
    for ep in range(epochs):
        perm = r.permutation(len(pairs)); ep_loss = []
        for s0 in range(0, len(pairs), batch):
            b = pairs[perm[s0:s0 + batch]]; B = len(b)
            neg = r.choice(V, size=(B, K), p=noise)
            v = v_in[b[:, 0]]; u = u_out[b[:, 1]]; un = u_out[neg]            # (B,d) (B,d) (B,K,d)
            s_pos = (u * v).sum(1); s_neg = np.einsum("bkd,bd->bk", un, v)
            loss = -(logsigmoid(s_pos) + logsigmoid(-s_neg).sum(1)).mean()
            g_pos = -(1 - sigmoid(s_pos)) / B                                 # dL/ds_pos
            g_neg = (1 - sigmoid(-s_neg)) / B                                 # dL/ds_neg
            gv = g_pos[:, None] * u + np.einsum("bk,bkd->bd", g_neg, un)
            gu = g_pos[:, None] * v; gun = g_neg[:, :, None] * v[:, None, :]
            G_in = np.zeros_like(v_in); G_out = np.zeros_like(u_out)
            np.add.at(G_in, b[:, 0], gv); np.add.at(G_out, b[:, 1], gu)
            np.add.at(G_out, neg.reshape(-1), gun.reshape(-1, dim))
            if step == 0:
                print(f"loss at step 0 (all scores 0): {loss:.4f}   (K+1) log 2 = {(K + 1) * math.log(2):.4f}")
            opt.step([v_in, u_out], [G_in, G_out])
            log.append(loss); ep_loss.append(loss); step += 1
        print(f"epoch {ep + 1}: mean loss {np.mean(ep_loss):.4f}  ({len(ep_loss)} steps)")
    return vocab, w2i, v_in, np.array(log), pairs


def normalized_embeddings(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


_W2V = {}


def get_w2v():
    if not _W2V:
        sents = make_toy_corpus()
        vocab, w2i, E, log, pairs = train_skipgram(sents)
        _W2V.update(vocab=vocab, w2i=w2i, E=E, log=log, pairs=pairs, sents=sents)
    return _W2V


def fig_skipgram_loss_curve():
    banner("Figure: skipgram_loss_curve  (Section 2.2 / 2.4)")
    # hand-sized worked example of the negative-sampling loss for one (center, context) pair, K = 2
    v_c = np.array([0.5, -0.2]); u_o = np.array([0.4, 0.1]); u_n = np.array([[-0.3, 0.2], [0.1, 0.3]])
    s_pos = u_o @ v_c; s_neg = u_n @ v_c
    print(f"worked example: v_c={v_c.tolist()} u_o={u_o.tolist()} negatives={u_n.tolist()}")
    print(f"  positive score u_o.v_c = {s_pos:.3f} -> sigma = {sigmoid(s_pos):.4f} -> -log sigma = {-logsigmoid(s_pos):.4f}")
    for k, s in enumerate(s_neg):
        print(f"  negative {k + 1} score = {s:.3f} -> sigma(-s) = {sigmoid(-s):.4f} -> -log sigma(-s) = {-logsigmoid(-s):.4f}")
    L = -logsigmoid(s_pos) - logsigmoid(-s_neg).sum()
    print(f"  loss = {L:.4f}   (an indifferent model with all scores 0 would pay 3 log 2 = {3 * math.log(2):.4f})")
    w = get_w2v()
    log = w["log"]; K = 5
    print(f"toy corpus: {len(w['sents'])} three-word sentences, vocabulary {len(w['vocab'])}, {len(w['pairs'])} (center, context) pairs")
    print(f"first logged loss {log[0]:.4f}, loss after {len(log)} steps {log[-1]:.4f} (mean of last 20 steps {log[-20:].mean():.4f})")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sm = np.convolve(log, np.ones(10) / 10, mode="valid")
    ax.plot(np.arange(len(sm)) + 10, sm, color=BLUE)
    ax.axhline((K + 1) * math.log(2), color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"(K+1) log 2 = {(K + 1) * math.log(2):.2f}", (len(log) * 0.55, (K + 1) * math.log(2)),
                xytext=(0, 5), textcoords="offset points", fontsize=8, color=INK_2)
    ax.annotate(f"{sm[-1]:.2f}", (len(log), sm[-1]), xytext=(-30, 10), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("optimizer step (batches of 128 pairs)"); ax.set_ylabel("negative-sampling loss (K = 5)")
    ax.set_title("Skip-gram with negative sampling on a toy corpus (10-step moving average)")
    save(fig, NB, "skipgram_loss_curve")


def fig_word2vec_toy_pca():
    banner("Figure: word2vec_toy_pca  (Section 2.4)")
    w = get_w2v(); vocab, w2i = w["vocab"], w["w2i"]
    En = normalized_embeddings(w["E"])
    for q in ["king", "dog", "bread"]:
        sims = En @ En[w2i[q]]; sims[w2i[q]] = -1
        top = np.argsort(-sims)[:3]
        print(f"nearest to {q:>6s}: " + ", ".join(f"{vocab[i]} ({sims[i]:.2f})" for i in top))
    # analogy in the toy space: king - queen + puppy -> ?  (same offset, different topic)
    t = En[w2i["king"]] - En[w2i["queen"]] + En[w2i["kitten"]]; t = t / np.linalg.norm(t)
    sims = En @ t
    for q in ("king", "queen", "kitten"):
        sims[w2i[q]] = -1
    print("toy analogy queen : king :: kitten : ? ->", vocab[int(np.argmax(sims))], f"({sims.max():.2f})  (toy corpus, illustrative only)")
    # PCA by SVD of the centred, normalized matrix
    Xc = En - En.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = S ** 2 / (S ** 2).sum()
    print(f"PCA: first two components explain {var[0]:.1%} + {var[1]:.1%} = {var[:2].sum():.1%} of the variance")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for (name, (nouns, verbs)), col in zip(TOPICS.items(), SERIES[:3]):
        idx = [w2i[x] for x in nouns + verbs]
        ax.scatter(proj[idx, 0], proj[idx, 1], s=36, color=col, label=name, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for i in idx:
            ax.annotate(vocab[i], proj[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xlabel("principal component 1"); ax.set_ylabel("principal component 2")
    ax.set_title("Toy word2vec embeddings, PCA projection (36 words)"); ax.legend(title=None, loc="best")
    save(fig, NB, "word2vec_toy_pca")


# ----------------------------------------------------------------------------------------------
# Figure: decoding knobs (temperature, top-k, top-p) on a toy next-token distribution (Section 3.3)
# ----------------------------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def fig_decoding_knobs():
    banner("Figure: decoding_knobs  (Section 3.3)")
    tokens = ["the", "a", "model", "cat", "runs", "quickly", "banana", "<eos>"]
    logits = np.array([2.0, 1.6, 1.2, 0.4, 0.0, -0.4, -1.5, -2.0])
    print("toy logits:", dict(zip(tokens, logits.tolist())))
    temps = [0.5, 1.0, 2.0]
    dists = {T: softmax(logits / T) for T in temps}
    for T in temps:
        print(f"T={T}: " + " ".join(f"{t}={p:.3f}" for t, p in zip(tokens, dists[T])))
    p1 = dists[1.0]
    order = np.argsort(-p1)
    topk = np.zeros_like(p1); topk[order[:3]] = p1[order[:3]]; topk /= topk.sum()
    cum = np.cumsum(p1[order]); keep = order[: int(np.searchsorted(cum, 0.8) + 1)]
    topp = np.zeros_like(p1); topp[keep] = p1[keep]; topp /= topp.sum()
    print(f"top-k (k=3) keeps {[tokens[i] for i in order[:3]]}, renormalized: " + " ".join(f"{tokens[i]}={topk[i]:.3f}" for i in order[:3]))
    print(f"top-p (p=0.8) keeps {[tokens[i] for i in keep]} (cumulative {cum[len(keep) - 1]:.3f}), renormalized: "
          + " ".join(f"{tokens[i]}={topp[i]:.3f}" for i in keep))
    x = np.arange(len(tokens)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, T in enumerate(temps):
        axes[0].bar(x + (k - 1) * w, dists[T], width=w, color=SERIES[k], label=f"temperature {T}", edgecolor=SURFACE, linewidth=0.8)
    axes[0].set_title("Temperature reshapes the softmax"); axes[0].legend()
    for k, (d, name) in enumerate(zip([p1, topk, topp], ["full distribution (T = 1)", "top-k, k = 3", "top-p, p = 0.8"])):
        axes[1].bar(x + (k - 1) * w, d, width=w, color=SERIES[k], label=name, edgecolor=SURFACE, linewidth=0.8)
    axes[1].set_title("Top-k and top-p cut the tail, then renormalize"); axes[1].legend()
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(tokens, fontsize=8); ax.set_ylabel("probability"); ax.set_xlabel("candidate next token")
    save(fig, NB, "decoding_knobs")


# ----------------------------------------------------------------------------------------------
# Figure: sentence similarity heat-map with toy mean-pooled vectors (Section 3.4)
# ----------------------------------------------------------------------------------------------
def fig_sentence_similarity_toy():
    banner("Figure: sentence_similarity_toy  (Section 3.4)")
    w = get_w2v(); w2i = w["w2i"]
    En = normalized_embeddings(w["E"])
    sentences = ["the king greets the queen",
                 "the knight honours the duke",
                 "the dog chases the cat",
                 "the fox bites the rabbit",
                 "the cheese flavours the soup"]

    def embed(s):
        ws = [x for x in s.split() if x in w2i]            # "the" is not in the toy vocabulary: it is ignored
        v = En[[w2i[x] for x in ws]].mean(0)               # mean pooling over the content words
        return v / np.linalg.norm(v)

    M = np.stack([embed(s) for s in sentences]); sim = M @ M.T
    print("cosine similarity matrix (rows/cols = S1..S5):")
    for i, row in enumerate(sim):
        print(f"  S{i + 1} " + " ".join(f"{x:6.2f}" for x in row) + f"   {sentences[i]}")
    query = "the prince rules the castle"
    q = embed(query); scores = M @ q
    print(f"query {query!r} (castle is out of vocabulary and ignored) ranked:")
    for i in np.argsort(-scores):
        print(f"  {scores[i]:6.2f}  S{i + 1}: {sentences[i]}")
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(sim, cmap=DIV_CMAP, vmin=-1, vmax=1)
    labels = [f"S{i + 1}" for i in range(len(sentences))]
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if abs(sim[i, j]) > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Cosine similarity of mean-pooled toy sentence vectors")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("cosine similarity")
    save(fig, NB, "sentence_similarity_toy")


# ----------------------------------------------------------------------------------------------
# Figure: the fine-tuning recipe: dynamic padding and the warmup/decay schedule (Section 4)
# ----------------------------------------------------------------------------------------------
def lr_factor(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def fig_finetune_recipe():
    banner("Figure: finetune_recipe  (Sections 4.2 and 4.3)")
    MAX_LEN = 32
    batches = [[9, 12, 7, 14], [21, 8, 10, 11], [6, 9, 5, 8], [30, 12, 9, 15]]      # token counts per sentence
    fixed = [len(b) * MAX_LEN for b in batches]; dyn = [len(b) * max(b) for b in batches]; real = [sum(b) for b in batches]
    print("batch | lengths            | fixed padding | dynamic padding | real tokens")
    for i, b in enumerate(batches):
        print(f"  {i + 1}   | {str(b):18s} | {fixed[i]:13d} | {dyn[i]:15d} | {real[i]:11d}")
    print(f"totals: fixed {sum(fixed)} | dynamic {sum(dyn)} | real {sum(real)}  -> dynamic padding processes "
          f"{sum(dyn) / sum(fixed):.0%} of the fixed-padding tokens; waste falls from {1 - sum(real) / sum(fixed):.0%} to {1 - sum(real) / sum(dyn):.0%}")
    warmup, total = 4, 20
    factors = [lr_factor(s, warmup, total) for s in range(total)]
    print("LR multipliers (4 warmup steps, 20 total):", [round(f, 3) for f in factors])
    n_train, batch, epochs, lr = 4000, 32, 3, 5e-5
    steps = epochs * math.ceil(n_train / batch); wu = int(0.1 * steps)
    print(f"notebook full budget: {epochs} epochs x ceil({n_train}/{batch}) = {steps} steps, warmup = int(0.1 x {steps}) = {wu} steps;"
          f" LR at step 10 = {lr * lr_factor(10, wu, steps):.2e}, at step {wu} = {lr * lr_factor(wu, wu, steps):.2e}, at step 200 = {lr * lr_factor(200, wu, steps):.2e}")
    x = np.arange(len(batches)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, (vals, name) in enumerate(zip([fixed, dyn, real], ["fixed padding (to 32)", "dynamic padding", "real tokens"])):
        axes[0].bar(x + (k - 1) * w, vals, width=w, color=SERIES[k], label=name, edgecolor=SURFACE, linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"batch {i + 1}" for i in x])
    axes[0].set_ylabel("tokens processed"); axes[0].set_title("Dynamic padding: tokens per batch of 4 sentences"); axes[0].legend()
    axes[1].plot(range(total), factors, marker="o", color=BLUE)
    axes[1].annotate("peak after 4 warmup steps", (4, 1.0), xytext=(8, -4), textcoords="offset points", fontsize=8, color=INK_2)
    axes[1].set_xlabel("optimizer step"); axes[1].set_ylabel("learning-rate multiplier")
    axes[1].set_title("Linear warmup (4 steps), linear decay to 0 (20 steps)")
    save(fig, NB, "finetune_recipe")


# ----------------------------------------------------------------------------------------------
# Figure: fine-tuning memory, full vs LoRA vs QLoRA (Section 5.1)
# ----------------------------------------------------------------------------------------------
def finetune_memory_gb(n_params, n_trainable, weight_bytes=4, state_bytes=4):
    weights = n_params * weight_bytes; grads = n_trainable * state_bytes; adam = 2 * n_trainable * state_bytes
    return weights / 1e9, grads / 1e9, adam / 1e9


def fig_finetune_memory():
    banner("Figure: finetune_memory  (Section 5.1)")
    configs = [("DistilBERT 66M, full FT (fp32)", 66e6, 66e6, 4),
               ("DistilBERT 66M, LoRA r=8 (fp32)", 66e6, 0.74e6, 4),
               ("7B, full FT (fp32)", 7e9, 7e9, 4),
               ("7B, full FT (bf16 weights)", 7e9, 7e9, 2),
               ("7B, LoRA r=16 (bf16 weights)", 7e9, 40e6, 2),
               ("7B, QLoRA r=16 (4-bit weights)", 7e9, 40e6, 0.5)]
    rows = []
    print(f"{'configuration':34s} {'weights':>8s} {'grads':>7s} {'Adam':>7s} {'total':>8s}  (GB)")
    for name, n, ntr, wb in configs:
        w, g, a = finetune_memory_gb(n, ntr, weight_bytes=wb); rows.append((w, g, a))
        print(f"{name:34s} {w:8.2f} {g:7.2f} {a:7.2f} {w + g + a:8.2f}")
    print("bytes per trainable parameter in fp32 with Adam: 4 (weight) + 4 (grad) + 8 (two moments) = 16")
    print(f"7B full FT total / 7B QLoRA total = {sum(rows[2]) / sum(rows[5]):.0f}x")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(configs))[::-1]
    left = np.zeros(len(configs))
    for k, name in enumerate(["weights", "gradients", "Adam moments"]):
        vals = np.array([r[k] for r in rows])
        ax.barh(y, vals, left=left, color=SERIES[k], label=name, edgecolor=SURFACE, linewidth=0.8, height=0.6)
        left += vals
    for yi, tot in zip(y, left):
        ax.annotate(f"{tot:.1f} GB", (tot, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in configs], fontsize=8.5)
    ax.set_xlabel("memory for weights + gradients + optimizer state (GB), activations excluded")
    ax.set_xlim(0, 125); ax.set_title("What fine-tuning costs in memory"); ax.legend(loc="lower right")
    save(fig, NB, "finetune_memory")


# ----------------------------------------------------------------------------------------------
# Figure: LoRA parameter count vs rank for DistilBERT (Section 5.2 / Exercise 4)
# ----------------------------------------------------------------------------------------------
def fig_lora_params_vs_rank():
    banner("Figure: lora_params_vs_rank  (Section 5.2)")
    d, dff, layers = 768, 3072, 6
    full = 66_362_880                                   # DistilBertModel parameters (model card)
    ranks = [1, 2, 4, 8, 16, 32, 64]
    qv = [layers * 2 * r * (d + d) for r in ranks]
    alll = [layers * (4 * r * (d + d) + 2 * r * (d + dff)) for r in ranks]
    print(f"one 768x768 projection, r=8: LoRA trains 8 x (768 + 768) = {8 * (d + d):,} instead of {d * d:,} -> {d * d / (8 * (d + d)):.0f}x fewer")
    print("rank | q,v only (12 layers) | all 6 linears per block (36 layers) | share of 66.4M (q,v)")
    for r, a, b in zip(ranks, qv, alll):
        print(f"{r:4d} | {a:20,d} | {b:34,d} | {a / full:.3%}")
    print(f"ratio all/q,v = {alll[0] / qv[0]:.1f} at every rank")
    # tiny LoRA update by hand: W (2x3), r = 1, alpha = 2
    W = np.array([[1.0, 0.0, 2.0], [0.5, -1.0, 0.0]]); A = np.array([[1.0, -1.0, 0.5]]); B0 = np.zeros((2, 1)); B1 = np.array([[0.2], [-0.4]])
    alpha, r = 2.0, 1
    print("hand example: W =", W.tolist(), " A =", A.tolist(), " alpha/r =", alpha / r)
    print("  at init B = 0 -> delta W =", ((alpha / r) * B0 @ A).tolist(), "-> adapted layer == base layer")
    print("  after training B =", B1.ravel().tolist(), "-> delta W =", np.round((alpha / r) * B1 @ A, 3).tolist())
    print("  merged W + delta W =", np.round(W + (alpha / r) * B1 @ A, 3).tolist())
    x = np.array([1.0, 2.0, -1.0])
    print("  check on x =", x.tolist(), ": W x + (alpha/r) B (A x) =", np.round(W @ x + (alpha / r) * (B1 @ (A @ x)), 3).tolist(),
          " merged W' x =", np.round((W + (alpha / r) * B1 @ A) @ x, 3).tolist())
    # DoRA decomposition of the same W
    m = np.linalg.norm(W, axis=1); print("  DoRA: row norms m =", np.round(m, 3).tolist(), " direction rows =", np.round(W / m[:, None], 3).tolist())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ranks, qv, marker="o", color=SERIES[0], label="LoRA on q_lin, v_lin (12 layers)")
    ax.plot(ranks, alll, marker="s", color=SERIES[1], label="LoRA on all 6 linears per block (36 layers)")
    ax.axhline(full, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("full fine-tuning: 66.4M", (ranks[0], full), xytext=(0, 5), textcoords="offset points", fontsize=8, color=INK_2)
    ax.annotate(f"r = 8: {qv[3] / 1e3:.0f}k", (8, qv[3]), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("trainable adapter parameters (log scale)")
    ax.set_title("LoRA parameters grow linearly with the rank"); ax.legend(loc="center right")
    save(fig, NB, "lora_params_vs_rank")


# ----------------------------------------------------------------------------------------------
# Figure: the DPO loss as a function of the implicit-reward margin (Section 6.3)
# ----------------------------------------------------------------------------------------------
def fig_dpo_loss_vs_margin():
    banner("Figure: dpo_loss_vs_margin  (Section 6.3)")
    # masked SFT loss (Section 6.2) by hand with a toy vocabulary of 6 tokens
    r = np.random.default_rng(3)
    V = 6; prompt = [1, 4, 2]; response = [5, 0, 3]
    seq = prompt + response; labels = [-100] * len(prompt) + response
    logits = r.normal(size=(len(seq), V))
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    per_tok = [-logp[t, seq[t + 1]] for t in range(len(seq) - 1)]       # position t predicts token t+1
    shifted_labels = labels[1:]
    kept = [l for l, lab in zip(per_tok, shifted_labels) if lab != -100]
    print("SFT masking: input_ids =", seq, " labels =", labels, " shifted labels =", shifted_labels)
    print("  per-position cross-entropy =", [round(x, 3) for x in per_tok])
    print(f"  mean over all 5 predictions = {np.mean(per_tok):.4f}; mean over the {len(kept)} response positions only = {np.mean(kept):.4f} (this is the SFT loss)")
    # one preference pair by hand
    lp = dict(pi_w=-2.0, ref_w=-2.5, pi_l=-3.0, ref_l=-2.2)
    print("DPO worked example: log pi(y_w)=-2.0, log pi_ref(y_w)=-2.5, log pi(y_l)=-3.0, log pi_ref(y_l)=-2.2")
    for beta in [0.1, 0.5, 2.0]:
        rw = beta * (lp["pi_w"] - lp["ref_w"]); rl = beta * (lp["pi_l"] - lp["ref_l"]); m = rw - rl
        print(f"  beta={beta}: implicit rewards {rw:+.2f} / {rl:+.2f}, margin {m:+.2f}, loss -log sigma(margin) = {-logsigmoid(m):.4f}")
    print(f"  at margin 0 every beta pays log 2 = {math.log(2):.4f}")
    margins = np.linspace(-6, 6, 241)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, beta in enumerate([0.1, 0.5, 2.0]):
        ax.plot(margins, -logsigmoid(beta * margins), color=SERIES[k], label=f"beta = {beta}")
    ax.axhline(math.log(2), color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("log 2 at zero margin", (-5.8, math.log(2)), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("log-ratio margin  (log pi/pi_ref of chosen) - (log pi/pi_ref of rejected)")
    ax.set_ylabel("DPO loss  -log sigma(beta x margin)"); ax.set_title("The DPO loss is a logistic loss on the implicit-reward margin")
    ax.legend()
    save(fig, NB, "dpo_loss_vs_margin")


# ----------------------------------------------------------------------------------------------
# Figure: the notebook's tabular DPO experiment, reproduced in numpy (Section 6.3)
# ----------------------------------------------------------------------------------------------
def log_softmax_rows(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def fig_dpo_toy_training():
    banner("Figure: dpo_toy_training  (Section 6.3)")
    r = np.random.default_rng(42)
    nP, nR = 4, 6
    ref_logits = r.normal(size=(nP, nR)); true_r = 2.0 * r.normal(size=(nP, nR))
    pairs = []
    for p in range(nP):
        for _ in range(12):
            i, j = r.choice(nR, 2, replace=False)
            p_i = sigmoid(true_r[p, i] - true_r[p, j])
            pairs.append((p, i, j) if r.random() < p_i else (p, j, i))
    P = np.array([a for a, _, _ in pairs]); C = np.array([b for _, b, _ in pairs]); R = np.array([c for _, _, c in pairs])
    ref_lp = log_softmax_rows(ref_logits)
    print(f"{len(pairs)} preference pairs from a Bradley-Terry annotator; the reference policy agrees with the annotator on "
          f"{np.mean(ref_lp[P, C] > ref_lp[P, R]):.0%} of them; {nP * nR * (nR - 1) // 2} underlying response pairs")

    def train(beta, steps=150, lr=0.1):
        th = ref_logits.copy(); opt = Adam([th.shape], lr); log = {"loss": [], "acc": [], "kl": []}
        for _ in range(steps):
            lp = log_softmax_rows(th)
            margin = beta * ((lp[P, C] - ref_lp[P, C]) - (lp[P, R] - ref_lp[P, R]))
            loss = -logsigmoid(margin).mean()
            dm = -(1 - sigmoid(margin)) / len(margin)
            dlp = np.zeros_like(th); np.add.at(dlp, (P, C), beta * dm); np.add.at(dlp, (P, R), -beta * dm)
            dz = dlp - np.exp(lp) * dlp.sum(-1, keepdims=True)
            log["loss"].append(loss); log["acc"].append((margin > 0).mean())
            log["kl"].append((np.exp(lp) * (lp - ref_lp)).sum(-1).mean())
            opt.step([th], [dz])
        return th, log

    runs = {b: train(b) for b in [0.1, 0.5, 2.0]}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, (beta, (th, log)) in enumerate(runs.items()):
        axes[0].plot(log["loss"], color=SERIES[k], label=f"beta = {beta}")
        axes[1].plot(log["acc"], color=SERIES[k], label=f"beta = {beta}")
        axes[2].plot(log["kl"], color=SERIES[k], label=f"beta = {beta}")
        imp = beta * (log_softmax_rows(th) - ref_lp); agree = n = 0
        for p in range(nP):
            for i in range(nR):
                for j in range(i + 1, nR):
                    agree += int((imp[p, i] > imp[p, j]) == (true_r[p, i] > true_r[p, j])); n += 1
        print(f"beta={beta:<4}: step-1 loss {log['loss'][0]:.3f} -> final loss {log['loss'][-1]:.3f}, implicit-reward accuracy "
              f"{log['acc'][-1]:.0%}, KL(policy||ref) {log['kl'][-1]:.2f}, agreement with the hidden reward {agree / n:.0%} of {n} pairs")
    axes[0].set_title("DPO loss"); axes[1].set_title("implicit-reward accuracy"); axes[2].set_title("KL(policy || reference)")
    axes[0].set_ylabel("loss"); axes[1].set_ylabel("fraction of training pairs"); axes[2].set_ylabel("nats")
    for ax in axes:
        ax.set_xlabel("step"); ax.legend(fontsize=8)
    save(fig, NB, "dpo_toy_training")


# ----------------------------------------------------------------------------------------------
# Figure: attention memory vs sequence length, and bytes per parameter (Section 7)
# ----------------------------------------------------------------------------------------------
def to_bfloat16(x):
    """Round float32 values to bfloat16 (round-to-nearest-even on the low 16 bits) and return as float32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32)


def fig_attention_memory():
    banner("Figure: attention_memory  (Sections 7.1 and 7.2)")
    B, H, LAYERS, BYTES = 8, 12, 6, 2
    Ls = [128, 512, 2048, 8192, 32768]
    print(f"score matrices (batch {B}, {H} heads, fp16): bytes per layer = {B} x {H} x L^2 x {BYTES}")
    print(f"{'L':>7s} {'per layer (GB)':>15s} {'6 layers (GB)':>14s}")
    per_layer = []
    for L in Ls:
        pl = B * H * L * L * BYTES; per_layer.append(pl)
        print(f"{L:7d} {pl / 1e9:15.3f} {pl * LAYERS / 1e9:14.2f}")
    print(f"doubling L from 512 to 1024 multiplies the memory by {(1024 / 512) ** 2:.0f}")
    precisions = [("fp32", 4), ("bf16 / fp16", 2), ("int8", 1), ("nf4 (4-bit)", 0.5)]
    print("7B weights by precision:", ", ".join(f"{n}: {7e9 * b / 1e9:.1f} GB" for n, b in precisions))
    vals = np.array([1.2345678, 0.001234, 345.678], dtype=np.float32)
    bf = to_bfloat16(vals)
    print("bfloat16 rounding of", vals.tolist(), "->", bf.tolist(), "relative errors", [f"{abs(a - b) / abs(a):.2e}" for a, b in zip(vals, bf)])
    print(f"bf16 has 7 explicit mantissa bits: relative spacing 2^-7 = {2 ** -7:.2e}, worst-case rounding error 2^-8 = {2 ** -8:.2e}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].plot(Ls, np.array(per_layer) * LAYERS / 1e9, marker="o", color=BLUE)
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xticks(Ls); axes[0].set_xticklabels([str(L) for L in Ls])
    axes[0].annotate(f"{per_layer[-1] * LAYERS / 1e9:,.0f} GB", (Ls[-1], per_layer[-1] * LAYERS / 1e9), xytext=(-52, -2), textcoords="offset points", fontsize=8, color=INK_2)
    axes[0].annotate(f"{per_layer[0] * LAYERS / 1e9:.2f} GB", (Ls[0], per_layer[0] * LAYERS / 1e9), xytext=(6, -2), textcoords="offset points", fontsize=8, color=INK_2)
    axes[0].set_xlabel("sequence length L (tokens)"); axes[0].set_ylabel("attention score matrices, 6 layers (GB)")
    axes[0].set_title("Attention memory is quadratic in L (batch 8, 12 heads, fp16)")
    bar(axes[1], [p[0] for p in precisions], [7e9 * p[1] / 1e9 for p in precisions], color=BLUE, label_fmt="{:.1f} GB")
    axes[1].set_ylabel("weights of a 7B model (GB)"); axes[1].set_xlabel("storage precision")
    axes[1].set_title("Lower precision shrinks the weights")
    save(fig, NB, "attention_memory")


def main():
    fig_bpe_toy_merges()
    fig_token_counts_by_scheme()
    fig_skipgram_loss_curve()
    fig_word2vec_toy_pca()
    fig_decoding_knobs()
    fig_sentence_similarity_toy()
    fig_finetune_recipe()
    fig_finetune_memory()
    fig_lora_params_vs_rank()
    fig_dpo_loss_vs_margin()
    fig_dpo_toy_training()
    fig_attention_memory()


if __name__ == "__main__":
    main()
