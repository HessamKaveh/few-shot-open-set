"""Open-set few-shot evaluation on the 20 novel classes.

Each test episode: 5 "taught" classes with K support images each, 15 queries per taught class, plus 75 queries
from 5 other novel classes that were never taught (the robot meets objects it does not know).

    python src/evaluate.py     # -> results/results.json, results/figures/*.png

Reported intervals are 95 % confidence intervals over 2,000 episodes (mean +- 1.96 sd / sqrt(n)).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import data
from model import Conv4, ResNet12

ROOT = data.ROOT
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "figures")
N_WAY, Q, N_UNK, ALPHA = 5, 15, 5, 0.05
EPISODES = int(os.environ.get("EPISODES", 2000))
SHOTS = [1, 5, 10, 20]
MODELS = {"protonet": "Conv4 ProtoNet", "pretrain": "ResNet-12 pre-trained"}
SCORES = {"dist": "distance to nearest prototype", "msp": "max softmax probability",
          "rel": "distance / support spread"}


@torch.no_grad()
def embed(method, X, Y, s, dev):
    model = (Conv4() if method == "protonet" else ResNet12(num_classes=64)).to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "cache", f"{method}.pt"), map_location=dev))
    model.eval()

    def run(idx):
        out = []
        for x, _ in DataLoader(data.Images(X[idx], Y[idx]), batch_size=500, num_workers=8):
            out.append(model(x.to(dev)).float())
        return torch.cat(out)
    E = {}
    keep = np.isin(Y, s["val"] + s["novel"])
    feats = run(np.where(keep)[0])
    if method == "pretrain":      # SimpleShot "CL2N": centre on the base-class mean, then L2-normalise
        base_idx = np.random.default_rng(0).choice(np.where(np.isin(Y, s["base"]))[0], 6400, replace=False)
        mu = run(base_idx).mean(0)
        feats = torch.nn.functional.normalize(feats - mu, dim=1)
    for c in s["val"] + s["novel"]:
        E[c] = feats[torch.from_numpy(Y[keep] == c).to(dev)]
    return E


def episode(E, classes, k, rng, coarse_of):
    cls = rng.choice(classes, N_WAY + N_UNK, replace=False)
    known, unk = cls[:N_WAY], cls[N_WAY:]
    sup, qry, unq = [], [], []
    for c in known:
        i = rng.choice(len(E[c]), k + Q, replace=False)
        sup.append(E[c][i[:k]]); qry.append(E[c][i[k:]])
    for c in unk:
        unq.append(E[c][rng.choice(len(E[c]), Q, replace=False)])
    sup = torch.stack(sup)                                   # (N, K, D)
    qry, unq = torch.cat(qry), torch.cat(unq)
    known_sc = {coarse_of[int(c)] for c in known}
    near = np.repeat([coarse_of[int(c)] in known_sc for c in unk], Q)
    return sup, qry, unq, near


def scores(sup, x, T):
    """Higher = more likely a taught class. Returns dict of (n,) arrays and the predicted class."""
    proto = sup.mean(1)
    d = torch.cdist(x, proto) ** 2                            # (n, N)
    dmin, pred = d.min(1)
    out = {"dist": -dmin, "msp": torch.softmax(-d / T, 1).max(1)[0]}
    if sup.shape[1] > 1:
        spread = ((sup - proto[:, None]) ** 2).sum(-1).mean()  # mean squared support-to-prototype distance
        out["rel"] = -dmin / spread
    return {k: v.cpu().numpy() for k, v in out.items()}, pred.cpu().numpy()


def loo_scores(sup, T):
    """Leave-one-out scores of the support images themselves: each image scored against prototypes where its own
    class prototype is computed without it. Available inside every episode, no extra data needed (K >= 2)."""
    N, K, _ = sup.shape
    out = {k: [] for k in SCORES}
    for c in range(N):
        for i in range(K):
            s2 = sup.clone()
            keep = [j for j in range(K) if j != i]
            proto = s2.mean(1)
            proto[c] = sup[c, keep].mean(0)
            d = torch.cdist(sup[c, i][None], proto)[0] ** 2
            out["dist"].append(-d[c].item())
            out["msp"].append(torch.softmax(-d / T, 0)[c].item())
            spread = ((sup - sup.mean(1)[:, None]) ** 2).sum(-1).mean()
            out["rel"].append((-d[c] / spread).item())
    return {k: np.array(v) for k, v in out.items()}


def lower_quantile(v, alpha):
    """Conformal lower threshold: accept x if score(x) >= t; P(accept a new taught-class query) >= 1 - alpha."""
    n = len(v)
    k = int(np.floor((n + 1) * alpha))
    return -np.inf if k < 1 else np.sort(v)[k - 1]


def fit_temperature(E, s, k, coarse_of):
    rng = np.random.default_rng(123)
    best, bt = np.inf, 1.0
    eps = [episode(E, s["val"], k, rng, coarse_of) for _ in range(200)]
    for T in np.geomspace(0.01, 100, 41):
        nll = 0
        for sup, qry, _, _ in eps:
            d = torch.cdist(qry, sup.mean(1)) ** 2
            y = torch.arange(N_WAY, device=d.device).repeat_interleave(Q)
            nll += torch.nn.functional.cross_entropy(-d / T, y).item()
        if nll < best:
            best, bt = nll, float(T)
    return bt


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(v.mean()), float(1.96 * v.std() / np.sqrt(len(v)))]


def main():
    os.makedirs(FIG, exist_ok=True)
    X, Y, coarse_of, s, names = data.split()
    dev = torch.device("cuda")
    R = {"episodes": EPISODES, "alpha": ALPHA, "models": {}}
    store = {}
    for m in MODELS:
        E = embed(m, X, Y, s, dev)
        R["models"][m] = {}
        for k in SHOTS:
            T = fit_temperature(E, s, k, coarse_of)
            # global threshold: known-query scores pooled over 1,000 validation-class episodes
            rng = np.random.default_rng(1000 + k)
            pool = {sc: [] for sc in SCORES}
            for _ in range(1000):
                sup, qry, _, _ = episode(E, s["val"], k, rng, coarse_of)
                sq, _ = scores(sup, qry, T)
                for sc in sq:
                    pool[sc].append(sq[sc])
            thr_global = {sc: lower_quantile(np.concatenate(v), ALPHA) for sc, v in pool.items() if v}
            rng = np.random.default_rng(k)
            acc, auroc, au_near, au_far = [], {sc: [] for sc in SCORES}, {sc: [] for sc in SCORES}, \
                {sc: [] for sc in SCORES}
            acc_known, acc_unk = {"global": {sc: [] for sc in SCORES}, "loo": {sc: [] for sc in SCORES}}, \
                {"global": {sc: [] for sc in SCORES}, "loo": {sc: [] for sc in SCORES}}
            for e in range(EPISODES):
                sup, qry, unq, near = episode(E, s["novel"], k, rng, coarse_of)
                sq, pred = scores(sup, qry, T)
                su, _ = scores(sup, unq, T)
                y = np.repeat(np.arange(N_WAY), Q)
                acc.append((pred == y).mean())
                lo = loo_scores(sup, T) if k > 1 else None
                for sc in sq:
                    lab = np.r_[np.ones(len(y)), np.zeros(len(su[sc]))]
                    auroc[sc].append(roc_auc_score(lab, np.r_[sq[sc], su[sc]]))
                    for sub, store_ in [(near, au_near), (~near, au_far)]:
                        store_[sc].append(roc_auc_score(np.r_[np.ones(len(y)), np.zeros(sub.sum())],
                                                        np.r_[sq[sc], su[sc][sub]]) if sub.any() else np.nan)
                    t = thr_global[sc]
                    acc_known["global"][sc].append((sq[sc] >= t).mean())
                    acc_unk["global"][sc].append((su[sc] >= t).mean())
                    if lo is not None:
                        t = lower_quantile(lo[sc], ALPHA)
                        acc_known["loo"][sc].append((sq[sc] >= t).mean())
                        acc_unk["loo"][sc].append((su[sc] >= t).mean())
            r = {"temperature": T, "accuracy": ci(acc), "auroc": {sc: ci(v) for sc, v in auroc.items() if v},
                 "auroc_near": {sc: ci(v) for sc, v in au_near.items() if v},
                 "auroc_far": {sc: ci(v) for sc, v in au_far.items() if v}, "threshold": {}}
            for how in ["global", "loo"]:
                for sc in SCORES:
                    a, u = np.array(acc_known[how][sc]), np.array(acc_unk[how][sc])
                    if len(a) == 0:
                        continue
                    r["threshold"][f"{how}/{sc}"] = {
                        "known_accept": ci(a), "unknown_accept": ci(u),
                        "episodes_below_90": float((a < 0.90).mean()),
                        "episodes_below_95": float((a < 0.95).mean())}
                    store[(m, k, how, sc)] = a
            R["models"][m][str(k)] = r
            print(m, k, "acc %.3f" % r["accuracy"][0],
                  {sc: round(v[0], 3) for sc, v in r["auroc"].items()},
                  {h: (round(v["known_accept"][0], 3), round(v["unknown_accept"][0], 3), v["episodes_below_90"])
                   for h, v in r["threshold"].items()}, flush=True)
    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(ROOT, "cache", "acceptance.npz"),
                        **{"|".join(map(str, k)): v for k, v in store.items()})
    figures(R, store, X, Y, s, coarse_of, names, dev)


def figures(R, store, X, Y, s, coarse_of, names, dev):
    M = R["models"]
    col = {"protonet": "#8da0cb", "pretrain": "#1f4e9c"}
    # 1. accuracy and AUROC vs shots
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m in M:
        ks = [int(k) for k in M[m]]
        axs[0].errorbar(ks, [M[m][str(k)]["accuracy"][0] for k in ks], [M[m][str(k)]["accuracy"][1] for k in ks],
                        fmt="o-", color=col[m], label=MODELS[m], capsize=3)
        for sc, ls in [("dist", "-"), ("msp", ":")]:
            axs[1].plot(ks, [M[m][str(k)]["auroc"][sc][0] for k in ks], "o" + ls, color=col[m],
                        label=f"{MODELS[m]}, {SCORES[sc]}")
        axs[1].plot(ks, [M[m][str(k)]["auroc_near"]["dist"][0] for k in ks], "x--", color=col[m], alpha=0.6,
                    label=f"{MODELS[m]}, near unknowns only")
    axs[0].set_title("Closed-set 5-way accuracy (taught classes)"); axs[1].set_title("Taught vs unknown: AUROC")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks(SHOTS); ax.set_xticklabels(SHOTS); ax.set_xlabel("shots per class")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8); axs[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shots.png"), dpi=130); plt.close()

    # 2. per-episode acceptance of taught objects: global vs leave-one-out threshold
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, k in zip(axs, [5, 10, 20]):
        data_, labels = [], []
        for m in M:
            for how in ["global", "loo"]:
                a = store.get((m, k, how, "dist"))
                if a is not None:
                    data_.append(a); labels.append(f"{'ProtoNet' if m == 'protonet' else 'ResNet-12'}\n{how}")
        ax.violinplot(data_, showmedians=True)
        ax.axhline(1 - ALPHA, color="r", ls="--", lw=1, label="target 95 %")
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{k}-shot: share of taught-object queries accepted, per episode", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("accepted fraction in one episode"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "acceptance.png"), dpi=130); plt.close()

    # 3. trade-off: unknown acceptance at the conformal threshold vs shots
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in M:
        for how, ls in [("global", "-"), ("loo", "--")]:
            ks = [int(k) for k in M[m] if f"{how}/dist" in M[m][k]["threshold"]]
            ax.plot(ks, [M[m][str(k)]["threshold"][f"{how}/dist"]["unknown_accept"][0] for k in ks], "o" + ls,
                    color=col[m], label=f"{MODELS[m]} ({how})")
    ax.set_xscale("log"); ax.set_xticks(SHOTS); ax.set_xticklabels(SHOTS); ax.set_xlabel("shots per class")
    ax.set_ylabel("unknown objects wrongly accepted"); ax.set_title("At 95 % acceptance of taught objects")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "false_accept.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
