"""CIFAR-FS-style few-shot split of CIFAR-100: 64 base / 16 validation / 20 novel classes (600 images each).

The class split is a seeded random permutation of the 100 fine classes (seed 0), stored in results/split.json.
CIFAR-100's 20 coarse superclasses are kept so that an unknown object can be labelled "near" (it shares a
superclass with one of the taught classes, e.g. a taught "palm tree" vs an unknown "pine tree") or "far".
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DATA = os.environ.get("CIFAR_DIR", os.path.expanduser("~/datasets/portfolio/cifar"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)


def load_cifar100():
    """All 60,000 CIFAR-100 images (train + test parts; few-shot splits are by class, not by the original split).
    Expects cifar100_{train,test}.npz with X (n,32,32,3) uint8, fine, coarse, fine_names (see README)."""
    X, Y, C = [], [], []
    for part in ["train", "test"]:
        d = np.load(os.path.join(DATA, f"cifar100_{part}.npz"))
        X.append(d["X"]); Y.append(d["fine"]); C.append(d["coarse"])
        names = [str(n) for n in d["fine_names"]]
    return np.concatenate(X), np.concatenate(Y), np.concatenate(C), names


def split():
    X, Y, C, names = load_cifar100()
    perm = np.random.default_rng(0).permutation(100)
    s = {"base": sorted(perm[:64].tolist()), "val": sorted(perm[64:80].tolist()), "novel": sorted(perm[80:].tolist())}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({k: [names[c] for c in v] for k, v in s.items()}, open(os.path.join(ROOT, "results", "split.json"), "w"),
              indent=1)
    coarse_of = {int(y): int(c) for y, c in zip(Y, C)}
    return X, Y, coarse_of, s, names


class Images(Dataset):
    def __init__(self, X, Y, train=False):
        self.X, self.Y = X, Y
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4)] if train else []
        self.t = T.Compose([T.ToPILImage()] + aug + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.t(self.X[i]), int(self.Y[i])


class Episodes(torch.utils.data.IterableDataset):
    """Infinite stream of N-way K-shot training episodes from the base classes."""

    def __init__(self, X, Y, classes, n_way, k_shot, q, seed=0):
        self.by_class = {c: np.where(Y == c)[0] for c in classes}
        self.X, self.classes, self.n, self.k, self.q, self.seed = X, list(classes), n_way, k_shot, q, seed
        self.t = Images(X, Y, train=True).t

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (w.id if w else 0) * 7919)
        while True:
            cls = rng.choice(self.classes, self.n, replace=False)
            idx = [rng.choice(self.by_class[c], self.k + self.q, replace=False) for c in cls]
            xs = torch.stack([self.t(self.X[i]) for row in idx for i in row[: self.k]])
            xq = torch.stack([self.t(self.X[i]) for row in idx for i in row[self.k:]])
            yield xs, xq
