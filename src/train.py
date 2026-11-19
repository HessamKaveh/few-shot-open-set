"""Train the two embedding networks on the 64 base classes only.

    python src/train.py --method protonet     # Conv4, episodic Prototypical Networks training (Snell et al., 2017)
    python src/train.py --method pretrain     # ResNet-12, plain 64-way classification ("RFS", Tian et al., 2020)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from model import Conv4, ResNet12

ROOT = data.ROOT


def protonet(X, Y, s, dev, episodes):
    model = Conv4().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=episodes // 4, gamma=0.5)
    n, k, q = 20, 5, 15                       # train "higher-way" than test, as in the paper
    loader = DataLoader(data.Episodes(X, Y, s["base"], n, k, q), batch_size=None, num_workers=12)
    y = torch.arange(n, device=dev).repeat_interleave(q)
    hist, t0, run = [], time.time(), []
    for ep, (xs, xq) in enumerate(loader):
        if ep == episodes:
            break
        z = model(torch.cat([xs, xq]).to(dev, non_blocking=True))
        proto = z[: n * k].view(n, k, -1).mean(1)
        logits = -torch.cdist(z[n * k:], proto) ** 2
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run.append((loss.item(), (logits.argmax(1) == y).float().mean().item()))
        if (ep + 1) % 500 == 0:
            l, a = np.mean(run, 0); run = []
            hist.append((ep + 1, float(l), float(a)))
            print(f"protonet ep {ep + 1}/{episodes} loss {l:.3f} acc(20-way) {a:.3f} {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    return model, hist


def pretrain(X, Y, s, dev, epochs):
    base = np.isin(Y, s["base"])
    remap = {c: i for i, c in enumerate(s["base"])}
    yb = np.array([remap[c] for c in Y[base]])
    loader = DataLoader(data.Images(X[base], yb, train=True), batch_size=64, shuffle=True, drop_last=True,
                        num_workers=12, persistent_workers=True)
    model = ResNet12(num_classes=64).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    hist, t0 = [], time.time()
    for ep in range(epochs):
        model.train(); tot = cor = m = 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model.fc(model(x))
                loss = F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            tot += loss.item() * len(y); cor += (out.argmax(1) == y).sum().item(); m += len(y)
        hist.append((ep + 1, tot / m, cor / m))
        print(f"pretrain ep {ep + 1}/{epochs} loss {tot / m:.3f} acc(64-way train) {cor / m:.3f} "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["protonet", "pretrain"], required=True)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    X, Y, _, s, _ = data.split()
    dev = torch.device("cuda")
    if a.method == "protonet":
        model, hist = protonet(X, Y, s, dev, a.episodes)
    else:
        model, hist = pretrain(X, Y, s, dev, a.epochs)
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "history"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "cache", f"{a.method}.pt"))
    json.dump(hist, open(os.path.join(ROOT, "results", "history", f"{a.method}.json"), "w"))


if __name__ == "__main__":
    main()
