#!/usr/bin/env python3
"""
Diagnose why learned λ under-performs fixed λ=0.3 (recall@8 on test splits).

D1  constant-λ ceiling: recall@8 for every constant λ, and a per-task
    constant chosen on train (= what a head with perfect task ID can reach).
D2  surrogate alignment: the λ that minimises soft_topk loss (per query, and
    per task on train) vs the λ that maximises recall@8.
D3  learnability: non-CL joint head trained with soft_topk, feature variants,
    train vs test recall, argmax vs expected-λ decoding, predicted λ spread.

Usage:
    python cl/diagnose_lambda.py \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache/LongVideoBench \
        --out results_diag/diagnose.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.dataset import CLDataset
from cl.fusion_head import FusionHead, soft_topk_loss
from cl.oracle import (LAMBDA_GRID, gt_to_score_index, normalize_minmax,
                       tolerance_steps)


def recall_grid(s):
    """recall@8 (±5 s) for every λ on the grid — one row per query."""
    o, t = normalize_minmax(s["s_obj"]), normalize_minmax(s["s_text"])
    gt = gt_to_score_index(s)
    th = tolerance_steps(s, 5.0)
    out = np.zeros(len(LAMBDA_GRID))
    if len(gt) == 0:
        return out
    for i, lam in enumerate(LAMBDA_GRID):
        top = np.argsort((1 - lam) * o + lam * t)[-8:]
        out[i] = np.any(np.abs(gt[:, None] - top[None, :]) <= th, 1).mean()
    return out


def surrogate_grid(s, tau):
    """soft_topk loss for every λ on the grid (lower is better)."""
    o, t = normalize_minmax(s["s_obj"]), normalize_minmax(s["s_text"])
    gt = np.unique(np.clip(np.round(gt_to_score_index(s)).astype(int),
                           0, len(o) - 1))
    out = np.zeros(len(LAMBDA_GRID))
    if len(gt) == 0:
        return out
    for i, lam in enumerate(LAMBDA_GRID):
        z = ((1 - lam) * o + lam * t) / tau
        logp = z - np.logaddexp.reduce(z)
        out[i] = -np.logaddexp.reduce(logp[gt])
    return out


def curve_tuple(s, device):
    o = torch.tensor(normalize_minmax(np.asarray(s["s_obj"], float)),
                     dtype=torch.float32, device=device)
    t = torch.tensor(normalize_minmax(np.asarray(s["s_text"], float)),
                     dtype=torch.float32, device=device)
    gt = np.unique(np.clip(np.round(gt_to_score_index(s)).astype(int),
                           0, len(o) - 1))
    return o, t, torch.tensor(gt, dtype=torch.long, device=device)


def train_head(Xtr, curves, in_dim, hidden, epochs, lr, tau, seed, wd=1e-4):
    torch.manual_seed(seed)
    if hidden == 0:
        model = torch.nn.Linear(in_dim, len(LAMBDA_GRID))
    else:
        model = FusionHead(in_dim=in_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    grid = torch.tensor(LAMBDA_GRID, dtype=torch.float32)
    X = torch.tensor(Xtr, dtype=torch.float32)
    n = len(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            loss = soft_topk_loss(model(X[idx]), [curves[j] for j in idx],
                                  grid, temperature=tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def decode(model, X):
    logits = model(torch.tensor(X, dtype=torch.float32))
    p = F.softmax(logits, -1).numpy()
    argmax = p.argmax(1)
    expect = np.clip(np.round((p * LAMBDA_GRID).sum(1) * 10), 0, 10).astype(int)
    return argmax, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="configs/tasks_lvb4.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="results_diag/diagnose.json")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    taus = [0.02, 0.05, 0.1, 0.3, 1.0]
    report = {"seeds": {}}

    for seed in seeds:
        ds = CLDataset(args.cache_dir, args.config, split_seed=seed)
        tasks = ds.task_names
        tr, te = {}, {}
        for t in tasks:
            tr[t], te[t] = ds.get_task_splits(t)
        R = {id(s): recall_grid(s) for s in ds.samples}
        S = {tau: {id(s): surrogate_grid(s, tau) for s in ds.samples}
             for tau in taus}
        all_tr = [s for t in tasks for s in tr[t]]
        all_te = [s for t in tasks for s in te[t]]
        rec = lambda ss, idx: float(np.mean([R[id(s)][i]
                                             for s, i in zip(ss, idx)]))
        res = {}

        # ---- D1: constant λ
        const_te = np.mean([R[id(s)] for s in all_te], 0)
        const_tr = np.mean([R[id(s)] for s in all_tr], 0)
        res["D1_const_curve_test"] = const_te.round(4).tolist()
        res["D1_fixed0.3_test"] = float(const_te[3])
        res["D1_best_const_on_train"] = float(LAMBDA_GRID[const_tr.argmax()])
        res["D1_best_const_on_train_test_recall"] = float(
            const_te[const_tr.argmax()])
        per_task = {}
        idx_te = []
        for t in tasks:
            b = int(np.mean([R[id(s)] for s in tr[t]], 0).argmax())
            per_task[t] = {
                "lambda_train_opt": float(LAMBDA_GRID[b]),
                "test_curve": np.mean([R[id(s)] for s in te[t]], 0)
                .round(3).tolist()}
            idx_te += [b] * len(te[t])
        res["D1_per_task"] = per_task
        res["D1_task_const_test_recall"] = rec(all_te, idx_te)
        res["D1_query_oracle_test_recall"] = float(
            np.mean([R[id(s)].max() for s in all_te]))
        res["D1_frac_flat_curves_test"] = float(
            np.mean([np.ptp(R[id(s)]) == 0 for s in all_te]))

        # ---- D2: surrogate alignment
        d2 = {}
        for tau in taus:
            q_idx = [int(S[tau][id(s)].argmin()) for s in all_te]
            t_idx, t_lams = [], {}
            for t in tasks:
                b = int(np.mean([S[tau][id(s)] for s in tr[t]], 0).argmin())
                t_lams[t] = float(LAMBDA_GRID[b])
                t_idx += [b] * len(te[t])
            g = int(np.mean([S[tau][id(s)] for s in all_tr], 0).argmin())
            d2[str(tau)] = {
                "per_query_surrogate_opt_recall": rec(all_te, q_idx),
                "per_task_surrogate_opt_lambda": t_lams,
                "per_task_surrogate_opt_recall": rec(all_te, t_idx),
                "global_surrogate_opt_lambda": float(LAMBDA_GRID[g]),
                "global_surrogate_opt_recall": float(const_te[g]),
            }
        res["D2_surrogate"] = d2

        # ---- D3: learnability (joint, non-CL)
        Xtr = np.stack([s["feat"] for s in all_tr])
        Xte = np.stack([s["feat"] for s in all_te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        variants = {
            "raw_778_mlp": (Xtr, Xte, 128),
            "std_778_mlp": ((Xtr - mu) / sd, (Xte - mu) / sd, 128),
            "std_stats10_mlp": (((Xtr - mu) / sd)[:, 768:],
                                ((Xte - mu) / sd)[:, 768:], 128),
            "std_qemb768_linear": (((Xtr - mu) / sd)[:, :768],
                                   ((Xte - mu) / sd)[:, :768], 0),
        }
        d3 = {}
        for tau in [0.1, 1.0]:
            curves = [curve_tuple(s, "cpu") for s in all_tr]
            for name, (A, B, h) in variants.items():
                m = train_head(A, curves, A.shape[1], h, args.epochs, 1e-3,
                               tau, seed)
                am_tr, ex_tr = decode(m, A)
                am_te, ex_te = decode(m, B)
                d3[f"{name}_tau{tau}"] = {
                    "train_recall_argmax": rec(all_tr, am_tr),
                    "test_recall_argmax": rec(all_te, am_te),
                    "test_recall_expect": rec(all_te, ex_te),
                    "test_pred_lambda_hist": np.bincount(
                        am_te, minlength=11).tolist(),
                }
                print(seed, name, tau, d3[f"{name}_tau{tau}"], flush=True)
        res["D3_joint_head"] = d3
        report["seeds"][seed] = res
        print(json.dumps({k: v for k, v in res.items()
                          if k.startswith(("D1_f", "D1_b", "D1_t", "D1_q"))}),
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("Saved →", args.out)


if __name__ == "__main__":
    main()
