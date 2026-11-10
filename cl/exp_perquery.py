#!/usr/bin/env python3
"""
Do per-query features (cl/perquery_features.py) let a head beat fixed λ?

Non-CL check: one head trained on the union of all task train splits
(soft_topk, τ=0.1, expected-λ decode), evaluated on the union of test splits,
3 split seeds. Feature sets are compared against each other, fixed λ=0.3, the
best per-task constant, the no-learning consensus-λ heuristic, and a
gradient-boosted regressor that predicts the 11-point recall curve directly.

Usage:
    python cl/exp_perquery.py \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache/LongVideoBench \
        --out results_diag/perquery.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.dataset import CLDataset
from cl.diagnose_lambda import curve_tuple, decode, recall_grid, train_head
from cl.oracle import LAMBDA_GRID
from cl.perquery_features import perquery_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="configs/tasks_lvb4.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="results_diag/perquery.json")
    args = ap.parse_args()

    report = {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        ds = CLDataset(args.cache_dir, args.config, split_seed=seed)
        tr, te = [], []
        for t in ds.task_names:
            a, b = ds.get_task_splits(t)
            tr += a
            te += b
        for s in ds.samples:
            s["pq"] = perquery_features(s)
            s["R"] = recall_grid(s)
        rec = lambda ss, lam: float(np.mean(
            [s["R"][int(round(l * 10))] for s, l in zip(ss, lam)]))
        flat_te = np.array([np.ptp(s["R"]) == 0 for s in te])

        res = {"fixed0.3": rec(te, [0.3] * len(te)),
               "oracle": float(np.mean([s["R"].max() for s in te])),
               "consensus_heuristic": rec(te, [s["pq"][-1] for s in te])}
        task_best = {}
        for t in ds.task_names:
            task_best[t] = LAMBDA_GRID[int(np.mean(
                [s["R"] for s in tr if s["task"] == t], 0).argmax())]
        res["task_const"] = rec(te, [task_best[s["task"]] for s in te])

        base = lambda s: s["feat"]
        sets = {
            "base778": base,
            "stats10": lambda s: s["feat"][768:],
            "pq44": lambda s: s["pq"],
            "stats10+pq44": lambda s: np.concatenate([s["feat"][768:],
                                                      s["pq"]]),
            "base778+pq44": lambda s: np.concatenate([s["feat"], s["pq"]]),
        }
        curves = [curve_tuple(s, "cpu") for s in tr]
        for name, fn in sets.items():
            A = np.stack([fn(s) for s in tr])
            B = np.stack([fn(s) for s in te])
            mu, sd = A.mean(0), A.std(0) + 1e-6
            A, B = (A - mu) / sd, (B - mu) / sd
            m = train_head(A, curves, A.shape[1], 128, args.epochs, 1e-3,
                           0.1, seed)
            _, ex_te = decode(m, B)
            _, ex_tr = decode(m, A)
            lam_te = ex_te / 10
            res[f"head_{name}"] = {
                "test": rec(te, lam_te), "train": rec(tr, ex_tr / 10),
                "test_nonflat": rec([s for s, f in zip(te, flat_te) if not f],
                                    lam_te[~flat_te]),
                "pred_lambda_hist": np.bincount(ex_te, minlength=11).tolist()}
            # gradient boosting: regress the recall curve, pick its argmax
            A_raw = np.stack([fn(s) for s in tr])
            B_raw = np.stack([fn(s) for s in te])
            Y = np.stack([s["R"] for s in tr])
            P = np.stack([HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=3,
                random_state=seed).fit(A_raw, Y[:, b]).predict(B_raw)
                for b in range(11)], 1)
            res[f"gbm_{name}"] = rec(te, LAMBDA_GRID[P.argmax(1)])
            print(seed, name, res[f"head_{name}"]["test"],
                  res[f"gbm_{name}"], flush=True)
        res["nonflat_ref"] = {
            "fixed0.3": rec([s for s, f in zip(te, flat_te) if not f],
                            [0.3] * int((~flat_te).sum())),
            "frac_nonflat": float((~flat_te).mean())}
        report[seed] = res
        print(seed, {k: v for k, v in res.items()
                     if not isinstance(v, dict)}, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("Saved →", args.out)


if __name__ == "__main__":
    main()
