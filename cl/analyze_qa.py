#!/usr/bin/env python3
"""
Paired analysis of cl/run_downstream_qa.py output (results_qa/).

Each head is compared with the baselines on the test split of its own seed;
head − fixed(0.3) gets a bootstrap CI and an exact McNemar test. Also reports
per-task accuracy and QA accuracy split by whether keyframe recall@8 > 0.

    python cl/analyze_qa.py --qa_dir results_qa \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.dataset import CLDataset

BASE = ["uniform", "fixed:0.0", "fixed:0.3", "fixed:0.7", "fixed:1.0", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_dir", default="results_qa")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="configs/tasks_lvb4.yaml")
    ap.add_argument("--heads", default="lwf,naive")
    ap.add_argument("--order", default="default")
    ap.add_argument("--results", default="results_v4")
    args = ap.parse_args()

    def load(name):
        with open(f"{args.qa_dir}/{name.replace('/', '__')}.jsonl") as f:
            return {r["qid"]: r for r in map(json.loads, f)}

    base = {m: load(m) for m in BASE}
    tasks = sorted({r["task"] for r in base["uniform"].values()})
    rng = np.random.default_rng(0)
    report = {}
    for st in args.heads.split(","):
        acc = {k: [] for k in ["head"] + BASE}
        per_task = {t: {m: [] for m in ["head", "fixed:0.3", "oracle"]}
                    for t in tasks}
        diffs, b, c = [], 0, 0
        for s in range(3):
            H = load(f"{args.results}/{st}_{args.order}_seed{s}")
            acc["head"].append(float(np.mean([r["correct"] for r in H.values()])))
            for m in BASE:
                acc[m].append(float(np.mean([base[m][q]["correct"] for q in H])))
            for q, r in H.items():
                h, f = r["correct"], base["fixed:0.3"][q]["correct"]
                b += h and not f
                c += f and not h
                diffs.append(float(h) - float(f))
                for m in per_task[r["task"]]:
                    per_task[r["task"]][m].append(
                        (r if m == "head" else base[m][q])["correct"])
        diffs = np.array(diffs)
        boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
        report[st] = {
            "acc_mean": {m: float(np.mean(v)) for m, v in acc.items()},
            "acc_per_seed": acc,
            "head_minus_fixed0.3": float(diffs.mean()),
            "ci95": [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))],
            "discordant": {"head_only": int(b), "fixed_only": int(c)},
            "mcnemar_p": float(binomtest(b, b + c).pvalue) if b + c else 1.0,
            "per_task": {t: {m: float(np.mean(v)) for m, v in d.items()}
                         for t, d in per_task.items()},
        }

    ds = CLDataset(args.cache_dir, args.config)
    rc = {s["qid"]: s["recall_curve"] for s in ds.samples}
    report["all753"] = {}
    for m in BASE:
        rows = base[m]
        e = {"acc": float(np.mean([r["correct"] for r in rows.values()])),
             "per_task": {t: float(np.mean([r["correct"] for r in rows.values()
                                            if r["task"] == t]))
                          for t in tasks}}
        if m.startswith("fixed:"):
            li = int(round(float(m[6:]) * 10))
            r = np.array([rc[q][li] for q in rows])
            y = np.array([rows[q]["correct"] for q in rows])
            e.update(recall=float(r.mean()),
                     qa_if_recall0=float(y[r == 0].mean()),
                     qa_if_recall_pos=float(y[r > 0].mean()))
        report["all753"][m] = e

    out = f"{args.qa_dir}/analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "all753"},
                     indent=1)[:3000])
    print("Saved →", out)


if __name__ == "__main__":
    main()
