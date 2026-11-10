#!/usr/bin/env python3
"""
Stage 5: Analyze CL experiment results and produce figures.

Reads results/*.json files and generates:
  1. CL metrics comparison table (ACC, FM, BWT, FWT)
  2. Accuracy matrix heatmaps (one per strategy)
  3. Per-task recall@8 bar chart across strategies
  4. Lambda distribution shift plot

Usage:
    python cl/analyze.py --results_dir results/ --fig_dir figures/
"""
import argparse
import json
import glob
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — text tables only")


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    results = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        key = f"{data['strategy']}_{data['order']}"
        results[key] = data
    return results


def print_summary_table(results):
    print(f"\n{'Strategy':<20} {'Order':<12} {'ACC':>6} {'FM':>6} "
          f"{'BWT':>7} {'FWT':>7} {'R@8':>6}")
    print("-" * 70)
    for key, r in sorted(results.items()):
        cm = r["cl_metrics"]
        # Last round average recall
        last_round = r["per_round"][-1]
        avg_recall = np.mean([v["recall"] for v in last_round.values()])
        print(f"{r['strategy']:<20} {r['order']:<12} "
              f"{cm['ACC']:6.4f} {cm['FM']:6.4f} "
              f"{cm['BWT']:7.4f} {cm.get('FWT', 0):7.4f} "
              f"{avg_recall:6.4f}")


def plot_accuracy_matrix(results, fig_dir):
    if not HAS_MPL:
        return
    for key, r in results.items():
        matrix = np.array(r["cl_metrics"]["matrix"])
        n = matrix.shape[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        tasks = r["task_order"]
        short = [t.split("_")[0] for t in tasks]
        ax.set_xticklabels(short, fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Evaluated on task")
        ax.set_ylabel("After training task")
        ax.set_title(f"{r['strategy']} ({r['order']})")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.5 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"matrix_{key}.pdf"), dpi=150)
        plt.close(fig)


def plot_strategy_comparison(results, fig_dir):
    if not HAS_MPL:
        return
    strategies = []
    acc_vals, fm_vals, bwt_vals = [], [], []
    for key, r in sorted(results.items()):
        strategies.append(f"{r['strategy']}\n{r['order']}")
        cm = r["cl_metrics"]
        acc_vals.append(cm["ACC"])
        fm_vals.append(cm["FM"])
        bwt_vals.append(cm["BWT"])

    x = np.arange(len(strategies))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 5))
    ax.bar(x - w, acc_vals, w, label="ACC", color="#4C72B0")
    ax.bar(x, fm_vals, w, label="FM", color="#DD8452")
    ax.bar(x + w, bwt_vals, w, label="BWT", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("CL Strategy Comparison")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "strategy_comparison.pdf"), dpi=150)
    plt.close(fig)


def plot_per_task_recall(results, fig_dir):
    if not HAS_MPL:
        return
    # Pick first result to get task names
    first = next(iter(results.values()))
    tasks = first["task_order"]
    short_tasks = [t.split("_", 1)[1] if "_" in t else t for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    n_strat = len(results)
    w = 0.8 / max(n_strat, 1)

    for i, (key, r) in enumerate(sorted(results.items())):
        last_round = r["per_round"][-1]
        recalls = [last_round[t]["recall"] for t in tasks]
        ax.bar(x + i * w - 0.4 + w / 2, recalls, w,
               label=f"{r['strategy']}")

    ax.set_xticks(x)
    ax.set_xticklabels(short_tasks, fontsize=8, rotation=15)
    ax.set_ylabel("Recall@8")
    ax.set_title("Per-task recall after full training")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_task_recall.pdf"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--fig_dir", default="figures/")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    results = load_results(args.results_dir)

    if not results:
        print("No result files found in", args.results_dir)
        return

    print_summary_table(results)
    plot_accuracy_matrix(results, args.fig_dir)
    plot_strategy_comparison(results, args.fig_dir)
    plot_per_task_recall(results, args.fig_dir)

    if HAS_MPL:
        print(f"\nFigures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
