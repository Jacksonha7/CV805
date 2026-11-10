#!/usr/bin/env python3
"""
Stage 4: CL training loop.

Iterates over task stream, trains fusion head with the selected strategy,
evaluates on ALL seen+unseen tasks after each round, records the accuracy
matrix, and saves results to JSON.

Usage:
    python cl/train_cl.py \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache/LongVideoBench \
        --config configs/tasks_lvb4.yaml \
        --strategy naive \
        --order default \
        --loss ce_oracle \
        --epochs 50 \
        --lr 1e-3 \
        --seed 0 \
        --out results/naive_default.json

Strategies: naive, ewc, lwf, er, joint, fixed, oracle
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.dataset import CLDataset
from cl.fusion_head import FusionHead
from cl.metrics import CLMetricTracker, evaluate_keyframes
from cl.oracle import LAMBDA_GRID, gt_to_score_index, normalize_minmax
from cl.strategies import STRATEGIES


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(args):
    set_seed(args.seed)

    ds = CLDataset(args.cache_dir, args.config, split_seed=args.seed,
                   feature_set=args.features)
    ds.stats()

    task_order = ds.get_task_order(args.order)
    n_tasks = len(task_order)
    print(f"Task order ({args.order}): {task_order}")

    # Prepare per-task train/test splits.
    # soft_topk needs each training sample's own score curves, so every sample
    # gets an integer id into a shared bank; ER's buffer and Joint's
    # concatenation carry the id along instead of a parallel array, which is
    # what keeps a replayed feature vector attached to its own curves.
    curve_bank = []

    def register_curves(sample):
        s_obj = torch.tensor(normalize_minmax(np.asarray(sample["s_obj"],
                                                         dtype=np.float64)),
                             dtype=torch.float32, device=args.device)
        s_text = torch.tensor(normalize_minmax(np.asarray(sample["s_text"],
                                                          dtype=np.float64)),
                              dtype=torch.float32, device=args.device)
        gt = np.round(gt_to_score_index(sample)).astype(np.int64)
        gt = np.unique(np.clip(gt, 0, len(s_obj) - 1))
        curve_bank.append((s_obj, s_text,
                           torch.tensor(gt, dtype=torch.long,
                                        device=args.device)))
        return len(curve_bank) - 1

    task_splits = {}
    for t_name in task_order:
        train, test = ds.get_task_splits(t_name)
        X_tr, y_tr = ds.as_numpy(train)
        X_te, y_te = ds.as_numpy(test)
        cid_tr = np.array([register_curves(s) for s in train], dtype=np.int64)
        task_splits[t_name] = {
            "train_X": X_tr, "train_y": y_tr,
            "test_X": X_te, "test_y": y_te,
            "train_samples": train, "test_samples": test,
            "curve_ids": cid_tr,
        }

    # Optional task-level labels.
    # Per-query λ* is mostly noise here: recall@8 is flat in λ for ~67% of
    # queries, so those labels are just the 0.5 tie-break. Averaging each
    # task's recall curves first recovers a clean signal (text tasks peak at
    # λ≈0.8, visual at λ≈0.0-0.3), and the head must still infer the task from
    # the query alone — so forgetting a task still means losing its λ.
    # The peak is taken over the TRAIN split only; using test curves here
    # would leak the evaluation target into the label.
    task_lambda_bins = {}
    if args.label == "task_oracle":
        for t_name in task_order:
            sp = task_splits[t_name]
            curves = np.stack([np.asarray(s["recall_curve"], dtype=float)
                               for s in sp["train_samples"]])
            b = int(curves.mean(axis=0).argmax())
            task_lambda_bins[t_name] = b
            sp["train_y"] = np.full(len(sp["train_y"]), b, dtype=np.int64)
            sp["test_y"] = np.full(len(sp["test_y"]), b, dtype=np.int64)
        print("Task-level λ targets: " + ", ".join(
            f"{t}={LAMBDA_GRID[b]:.1f}" for t, b in task_lambda_bins.items()))

    # Optional z-scoring, fitted on the FIRST task's train split only: later
    # tasks are not available yet at that point in a CL stream.
    mu = sd = None
    if args.standardize:
        X0 = task_splits[task_order[0]]["train_X"]
        mu, sd = X0.mean(0), X0.std(0) + 1e-6
        for sp in task_splits.values():
            sp["train_X"] = (sp["train_X"] - mu) / sd
            sp["test_X"] = (sp["test_X"] - mu) / sd

    # Build model + strategy
    in_dim = task_splits[task_order[0]]["train_X"].shape[1]
    model = FusionHead(in_dim=in_dim, n_bins=ds.n_bins).to(args.device)
    print(f"FusionHead params: {model.param_count()}")

    strategy_cls = STRATEGIES[args.strategy]
    strategy_kwargs = {"lr": args.lr, "loss_type": args.loss,
                       "device": args.device, "tau": args.tau,
                       "decode": args.decode}
    if args.strategy == "ewc":
        strategy_kwargs["ewc_lambda"] = args.ewc_lambda
    elif args.strategy == "er":
        strategy_kwargs["buffer_size"] = args.buffer_size
    elif args.strategy == "lwf":
        strategy_kwargs["lwf_alpha"] = args.lwf_alpha
    elif args.strategy == "fixed":
        strategy_kwargs["fixed_lambda"] = args.fixed_lambda
    elif args.strategy == "knn":
        strategy_kwargs["k"] = args.knn_k

    strategy = strategy_cls(model, **strategy_kwargs)
    strategy.curve_bank = curve_bank
    strategy.lambda_grid = torch.tensor(LAMBDA_GRID, dtype=torch.float32,
                                        device=args.device)

    tracker = CLMetricTracker(n_tasks)
    lambda_acc_matrix = np.zeros((n_tasks, n_tasks))
    results_per_round = []

    def eval_task(strat, split):
        res = strat.evaluate(split["test_X"], split["test_y"])
        kf = evaluate_keyframes(split["test_samples"], res["lambdas"],
                                k=8, threshold_sec=5.0)
        return res, kf

    # b_j for FWT is an untrained head, per the project spec. It gets its own
    # FusionHead so that constructing it cannot perturb the strategy's weights.
    baseline_head = FusionHead(in_dim=in_dim, n_bins=ds.n_bins).to(args.device)
    baseline_strategy = STRATEGIES["naive"](
        baseline_head, lr=args.lr, loss_type=args.loss, device=args.device,
        decode=args.decode)
    baseline_recall = np.zeros(n_tasks)

    # Fixed-λ VSI is the headline non-learning baseline; report it alongside.
    fixed_strategy = STRATEGIES["fixed"](
        model, fixed_lambda=args.fixed_lambda, device=args.device)
    fixed_recall = np.zeros(n_tasks)

    for j, t_name in enumerate(task_order):
        sp = task_splits[t_name]
        _, kf_base = eval_task(baseline_strategy, sp)
        baseline_recall[j] = kf_base["recall"]
        _, kf_fixed = eval_task(fixed_strategy, sp)
        fixed_recall[j] = kf_fixed["recall"]

    t0 = time.time()
    for t_idx, t_name in enumerate(task_order):
        print(f"\n{'='*60}")
        print(f"Task {t_idx}: {t_name}")
        print(f"{'='*60}")

        sp = task_splits[t_name]
        strategy.train_task(
            sp["train_X"], sp["train_y"],
            epochs=args.epochs, batch_size=args.batch_size,
            curve_ids=sp["curve_ids"],
        )

        # Evaluate on ALL tasks.
        # R_{i,j} is keyframe recall@8, NOT λ-bin accuracy: predicting bin 4
        # instead of 5 counts as a classification error yet barely moves
        # retrieval, while 0 instead of 10 is also a single error and is
        # catastrophic. The project's claim is about retrieval quality, so the
        # CL matrix is built on recall; λ-accuracy is kept as a side series.
        round_results = {}
        for j, eval_name in enumerate(task_order):
            esp = task_splits[eval_name]
            res, kf = eval_task(strategy, esp)
            tracker.record(t_idx, j, kf["recall"])
            lambda_acc_matrix[t_idx, j] = res["accuracy"]

            pred_lambdas = res["lambdas"]
            round_results[eval_name] = {
                "lambda_acc": res["accuracy"],
                "pred_lambda_mean": float(pred_lambdas.mean()),
                "pred_lambda_std": float(pred_lambdas.std()),
                # histogram of the λ used for fusion, rounded to the grid
                "pred_lambda_hist": np.bincount(
                    np.rint(pred_lambdas * (ds.n_bins - 1)).astype(np.int64),
                    minlength=ds.n_bins).tolist(),
                "argmax_bin_hist": np.bincount(
                    res["predictions"], minlength=ds.n_bins).tolist(),
                **kf,
            }
            status = "CURRENT" if j == t_idx else (
                "PAST" if j < t_idx else "FUTURE")
            print(f"  [{status}] {eval_name}: "
                  f"λ-acc={res['accuracy']:.4f}  "
                  f"R@8={kf['recall']:.4f}  F1={kf['f1']:.4f}")

        results_per_round.append(round_results)

    elapsed = time.time() - t0
    cl_summary = tracker.summary(baseline_recall)
    print(f"\n{'='*60}")
    print(f"CL Summary ({args.strategy}, {args.order})")
    print(f"  ACC={cl_summary['ACC']:.4f}  "
          f"FM={cl_summary['FM']:.4f}  "
          f"BWT={cl_summary['BWT']:.4f}  "
          f"FWT={cl_summary.get('FWT', 0):.4f}")
    print(f"  Time: {elapsed:.1f}s")

    output = {
        "strategy": args.strategy,
        "order": args.order,
        "loss": args.loss,
        "decode": args.decode,
        "features": args.features, "standardize": args.standardize,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "task_order": task_order,
        "cl_metrics": cl_summary,
        "matrix_metric": "keyframe_recall@8",
        "label_mode": args.label,
        "task_lambda_bins": task_lambda_bins,
        "lambda_acc_matrix": lambda_acc_matrix.tolist(),
        "baseline_recall": baseline_recall.tolist(),
        "fixed_lambda_recall": fixed_recall.tolist(),
        "per_round": results_per_round,
        "elapsed_sec": elapsed,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    # Checkpoint the trained head so cl/run_downstream_qa.py can reproduce this
    # run's frame selection without retraining.
    ckpt_path = os.path.splitext(args.out)[0] + ".pt"
    torch.save({"state_dict": model.state_dict(),
                "in_dim": in_dim, "n_bins": ds.n_bins,
                "strategy": args.strategy, "order": args.order,
                "loss": args.loss, "decode": args.decode,
                "features": args.features,
                "feat_mean": None if mu is None else torch.tensor(mu),
                "feat_std": None if sd is None else torch.tensor(sd),
                "seed": args.seed}, ckpt_path)
    print(f"Saved head → {ckpt_path}")
    print(f"\nSaved → {args.out}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="configs/tasks_lvb4.yaml")
    ap.add_argument("--features", default="base778",
                    choices=["base778", "stats10_pq44"])
    ap.add_argument("--standardize", action="store_true",
                    help="z-score features with first-task train stats")
    ap.add_argument("--strategy", default="naive",
                    choices=list(STRATEGIES.keys()))
    ap.add_argument("--order", default="default")
    ap.add_argument("--loss", default="ce_oracle",
                    choices=["ce_oracle", "soft_topk"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    # Strategy-specific
    ap.add_argument("--ewc_lambda", type=float, default=400.0)
    ap.add_argument("--buffer_size", type=int, default=200)
    ap.add_argument("--lwf_alpha", type=float, default=1.0)
    ap.add_argument("--label", default="query_oracle",
                    choices=["query_oracle", "task_oracle"],
                    help="per-query lambda* (noisy) vs per-task optimum")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="soft_topk softmax temperature over frames")
    ap.add_argument("--fixed_lambda", type=float, default=0.3)
    ap.add_argument("--decode", default="auto",
                    choices=["auto", "expected", "argmax"],
                    help="how logits become λ at eval; auto = expected for "
                         "soft_topk (matches training), argmax for ce_oracle")
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--out", default="results/experiment.json")
    args = ap.parse_args()
    if args.decode == "auto":
        args.decode = "expected" if args.loss == "soft_topk" else "argmax"

    run_experiment(args)


if __name__ == "__main__":
    main()
