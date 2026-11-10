"""
Stage 2c: CL dataset — loads cached .npz files, assigns task labels,
and yields (features, oracle_λ_idx, task_id, category) tuples.
"""
import glob
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .features import extract_features


class CLDataset:
    """In-memory dataset built from cached .npz files + task YAML."""

    def __init__(self, cache_dir: str, config_path: str, split_seed: int = 0,
                 feature_set: str = "base778"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.task_defs = cfg["tasks"]
        self.orders = cfg["orders"]
        self.train_ratio = cfg["split"]["train"]
        self.n_bins = cfg["lambda_bins"]
        self.split_seed = split_seed

        # Build category -> task_name map
        self.cat2task: Dict[str, str] = {}
        for task_name, cats in self.task_defs.items():
            for cat in cats:
                self.cat2task[cat] = task_name

        self.task_names = list(self.task_defs.keys())
        self.task2id = {t: i for i, t in enumerate(self.task_names)}

        # Load all .npz
        self.samples: List[dict] = []
        npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        unknown_cats: Dict[str, int] = {}
        missing_oracle: List[str] = []
        for path in npz_files:
            data = dict(np.load(path, allow_pickle=True))
            cat = str(data.get("category", ""))
            if cat not in self.cat2task:
                unknown_cats[cat] = unknown_cats.get(cat, 0) + 1
                continue
            task_name = self.cat2task[cat]
            feat = extract_features(data, feature_set)
            if "oracle_lambda_idx" not in data:
                missing_oracle.append(os.path.basename(path))
                continue
            label = int(data["oracle_lambda_idx"])
            self.samples.append({
                "feat": feat,
                "label": label,
                "task": task_name,
                "task_id": self.task2id[task_name],
                "category": cat,
                "qid": os.path.splitext(os.path.basename(path))[0],
                "s_obj": data["s_obj"],
                "s_text": data["s_text"],
                "gt_frames": data["gt_frames"],
                "n_frames": int(data.get("n_frames", len(data["s_obj"]))),
                # duration is required by oracle.gt_to_score_index to convert
                # raw GT frame numbers into score-curve indices.
                "duration": float(data.get("duration", 0.0) or 0.0),
                "recall_curve": data.get("recall_curve", None),
            })
        # Fail loudly rather than quietly shrinking the dataset: a renamed or
        # unmapped category would otherwise just make a task look small, and a
        # cache written before oracle.py ran would silently become all-label-0.
        if unknown_cats:
            listing = ", ".join(f"{c!r}×{n}" for c, n in
                                sorted(unknown_cats.items(),
                                       key=lambda kv: -kv[1]))
            raise ValueError(
                f"{sum(unknown_cats.values())} cached queries carry a category "
                f"absent from {os.path.basename(config_path)}: {listing}. "
                f"Known categories: {sorted(self.cat2task)}. "
                f"Add them to the config or filter the cache deliberately."
            )
        if missing_oracle:
            raise ValueError(
                f"{len(missing_oracle)} cached queries have no "
                f"'oracle_lambda_idx' (e.g. {missing_oracle[:5]}). "
                f"Run cl/oracle.py over the cache first."
            )
        print(f"[CLDataset] Loaded {len(self.samples)} queries across "
              f"{len(self.task_names)} tasks")

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_task_splits(self, task_name: str
                        ) -> Tuple[List[dict], List[dict]]:
        """Return (train, test) for one task, deterministic split."""
        task_samples = [s for s in self.samples if s["task"] == task_name]
        rng = random.Random(self.split_seed)
        rng.shuffle(task_samples)
        n_train = int(len(task_samples) * self.train_ratio)
        return task_samples[:n_train], task_samples[n_train:]

    def get_task_order(self, order_name: str = "default") -> List[str]:
        order = self.orders[order_name]
        if order == "shuffle":
            rng = random.Random(self.split_seed)
            names = list(self.task_names)
            rng.shuffle(names)
            return names
        return list(order)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def as_numpy(self, samples: List[dict]):
        X = np.stack([s["feat"] for s in samples])
        y = np.array([s["label"] for s in samples], dtype=np.int64)
        return X, y

    def stats(self):
        from collections import Counter
        task_counts = Counter(s["task"] for s in self.samples)
        cat_counts = Counter(s["category"] for s in self.samples)
        labels = [s["label"] for s in self.samples]
        print(f"\n=== Dataset stats ===")
        print(f"Total samples: {len(self.samples)}")
        print(f"Tasks: {dict(task_counts)}")
        print(f"Categories: {dict(cat_counts)}")
        print(f"Label dist: mean={np.mean(labels):.2f} "
              f"std={np.std(labels):.2f}")
        if self.samples and self.samples[0].get("recall_curve") is not None:
            curves = np.stack([s["recall_curve"] for s in self.samples])
            print(f"Oracle recall: mean={curves.max(1).mean():.4f}")
        print()
