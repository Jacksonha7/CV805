"""
Stage 3b: Keyframe retrieval + CL metrics.

Keyframe metrics (per-query):
    recall@k, precision@k, F1@k — with proximity matching (threshold*fps)

CL metrics (over task sequence):
    ACC  — average accuracy across all tasks after training the last task
    FM   — forgetting measure: avg drop in accuracy on earlier tasks
    BWT  — backward transfer: mean(acc_T,i - acc_i,i) for i < T
    FWT  — forward transfer:  mean(acc_i-1,i - baseline_i) for i > 0
"""
import numpy as np
from .oracle import (normalize_minmax, LAMBDA_GRID,
                     gt_to_score_index, tolerance_steps)


def recall_at_k(pred_frames: np.ndarray, gt_frames: np.ndarray,
                threshold_frames: float) -> float:
    if len(gt_frames) == 0:
        return 0.0
    dists = np.abs(gt_frames[:, None] - pred_frames[None, :])
    covered = np.any(dists <= threshold_frames, axis=1).sum()
    return covered / len(gt_frames)


def precision_at_k(pred_frames: np.ndarray, gt_frames: np.ndarray,
                   threshold_frames: float) -> float:
    if len(pred_frames) == 0:
        return 0.0
    dists = np.abs(pred_frames[:, None] - gt_frames[None, :])
    covered = np.any(dists <= threshold_frames, axis=1).sum()
    return covered / len(pred_frames)


def f1_score(p: float, r: float) -> float:
    if p + r < 1e-12:
        return 0.0
    return 2 * p * r / (p + r)


def lambda_accuracy(pred_idx: np.ndarray, gt_idx: np.ndarray) -> float:
    return (pred_idx == gt_idx).mean()


def evaluate_keyframes(samples: list, pred_lambdas: np.ndarray,
                       k: int = 8, threshold_sec: float = 5.0) -> dict:
    """Evaluate keyframe retrieval quality using predicted λ values.

    Uses the same frame->score-index mapping as oracle.py, so the labels the
    head is trained on and the recall it is scored by live in one space.
    pred_lambdas are λ values in [0, 1], not bin indices.
    """
    recalls, precisions, f1s = [], [], []
    for i, s in enumerate(samples):
        lam = float(pred_lambdas[i])
        fused = (1 - lam) * normalize_minmax(s["s_obj"]) + \
                lam * normalize_minmax(s["s_text"])
        top_k = np.argsort(fused)[-k:]

        gt_idx = gt_to_score_index(s)
        th = tolerance_steps(s, threshold_sec)

        r = recall_at_k(top_k, gt_idx, th)
        p = precision_at_k(top_k, gt_idx, th)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f1_score(p, r))

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "f1": np.mean(f1s),
    }


# ------------------------------------------------------------------
# CL metrics
# ------------------------------------------------------------------

class CLMetricTracker:
    """Track per-task accuracy after each training round."""

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # acc_matrix[t][j] = accuracy on task j after training task t
        self.acc_matrix = np.zeros((n_tasks, n_tasks))

    def record(self, trained_up_to: int, task_id: int, accuracy: float):
        self.acc_matrix[trained_up_to, task_id] = accuracy

    def acc(self) -> float:
        """Average accuracy after all tasks."""
        return self.acc_matrix[-1].mean()

    def fm(self) -> float:
        """Forgetting measure: FM = mean_j max_{i<N} (R_ij - R_Nj).

        The max deliberately excludes the final round. Including it would make
        the max at least R_Nj and so clamp FM at >= 0, hiding the case where a
        task ends up better than at any earlier point.
        """
        fgt = []
        for j in range(self.n_tasks - 1):
            best = max(self.acc_matrix[t, j]
                       for t in range(j, self.n_tasks - 1))
            fgt.append(best - self.acc_matrix[-1, j])
        return np.mean(fgt) if fgt else 0.0

    def bwt(self) -> float:
        """Backward transfer."""
        vals = []
        for j in range(self.n_tasks - 1):
            vals.append(self.acc_matrix[-1, j] - self.acc_matrix[j, j])
        return np.mean(vals) if vals else 0.0

    def fwt(self, baseline_acc: np.ndarray) -> float:
        """Forward transfer (requires baseline per-task accuracy)."""
        vals = []
        for j in range(1, self.n_tasks):
            vals.append(self.acc_matrix[j - 1, j] - baseline_acc[j])
        return np.mean(vals) if vals else 0.0

    def summary(self, baseline_acc: np.ndarray = None) -> dict:
        d = {
            "ACC": self.acc(),
            "FM": self.fm(),
            "BWT": self.bwt(),
        }
        if baseline_acc is not None:
            d["FWT"] = self.fwt(baseline_acc)
        d["matrix"] = self.acc_matrix.tolist()
        return d
