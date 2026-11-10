#!/usr/bin/env python3
"""
Stage 2a: Oracle λ* computation.

For each cached query, sweep λ ∈ {0.0, 0.1, ..., 1.0} and pick the one
that maximises recall@k against ground-truth keyframes.  The winning bin
index becomes the classification label for the fusion head.

Usage:
    python cl/oracle.py \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache/LongVideoBench \
        --k 8 --threshold 5.0 --fps_default 30.0
"""
import argparse
import glob
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LAMBDA_GRID = np.linspace(0.0, 1.0, 11)


GT_FPS = 24.0  # frame rate of LongVideoBench `position` annotations


def score_fps(data: dict) -> float:
    """Frames-per-second of the SCORE curve (not of the video).

    VSI's search returns one score per sampled second, so len(s_obj) tracks
    duration, while n_frames is the raw video frame count at native fps.
    """
    dur = float(data.get("duration", 0.0) or 0.0)
    n = len(np.asarray(data["s_obj"]))
    return (n / dur) if dur > 0 else 1.0


def gt_to_score_index(data: dict) -> np.ndarray:
    """Map ground-truth keyframes from RAW VIDEO FRAME space into score-curve
    index space, dropping any that fall outside the curve.

    KFSBench stores gt_frames as frame numbers. Our fused score array is
    indexed by sampled second, so comparing the two directly silently yields
    recall 0 for nearly every query.

    LongVideoBench `position` counts frames at a fixed GT_FPS=24, NOT at the
    video's native fps (median 30 here). Checked against the answer subtitle's
    timestamp on 39 subtitle questions with non-24 fps videos: 38/39 land
    within 5 s at 24 fps vs 14/39 at native fps.
    """
    gt = np.atleast_1d(np.asarray(data["gt_frames"])).astype(np.float64)
    T = len(np.asarray(data["s_obj"]))
    gt = gt / GT_FPS * score_fps(data)
    return gt[(gt >= 0) & (gt < T)]


def tolerance_steps(data: dict, threshold_sec: float) -> float:
    """The eval_json.py 5-second tolerance, expressed in score-curve steps."""
    return threshold_sec * score_fps(data)


def normalize_minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        return (scores - lo) / (hi - lo)
    return np.full_like(scores, 0.5)


def fuse(s_obj: np.ndarray, s_text: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * normalize_minmax(s_obj) + lam * normalize_minmax(s_text)


def recall_at_k(fused: np.ndarray, gt_frames: np.ndarray, k: int,
                threshold_frames: float) -> float:
    if len(gt_frames) == 0:
        return 0.0
    top_k = np.argsort(fused)[-k:]
    dists = np.abs(gt_frames[:, None] - top_k[None, :])
    covered = np.any(dists <= threshold_frames, axis=1).sum()
    return covered / len(gt_frames)


def sweep_lambda(s_obj, s_text, gt_frames, k, threshold_frames):
    best_lam_idx, best_recall = 0, -1.0
    recalls = []
    for i, lam in enumerate(LAMBDA_GRID):
        f = fuse(s_obj, s_text, lam)
        r = recall_at_k(f, gt_frames, k, threshold_frames)
        recalls.append(r)
        if r > best_recall or (r == best_recall and
                                abs(lam - 0.5) < abs(LAMBDA_GRID[best_lam_idx] - 0.5)):
            best_recall = r
            best_lam_idx = i
    return best_lam_idx, best_recall, np.array(recalls, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="seconds; will be multiplied by fps")
    ap.add_argument("--fps_default", type=float, default=30.0)
    args = ap.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    log.info(f"Found {len(npz_files)} cached queries")

    updated = 0
    for path in npz_files:
        data = dict(np.load(path, allow_pickle=True))

        gt_idx = gt_to_score_index(data)
        threshold_frames = tolerance_steps(data, args.threshold)

        best_idx, best_r, curve = sweep_lambda(
            data["s_obj"], data["s_text"], gt_idx,
            args.k, threshold_frames
        )

        data["oracle_lambda_idx"] = np.int64(best_idx)
        data["oracle_recall"] = np.float32(best_r)
        data["recall_curve"] = curve

        np.savez_compressed(path, **data)
        updated += 1

        if updated % 200 == 0:
            log.info(f"Updated {updated}/{len(npz_files)}")

    log.info(f"Done. Updated {updated} files. "
             f"Mean oracle recall: will be computed in dataset.py stats")


if __name__ == "__main__":
    main()
