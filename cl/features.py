"""
Stage 2b: 778-dim feature extraction from cached scores.

Feature vector per query:
    q_emb (768)  +  curve_stats (10)  =  778

curve_stats = [
    s_obj_mean, s_obj_std, s_obj_max, s_obj_entropy,   # visual branch
    s_text_mean, s_text_std, s_text_max, s_text_entropy, # text branch
    s_text_coverage,   # fraction of frames with s_text > 0.2
    duration_log,      # log(1 + duration)
]
"""
import numpy as np


def entropy(scores: np.ndarray) -> float:
    p = scores - scores.min()
    s = p.sum()
    if s < 1e-12:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def curve_stats(s_obj: np.ndarray, s_text: np.ndarray,
                duration: float) -> np.ndarray:
    return np.array([
        s_obj.mean(), s_obj.std(), s_obj.max(), entropy(s_obj),
        s_text.mean(), s_text.std(), s_text.max(), entropy(s_text),
        (s_text > 0.2).mean(),
        np.log1p(duration),
    ], dtype=np.float32)


FEATURE_SETS = ("base778", "stats10_pq44")


def extract_features(npz: dict, feature_set: str = "base778") -> np.ndarray:
    """base778 = q_emb + curve_stats (original spec).
    stats10_pq44 = curve_stats + per-query branch-interaction features
    (cl/perquery_features.py); drops q_emb, which overfits on ~530 train
    queries (results_diag/perquery.json)."""
    cs = curve_stats(npz["s_obj"], npz["s_text"],
                     float(npz.get("duration", 0)))
    if feature_set == "base778":
        return np.concatenate([npz["q_emb"].astype(np.float32), cs])
    if feature_set == "stats10_pq44":
        from .perquery_features import perquery_features
        return np.concatenate([cs, perquery_features(npz)])
    raise ValueError(f"unknown feature_set: {feature_set}")
