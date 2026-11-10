"""
Per-query features for λ prediction, computed from the cached branch curves
only (no GT). The 10 curve_stats in features.py describe each branch in
isolation; what decides whether λ matters for a query is how the two branches
interact and how the fused top-8 moves as λ changes.

Groups (44 dims):
    branch (2 × 8)  per branch: frac_zero, is_const, mean, std, top-8 mass
                    share, top-8 temporal spread, log #peaks, 8th–9th margin
    cross (4)       Pearson, Spearman, top-8 / top-32 overlap within ±5 s
    sens (11 + 1)   overlap of top-8 at each grid λ with top-8 at λ=0.3;
                    #distinct top-8 sets across the grid / 11
    consensus (11 + 1)  for each grid λ, how many other λ's top-8 agree with
                    its top-8 frames (±5 s); and the argmax-consensus λ
"""
import numpy as np
from scipy.stats import spearmanr

from .oracle import LAMBDA_GRID, normalize_minmax, tolerance_steps

K = 8
N_PQ = 44


def _overlap(a, b, tol):
    """Fraction of frames in a that lie within tol of some frame in b."""
    return float(np.mean(np.any(np.abs(a[:, None] - b[None, :]) <= tol, 1)))


def _branch(raw, x, T):
    top = np.argsort(x)[-K:]
    srt = np.sort(x)
    peaks = np.sum((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0.5)) \
        if T > 2 else 0
    return [
        float(np.mean(raw <= 1e-8)),
        float(raw.max() - raw.min() < 1e-8),
        float(x.mean()), float(x.std()),
        float(x[top].sum() / (x.sum() + 1e-8)),
        float(np.std(top) / max(T, 1)),
        float(np.log1p(peaks)),
        float(srt[-K] - srt[-K - 1]) if T > K else 0.0,
    ]


def perquery_features(s):
    raw_o = np.asarray(s["s_obj"], dtype=np.float64)
    raw_t = np.asarray(s["s_text"], dtype=np.float64)
    o, t = normalize_minmax(raw_o), normalize_minmax(raw_t)
    T = len(o)
    tol = tolerance_steps(s, 5.0)

    f = _branch(raw_o, o, T) + _branch(raw_t, t, T)

    if o.std() > 0 and t.std() > 0:
        pear = float(np.corrcoef(o, t)[0, 1])
        spear = float(spearmanr(o, t).correlation)
    else:
        pear = spear = 0.0
    top_o, top_t = np.argsort(o)[-K:], np.argsort(t)[-K:]
    f += [pear, spear, _overlap(top_o, top_t, tol),
          _overlap(np.argsort(o)[-32:], np.argsort(t)[-32:], tol)]

    tops = [np.argsort((1 - lam) * o + lam * t)[-K:] for lam in LAMBDA_GRID]
    ref = tops[3]
    f += [_overlap(tp, ref, tol) for tp in tops]
    f += [len({frozenset(tp.tolist()) for tp in tops}) / len(tops)]

    cons = [np.mean([_overlap(tops[i], tops[j], tol)
                     for j in range(len(tops)) if j != i])
            for i in range(len(tops))]
    f += cons
    f += [float(LAMBDA_GRID[int(np.argmax(cons))])]

    out = np.nan_to_num(np.asarray(f, dtype=np.float32))
    assert out.shape[0] == N_PQ, out.shape
    return out
