#!/usr/bin/env python3
"""
Figures and tables for the exploration report (report/index.html).

Everything is recomputed from files already in the repo / cache:
results{,_task,_v2,_v3,_v4}/, results_diag/, results_qa/, and the two score
caches (cache/ = original GT mapping, cache_fps24/ = corrected 24 fps GT).

    python report/make_figures.py      # writes report/figs/*.png|pdf + numbers.json
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cl.dataset import CLDataset
from cl.oracle import LAMBDA_GRID

OUT = ROOT / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_OLD = os.environ.get("CACHE_OLD", "/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench")
CACHE_NEW = os.environ.get("CACHE_NEW", "/l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench")
CFG = "configs/tasks_lvb4.yaml"

TASKS = ["T1_text_perception", "T2_visual_perception",
         "T3_text_relation", "T4_visual_relation"]
TSHORT = {"T1_text_perception": "T1 Text-Perc.",
          "T2_visual_perception": "T2 Visual-Perc.",
          "T3_text_relation": "T3 Text-Rel.",
          "T4_visual_relation": "T4 Visual-Rel."}
TCOL = {"T1_text_perception": "#2563a8", "T3_text_relation": "#7aa6d6",
        "T2_visual_perception": "#c2571a", "T4_visual_relation": "#e8a26d"}
STRATS = ["naive", "ewc", "lwf", "er", "joint", "knn"]
SLABEL = {"naive": "Naive", "ewc": "EWC", "lwf": "LwF", "er": "ER",
          "joint": "Joint", "knn": "kNN", "fixed": "Fixed λ=0.3",
          "oracle": "Oracle λ"}
INK, GRID, ACC, MUTED = "#1f2937", "#e5e7eb", "#2563a8", "#9ca3af"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": "#6b7280", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "figure.dpi": 100,
})
NUM = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def load_grid(d):
    R = defaultdict(dict)
    for f in glob.glob(f"{d}/*.json"):
        j = json.load(open(f))
        R[j["strategy"]][(j["order"], j["seed"])] = j
    return R


def acc_fm(R, st):
    ks = sorted(R[st])
    a = np.array([R[st][k]["cl_metrics"]["ACC"] for k in ks])
    f = np.array([R[st][k]["cl_metrics"]["FM"] for k in ks])
    return ks, a, f


def const_baselines(cache):
    """Per split seed, under the CL-ACC protocol (test recall averaged over
    tasks): λ=0.3, the single constant tuned on the train splits, and one
    constant per task tuned on that task's train split (needs task labels)."""
    out = {}
    for seed in range(3):
        ds = CLDataset(cache, CFG, split_seed=seed)
        tr = {t: np.stack([s["recall_curve"] for s in ds.get_task_splits(t)[0]]) for t in ds.task_names}
        te = {t: np.stack([s["recall_curve"] for s in ds.get_task_splits(t)[1]]) for t in ds.task_names}
        curve = np.mean([te[t].mean(0) for t in te], 0)
        g = int(np.mean([tr[t].mean(0) for t in tr], 0).argmax())
        out[seed] = {"fixed0.3": float(curve[3]), "global_lambda": float(LAMBDA_GRID[g]),
                     "global": float(curve[g]),
                     "task": float(np.mean([te[t].mean(0)[tr[t].mean(0).argmax()] for t in te])),
                     "task_lambdas": {t: float(LAMBDA_GRID[tr[t].mean(0).argmax()]) for t in te}}
    return out


def best_set_center(r):
    return float(LAMBDA_GRID[r == r.max()].mean())


# ---------------------------------------------------------------- caches
ds_new = CLDataset(CACHE_NEW, CFG)
ds_old = CLDataset(CACHE_OLD, CFG)
RN = {s["qid"]: np.asarray(s["recall_curve"]) for s in ds_new.samples}
RO = {s["qid"]: np.asarray(s["recall_curve"]) for s in ds_old.samples}
TASK = {s["qid"]: s["task"] for s in ds_new.samples}

# ---------------------------------------------------------------- F2 headroom
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6),
                       gridspec_kw={"width_ratios": [2.2, 1]})
num = {}
for t in TASKS:
    R = np.stack([RN[q] for q in RN if TASK[q] == t])
    ax[0].plot(LAMBDA_GRID, R.mean(0), "-o", ms=3.5, color=TCOL[t],
               label=f"{TSHORT[t]} (n={len(R)})")
    num[t] = {"curve": R.mean(0).round(4).tolist(),
              "best_const": float(LAMBDA_GRID[R.mean(0).argmax()]),
              "flat": float((np.ptp(R, 1) == 0).mean()), "n": len(R)}
Rall = np.stack(list(RN.values()))
ax[0].plot(LAMBDA_GRID, Rall.mean(0), "-", lw=2.4, color=INK, label="All (n=753)")
ax[0].axvline(0.3, color=MUTED, ls="--", lw=1)
ax[0].text(0.31, 0.89, "VSI default λ=0.3",
           color="#6b7280", fontsize=8.5)
ax[0].set_xlabel("fusion weight λ  (0 = visual only, 1 = subtitle only)")
ax[0].set_ylabel("keyframe recall@8")
ax[0].set_title("(a) Recall vs. constant λ, per task")
ax[0].legend(fontsize=8, loc="lower center", ncol=2)
fl = [num[t]["flat"] for t in TASKS]
ax[1].bar(range(4), fl, color=[TCOL[t] for t in TASKS])
for i, v in enumerate(fl):
    ax[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8.5)
ax[1].set_xticks(range(4), [TSHORT[t].split()[0] for t in TASKS])
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("share of queries")
ax[1].set_title("(b) λ makes no difference")
save(fig, "f02_headroom")
NUM["headroom"] = {"per_task": num,
                   "all_curve": Rall.mean(0).round(4).tolist(),
                   "oracle": float(Rall.max(1).mean()),
                   "flat": float((np.ptp(Rall, 1) == 0).mean())}

# ---------------------------------------------------------------- F3 premise
from scipy.stats import mannwhitneyu
lt, lv = [], []
for q, r in RN.items():
    if np.ptp(r) == 0:
        continue
    (lt if "text" in TASK[q] else lv).append(best_set_center(r))
p = mannwhitneyu(lt, lv, alternative="greater").pvalue
fig, ax = plt.subplots(figsize=(6.2, 3.3))
bins = np.linspace(-0.05, 1.05, 12)
ax.hist(lt, bins=bins, alpha=0.75, color=TCOL["T1_text_perception"],
        label=f"text tasks T1+T3 (n={len(lt)}, mean {np.mean(lt):.2f})")
ax.hist(lv, bins=bins, alpha=0.6, color=TCOL["T2_visual_perception"],
        label=f"visual tasks T2+T4 (n={len(lv)}, mean {np.mean(lv):.2f})")
ax.set_xlabel("best λ for the query (centre of the optimal set)")
ax.set_ylabel("# queries")
ax.set_title(f"Per-query best λ, text vs. visual  (Mann-Whitney p = {p:.1e})")
ax.legend(fontsize=8.5)
save(fig, "f03_premise")
NUM["premise"] = {"text_mean": float(np.mean(lt)), "vis_mean": float(np.mean(lv)),
                  "n_text": len(lt), "n_vis": len(lv), "p": float(p)}
# the same test on the old GT, for the narrative
lt0 = [best_set_center(r) for q, r in RO.items() if np.ptp(r) > 0 and "text" in TASK[q]]
lv0 = [best_set_center(r) for q, r in RO.items() if np.ptp(r) > 0 and "text" not in TASK[q]]
NUM["premise_oldGT"] = {"text_mean": float(np.mean(lt0)), "vis_mean": float(np.mean(lv0)),
                        "p": float(mannwhitneyu(lt0, lv0, alternative="greater").pvalue)}

# ---------------------------------------------------------------- grids
G = {v: load_grid(d) for v, d in [("v1", "results"), ("task", "results_task"),
                                   ("v2", "results_v2"), ("v3", "results_v3"),
                                   ("v4", "results_v4")]}
NUM["grids"] = {}
for v, R in G.items():
    NUM["grids"][v] = {}
    for st in STRATS + ["fixed", "oracle"]:
        ks, a, f = acc_fm(R, st)
        _, fx, _ = acc_fm(R, "fixed")
        _, _, fn = acc_fm(R, "naive")
        NUM["grids"][v][st] = {"acc": float(a.mean()), "acc_sd": float(a.std()),
                               "fm": float(f.mean()), "fm_sd": float(f.std()),
                               "gain_vs_fixed": float((a - fx).mean()),
                               "wins_vs_fixed": int((a > fx).sum()),
                               "fm_below_naive": int((f < fn).sum()), "n": len(a)}

CB = {"old": const_baselines(CACHE_OLD), "new": const_baselines(CACHE_NEW)}
NUM["const"] = CB
for v, R in G.items():
    cb = CB["new" if v == "v4" else "old"]
    for st in STRATS:
        ks, a, _ = acc_fm(R, st)
        base = np.array([cb[k[1]]["global"] for k in ks])
        NUM["grids"][v][st]["gain_vs_tuned_const"] = float((a - base).mean())
        NUM["grids"][v][st]["wins_vs_tuned_const"] = int((a > base).sum())
    NUM["grids"][v]["tuned_const"] = float(np.mean([cb[k]["global"] for k in cb]))
    NUM["grids"][v]["task_const"] = float(np.mean([cb[k]["task"] for k in cb]))

# ---------------------------------------------------------------- F5 decode fix
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(STRATS[:5]))
for i, (v, lab, c) in enumerate([("v1", "argmax decode (Sept.)", MUTED),
                                  ("v2", "expected decode (fix)", ACC)]):
    m = [NUM["grids"][v][s]["acc"] for s in STRATS[:5]]
    e = [NUM["grids"][v][s]["acc_sd"] for s in STRATS[:5]]
    ax[0].bar(x + (i - 0.5) * 0.38, m, 0.36, yerr=e, capsize=2, color=c, label=lab)
ax[0].axhline(NUM["grids"]["v2"]["fixed"]["acc"], color=INK, ls="--", lw=1)
ax[0].text(4.45, NUM["grids"]["v2"]["fixed"]["acc"] + 0.002, "fixed λ=0.3",
           ha="right", fontsize=8.5)
ax[0].set_xticks(x, [SLABEL[s] for s in STRATS[:5]])
ax[0].set_ylim(0.48, 0.6)
ax[0].set_ylabel("CL ACC (recall@8)")
ax[0].set_title("(a) Learned λ before / after the decode fix")
ax[0].legend(fontsize=8.5, loc="upper left")
d3 = json.load(open("results_diag/diagnose.json"))["seeds"]
ha = np.mean([d3[s]["D3_joint_head"]["std_778_mlp_tau0.1"]["test_pred_lambda_hist"] for s in d3], 0)
pq = json.load(open("results_diag/perquery.json"))
he = np.mean([pq[s]["head_base778"]["pred_lambda_hist"] for s in pq], 0)
ax[1].bar(LAMBDA_GRID - 0.022, ha / ha.sum(), 0.042, color=MUTED, label="argmax of bins")
ax[1].bar(LAMBDA_GRID + 0.022, he / he.sum(), 0.042, color=ACC, label="softmax-weighted mean")
ax[1].set_xlabel("predicted λ on test queries")
ax[1].set_ylabel("share")
ax[1].set_title("(b) What the same head outputs")
ax[1].legend(fontsize=8.5)
save(fig, "f05_decode")

# ---------------------------------------------------------------- F6 diagnosis curve (old GT)
c = np.mean([d3[s]["D1_const_curve_test"] for s in d3], 0)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.plot(LAMBDA_GRID, c, "-o", color=INK, ms=4, label="constant λ (test)")
for name, val, col, ls in [
        ("best per-task constant", np.mean([d3[s]["D1_task_const_test_recall"] for s in d3]), "#2f855a", "--"),
        ("per-query oracle", np.mean([d3[s]["D1_query_oracle_test_recall"] for s in d3]), "#9b2c2c", ":"),
        ("learned head, argmax (Sept.)", np.mean([d3[s]["D3_joint_head"]["std_778_mlp_tau0.1"]["test_recall_argmax"] for s in d3]), MUTED, "-."),
        ("same head, expected decode", np.mean([d3[s]["D3_joint_head"]["raw_778_mlp_tau0.1"]["test_recall_expect"] for s in d3]), ACC, "-.")]:
    ax.axhline(val, color=col, ls=ls, lw=1.3, label=f"{name}: {val:.3f}")
ax.set_xlabel("λ")
ax.set_ylabel("recall@8 (test, mean of 3 splits)")
ax.set_title("Diagnosis (original GT mapping): where the headroom is")
ax.legend(fontsize=7.8, loc="lower center")
ax.set_ylim(0.5, 0.69)
save(fig, "f06_diagnosis")
NUM["diag"] = {"const_curve": c.round(4).tolist(),
               "flat_test": float(np.mean([d3[s]["D1_frac_flat_curves_test"] for s in d3])),
               "surrogate_per_query": float(np.mean([d3[s]["D2_surrogate"]["0.1"]["per_query_surrogate_opt_recall"] for s in d3]))}

# ---------------------------------------------------------------- F7 per-query features
items = [("fixed0.3", "Fixed λ=0.3"), ("consensus_heuristic", "Consensus rule (no training)"),
         ("task_const", "Best constant per task"),
         ("gbm_stats10+pq44", "Gradient boosting, stats10+pq44"),
         ("head_base778", "MLP, base778 (question emb. + stats)"),
         ("head_base778+pq44", "MLP, base778 + pq44"),
         ("head_stats10", "MLP, stats10"), ("head_pq44", "MLP, pq44"),
         ("head_stats10+pq44", "MLP, stats10 + pq44 (chosen)"), ("oracle", "Per-query oracle")]
vals, sds = [], []
for k, _ in items:
    xs = [pq[s][k]["test"] if isinstance(pq[s][k], dict) else pq[s][k] for s in pq]
    vals.append(np.mean(xs)); sds.append(np.std(xs))
fig, ax = plt.subplots(figsize=(7.2, 4.0))
cols = [MUTED if "head" not in k and "gbm" not in k else ACC for k, _ in items]
cols[-2] = "#1e3a8a"; cols[-1] = "#9b2c2c"
y = np.arange(len(items))
ax.barh(y, vals, xerr=sds, color=cols, capsize=2)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_yticks(y, [l for _, l in items])
ax.set_xlim(0.5, 0.7)
ax.set_xlabel("test recall@8 (joint training, mean ± sd of 3 splits)")
ax.set_title("Which inputs let a head predict λ? (original GT mapping)")
ax.axvline(vals[0], color=INK, ls="--", lw=0.9)
save(fig, "f07_features")
NUM["features"] = {k: float(v) for (k, _), v in zip(items, vals)}

# ---------------------------------------------------------------- F8 GT fix
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].bar([0, 1], [38 / 39, 14 / 39], color=[ACC, MUTED], width=0.55)
for i, (a, b) in enumerate([(38, 39), (14, 39)]):
    ax[0].text(i, a / b + 0.02, f"{a}/{b}", ha="center")
ax[0].set_xticks([0, 1], ["position ÷ 24 fps", "position ÷ native fps"])
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel("GT within 5 s of answer subtitle")
ax[0].set_title("(a) Which frame rate the GT uses")
ax[1].plot(LAMBDA_GRID, np.stack(list(RO.values())).mean(0), "-o", ms=3.5, color=MUTED, label="original mapping (native fps)")
ax[1].plot(LAMBDA_GRID, Rall.mean(0), "-o", ms=3.5, color=ACC, label="corrected (24 fps)")
ax[1].set_xlabel("λ")
ax[1].set_ylabel("recall@8, all 753 queries")
ax[1].set_title("(b) Constant-λ curve before / after the fix")
ax[1].legend(fontsize=8.5)
save(fig, "f08_gtfix")
NUM["gtfix_old_curve"] = np.stack(list(RO.values())).mean(0).round(4).tolist()
NUM["gtfix_old_oracle"] = float(np.stack(list(RO.values())).max(1).mean())
NUM["gtfix_old_flat"] = float(np.mean([np.ptp(r) == 0 for r in RO.values()]))

# ---------------------------------------------------------------- F9 v4 main
g4 = NUM["grids"]["v4"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.arange(len(STRATS))
ax[0].bar(x, [g4[s]["acc"] for s in STRATS], yerr=[g4[s]["acc_sd"] for s in STRATS],
          capsize=2, color=[MUTED] + [ACC] * 4 + ["#6b8fb8"])
for lab, val, col, ls in [("fixed λ=0.3", g4["fixed"]["acc"], INK, "--"),
                          ("one λ tuned on train", g4["tuned_const"], "#2f855a", "-."),
                          ("one λ per task (needs task label)", g4["task_const"], "#b7791f", "-."),
                          ("oracle λ", g4["oracle"]["acc"], "#9b2c2c", ":")]:
    ax[0].axhline(val, color=col, ls=ls, lw=1.2, label=f"{lab}: {val:.3f}")
ax[0].legend(fontsize=7.8, loc="upper left", ncol=2)
for i, s in enumerate(STRATS):
    ax[0].text(i, 0.588, f"{g4[s]['acc']:.3f}", ha="center", fontsize=8, color="white")
ax[0].set_xticks(x, [SLABEL[s] for s in STRATS])
ax[0].set_ylim(0.58, 0.81)
ax[0].set_ylabel("ACC (final recall@8, avg over tasks)")
ax[0].set_title("(a) Final accuracy, 3 orders × 3 seeds")
ax[1].bar(x, [g4[s]["fm"] for s in STRATS], yerr=[g4[s]["fm_sd"] for s in STRATS],
          capsize=2, color=[MUTED] + [ACC] * 4 + ["#6b8fb8"])
for i, s in enumerate(STRATS):
    ax[1].text(i, max(g4[s]["fm"] + g4[s]["fm_sd"], 0) + 0.002, f"{g4[s]['fm']:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x, [SLABEL[s] for s in STRATS])
ax[1].set_ylabel("FM (forgetting, lower is better)")
ax[1].set_title("(b) Forgetting")
save(fig, "f09_v4_main")

# ---------------------------------------------------------------- F10 paired gains
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.3), sharey=False)
R4 = G["v4"]
for ax, ref, title in [(axs[0], "fixed", "(a) vs. VSI default λ=0.3"),
                       (axs[1], "global", "(b) vs. one λ tuned on the train split")]:
    for i, s_ in enumerate(STRATS):
        ks, a, _ = acc_fm(R4, s_)
        base = acc_fm(R4, "fixed")[1] if ref == "fixed" else np.array([CB["new"][k[1]]["global"] for k in ks])
        d = (a - base) * 100
        jit = np.linspace(-0.15, 0.15, len(d))
        ax.scatter(i + jit, d, s=16, color=ACC if s_ not in ("naive", "knn") else MUTED, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [d.mean()] * 2, color=INK, lw=1.6)
        ax.text(i, d.max() + 0.35, f"{(d > 0).sum()}/9", ha="center", fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(STRATS)), [SLABEL[s_] for s_ in STRATS])
    ax.set_title(title)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
axs[0].set_ylabel("ACC gain (points); label = wins")
save(fig, "f10_paired")

# ---------------------------------------------------------------- F11 accuracy matrices
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, s in zip(axs, ["naive", "lwf", "joint"]):
    M = np.mean([np.array(R4[s][("default", sd)]["cl_metrics"]["matrix"]) for sd in range(3)], 0)
    im = ax.imshow(M, cmap="Blues", vmin=0.4, vmax=1.0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.75 else INK)
    ax.set_xticks(range(4), [t.split("_")[0] for t in TASKS])
    ax.set_yticks(range(4), [f"after {t.split('_')[0]}" for t in TASKS])
    ax.set_xlabel("evaluated on")
    ax.set_title(SLABEL[s])
    ax.grid(False)
fig.suptitle("Recall@8 matrix, default order (T1→T2→T3→T4), mean of 3 seeds", y=1.02)
save(fig, "f11_matrix")

# ---------------------------------------------------------------- F12 predicted λ per round
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
best = {t: NUM["headroom"]["per_task"][t]["best_const"] for t in TASKS}
for ax, s in zip(axs, ["naive", "lwf"]):
    for t in TASKS:
        y = np.mean([[r[t]["pred_lambda_mean"] for r in R4[s][("default", sd)]["per_round"]] for sd in range(3)], 0)
        ax.plot(range(1, 5), y, "-o", color=TCOL[t], ms=4, label=TSHORT[t])
        ax.scatter([4.25], [best[t]], marker="<", color=TCOL[t], s=40)
    ax.set_xticks(range(1, 5), ["after T1", "after T2", "after T3", "after T4"])
    ax.set_title(f"{SLABEL[s]}: mean predicted λ per task")
    ax.set_xlim(0.7, 4.45)
axs[0].set_ylabel("mean predicted λ")
axs[0].legend(fontsize=8, loc="lower left")
axs[1].text(4.22, 0.35, "◀ = best constant\nfor that task", fontsize=7.5, ha="right", color="#6b7280")
save(fig, "f12_lambda_rounds")
NUM["lambda_rounds"] = {s: {t: np.mean([[r[t]["pred_lambda_mean"] for r in R4[s][("default", sd)]["per_round"]] for sd in range(3)], 0).round(3).tolist() for t in TASKS} for s in ["naive", "lwf", "joint"]}

# ---------------------------------------------------------------- F13 progression
fig, ax = plt.subplots(figsize=(7.4, 3.4))
vers = [("v1", "Sept.: argmax\ndecode"), ("v2", "decode fix"), ("v3", "+ per-query\nfeatures"), ("v4", "+ GT 24 fps\nfix")]
for s, col in [("lwf", ACC), ("joint", "#1e3a8a"), ("naive", MUTED), ("ewc", "#6b8fb8")]:
    ax.plot(range(4), [NUM["grids"][v][s]["gain_vs_fixed"] * 100 for v, _ in vers], "-o", color=col, label=SLABEL[s])
ax.plot(range(4), [(NUM["grids"][v]["oracle"]["acc"] - NUM["grids"][v]["fixed"]["acc"]) * 100 for v, _ in vers], "--", color="#9b2c2c", label="Oracle (ceiling)")
ax.plot(range(4), [(NUM["grids"][v]["tuned_const"] - NUM["grids"][v]["fixed"]["acc"]) * 100 for v, _ in vers], "-.", color="#2f855a", label="one λ tuned on train")
ax.plot(range(4), [(NUM["grids"][v]["task_const"] - NUM["grids"][v]["fixed"]["acc"]) * 100 for v, _ in vers], "-.", color="#b7791f", label="one λ per task")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(4), [l for _, l in vers])
ax.set_ylabel("ACC − fixed λ (points)")
ax.set_title("Learned λ vs. fixed λ across the four rounds of fixes")
ax.legend(fontsize=8, ncol=2)
save(fig, "f13_progression")

# ---------------------------------------------------------------- QA
QA = json.load(open("results_qa/analysis.json"))
L = lambda n: {r["qid"]: r for r in map(json.loads, open(f"results_qa/{n.replace('/', '__')}.jsonl"))}
B = {m: L(m) for m in ["uniform", "fixed:0.0", "fixed:0.3", "fixed:0.7", "fixed:1.0", "oracle"]}
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1, 1.5]})
names = [("uniform", "Uniform 8"), ("fixed:0.0", "λ=0"), ("fixed:0.3", "λ=0.3"),
         ("fixed:0.7", "λ=0.7"), ("fixed:1.0", "λ=1"), ("oracle", "Oracle λ")]
v = [QA["all753"][k]["acc"] for k, _ in names]
ax[0].bar(range(6), v, color=[MUTED, "#c9a27a", ACC, "#6b8fb8", "#c9a27a", "#9b2c2c"])
for i, a in enumerate(v):
    ax[0].text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(range(6), [l for _, l in names], fontsize=8.5)
ax[0].set_ylim(0.55, 0.67)
ax[0].set_ylabel("QA accuracy (753 questions)")
ax[0].set_title("(a) Qwen3-VL-8B accuracy by frame choice")
w = 0.2
for i, (k, lab, col) in enumerate([("uniform", "Uniform", MUTED), ("fixed:0.3", "λ=0.3", ACC),
                                    ("fixed:0.7", "λ=0.7", "#6b8fb8"), ("oracle", "Oracle λ", "#9b2c2c")]):
    ax[1].bar(np.arange(4) + (i - 1.5) * w, [QA["all753"][k]["per_task"][t] for t in TASKS], w, color=col, label=lab)
ax[1].set_xticks(range(4), [TSHORT[t] for t in TASKS], fontsize=8.5)
ax[1].set_ylim(0.45, 0.85)
ax[1].set_ylabel("QA accuracy")
ax[1].set_title("(b) Per task")
ax[1].legend(fontsize=8, ncol=4, loc="upper right")
save(fig, "f14_qa")

sens = {q for q, r in RN.items() if np.ptp(r) > 0}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
y0 = [];  # QA by recall level at λ=0.3
lv = sorted({round(float(RN[q][3]), 2) for q in RN})
groups = [("recall = 0", lambda r: r == 0), ("0 < recall < 1", lambda r: 0 < r < 1), ("recall = 1", lambda r: r == 1)]
vals = []
for lab, fn in groups:
    qs = [q for q in B["fixed:0.3"] if fn(RN[q][3])]
    vals.append((np.mean([B["fixed:0.3"][q]["correct"] for q in qs]), len(qs)))
ax[0].bar(range(3), [a for a, _ in vals], color=[MUTED, "#6b8fb8", ACC])
for i, (a, n) in enumerate(vals):
    ax[0].text(i, a + 0.01, f"{a:.2f}\n(n={n})", ha="center", fontsize=8.5)
ax[0].set_xticks(range(3), [l for l, _ in groups])
ax[0].set_ylim(0, 0.95)
ax[0].set_ylabel("QA accuracy at λ=0.3")
ax[0].set_title("(a) Hitting a GT keyframe matters for QA")
sub = {}
for k in ["uniform", "fixed:0.3", "fixed:0.7", "oracle"]:
    sub[k] = (np.mean([B[k][q]["correct"] for q in sens]), np.mean([B[k][q]["correct"] for q in B[k] if q not in sens]))
hs, hf = [], []
for sd in range(3):
    H = L(f"results_v4/lwf_default_seed{sd}")
    for q, r in H.items():
        if q in sens:
            hs.append(r["correct"]); hf.append(B["fixed:0.3"][q]["correct"])
labs = ["Uniform", "λ=0.3", "λ=0.7", "Oracle λ"]
ax[1].bar(np.arange(4) - 0.2, [sub[k][0] for k in sub], 0.38, color=ACC, label=f"λ-sensitive ({len(sens)} q)")
ax[1].bar(np.arange(4) + 0.2, [sub[k][1] for k in sub], 0.38, color=MUTED, label=f"λ-insensitive ({753 - len(sens)} q)")
for i, k in enumerate(sub):
    ax[1].text(i - 0.2, sub[k][0] + 0.005, f"{sub[k][0]:.3f}", ha="center", fontsize=7.8)
ax[1].set_xticks(range(4), labs)
ax[1].set_ylim(0.5, 0.72)
ax[1].set_ylabel("QA accuracy")
ax[1].set_title("(b) Where λ can matter for QA")
ax[1].legend(fontsize=8, loc="upper left")
save(fig, "f15_qa_recall")
d = np.array(hs, float) - np.array(hf, float)
bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)]
NUM["qa"] = {"by_recall": [(l, float(a), int(n)) for (l, _), (a, n) in zip(groups, vals)],
             "subset": {k: [float(a), float(b)] for k, (a, b) in sub.items()},
             "n_sens": len(sens),
             "lwf_sens": [float(np.mean(hs)), float(np.mean(hf)), float(d.mean()),
                          float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d)],
             "analysis": QA}

json.dump(NUM, open(ROOT / "report" / "numbers.json", "w"), indent=1, default=float)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))

# ---------------------------------------------------------------- F16 headroom decomposition (v4)
g4 = NUM["grids"]["v4"]
steps = [("VSI default\nλ=0.3", g4["fixed"]["acc"]), ("one λ\ntuned on train", g4["tuned_const"]),
         ("one λ\nper task", g4["task_const"]), ("per-query\noracle λ", g4["oracle"]["acc"])]
fig, ax = plt.subplots(figsize=(6.8, 3.4))
prev = steps[0][1]
ax.bar(0, prev, color=MUTED, width=0.6)
ax.text(0, prev + 0.004, f"{prev:.3f}", ha="center", fontsize=8.5)
cols = ["#2f855a", "#b7791f", "#9b2c2c"]
for i, ((lab, v), c) in enumerate(zip(steps[1:], cols), start=1):
    ax.bar(i, v - prev, bottom=prev, color=c, width=0.6)
    ax.text(i, v + 0.004, f"{v:.3f}\n(+{(v - prev) * 100:.1f})", ha="center", fontsize=8.5)
    prev = v
for s_, mk in [("lwf", "D"), ("naive", "o")]:
    ax.axhline(g4[s_]["acc"], color=ACC if s_ == "lwf" else MUTED, ls="--", lw=1.1,
               label=f"CL head, {SLABEL[s_]}: {g4[s_]['acc']:.3f}")
ax.set_xticks(range(4), [l for l, _ in steps], fontsize=8.5)
ax.set_ylim(0.58, 0.8)
ax.set_ylabel("ACC (recall@8)")
ax.set_title("Where the headroom is: global shift → task → query")
ax.legend(fontsize=8, loc="upper left")
save(fig, "f16_decomp")
