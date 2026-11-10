#!/usr/bin/env python3
"""Figure 1 of the paper: one real LongVideoBench query where λ decides
whether the 8 selected frames hit the ground-truth moment."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from cl.dataset import CLDataset
from cl.oracle import gt_to_score_index, normalize_minmax, score_fps

QID = "cc0T2vtuJtc"
ds = CLDataset("/l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench",
               "configs/tasks_lvb4.yaml")
s = next(x for x in ds.samples if x["qid"] == QID)
o = normalize_minmax(np.asarray(s["s_obj"], float))
t = normalize_minmax(np.asarray(s["s_text"], float))
sec = np.arange(len(o)) / score_fps(s)
gt = gt_to_score_index(s) / score_fps(s)

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.5), sharex=True,
                        gridspec_kw={"height_ratios": [2, 1.15], "hspace": 0.12})
ax = axs[0]
for g in gt:
    ax.axvspan(g - 5, g + 5, color="#2f855a", alpha=0.18, lw=0)
ax.plot(sec, o, color="#c2571a", lw=1.0, label="visual score")
ax.plot(sec, t, color="#2563a8", lw=1.0, label="subtitle score")
ax.set_ylabel("score")
ax.set_yticks([0, 1])
ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0, 1.25))
ax = axs[1]
for g in gt:
    ax.axvspan(g - 5, g + 5, color="#2f855a", alpha=0.18, lw=0)
for row, lam, col in [(1, 0.3, "#6b7280"), (0, 0.8, "#1e3a8a")]:
    top = np.argsort((1 - lam) * o + lam * t)[-8:]
    hit = np.any(np.abs(sec[top][:, None] - gt[None, :]) <= 5, axis=1)
    ax.scatter(sec[top], np.full(8, row), marker="|", s=70, lw=1.5, color=col)
    rec = np.mean(np.any(np.abs(gt[:, None] - sec[top][None, :]) <= 5, axis=1))
    ax.text(sec[-1] + 4, row, f"recall\n{rec:.0%}", va="center", fontsize=6.5, color=col)
ax.set_yticks([0, 1], ["λ = 0.8", "λ = 0.3\n(default)"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("time in video (s)")
axs[0].text(gt[0] + 7, 0.93, "answer\nmoment", color="#2f855a", fontsize=6.5, va="top")
fig.savefig(ROOT / "report/figs/f01_teaser.pdf", bbox_inches="tight")
fig.savefig(ROOT / "report/figs/f01_teaser.png", bbox_inches="tight", dpi=200)
print(sec[-1], gt)
