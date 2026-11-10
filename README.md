# CV805: Should the Keyframe Fusion Weight Change with the Question?

Course project for **CV805** by Jianxiang He, Longtao Jiang and Makhmud Assentay.

[VSI](https://github.com/Jacksonha7/Visual-Subtitle-Integration) picks the 8
frames a vision-language model sees from a long video. It scores every second
with two clues and mixes them with one fixed weight λ:

```
S(t) = (1 − λ) · minmax(S_obj(t)) + λ · minmax(S_text(t))      # VSI default λ = 0.3
```

Here `S_obj` asks whether the question's objects are visible, and `S_text`
asks whether the subtitle matches the question. We study whether λ should
depend on the question, and whether a small predictor can learn it when
question types arrive one after another (continual learning).

The original VSI README is kept as [`README_VSI_upstream.md`](README_VSI_upstream.md).

## Findings (LongVideoBench val, 753 questions, 4 question types, 3 orders × 3 seeds)

| | recall@8 (±5 s) |
|---|---|
| Default λ = 0.3 | 0.617 |
| One λ tuned on the training split | 0.662 |
| One λ per question type | 0.677 |
| Continual learning: Naive / EWC / LwF / ER / kNN | 0.656 / 0.662 / **0.668** / 0.660 / 0.645 |
| Joint training (reference) | 0.668 |
| Best λ per question (oracle) | 0.765 |

**Q1. Do question types want different λ?** Yes. Text questions prefer about
0.62 and visual questions about 0.49 (p = 1.6e-6). Still, most of the room to
improve comes from differences between single questions, not between types.

**Q2. Can continual learning learn λ, and does it forget?** Every method beats
the default in 9 of 9 runs. Against one tuned λ, however, it only ties (+0.6 at
best). Forgetting is small for all methods.

**Q3. Why is the gain small?** The predictor gives every type about the same λ
(≈0.6): it learns a better global value. One reason is that 62% of questions
give the same recall for every λ, so the training signal is sparse. The other
is that the inputs do not include the question type.

**Q4. Do better frames give better answers?** We test with Qwen3-VL-8B, giving
it 8 frames plus subtitles. Overall, a learned λ answers the same as the
default (+0.4, p = 0.77). Frames still matter: accuracy is 0.49 when no answer
moment is hit and 0.75 when all are hit. On the 284 questions where λ changes
the frames, the best λ answers 5 points more.

Paper: [`report/paper/main.pdf`](report/paper/main.pdf). Talks:
[`report/slides/`](report/slides/).

## Repository layout

| Path | What it is |
|---|---|
| `VSI_keyframe_search.py`, `Keyframe_Matching.py`, `TStar/`, `KFSBench/`, `examples/` | Upstream VSI code, unmodified |
| `cl/cache_scores.py` | Runs both VSI branches once per question and caches the two score curves and ground-truth moments (`.npz`) |
| `cl/oracle.py` | Sweeps λ ∈ {0, 0.1, …, 1}, gives recall per λ and the per-question best λ. It also converts annotated frames to seconds (`GT_FPS = 24`) |
| `cl/dataset.py`, `configs/tasks_lvb4.yaml` | Maps LongVideoBench categories to the 4 tasks and makes the 70/30 splits |
| `cl/features.py`, `cl/perquery_features.py` | Predictor inputs: 10 curve statistics plus 44 per-question curve features |
| `cl/fusion_head.py` | Small 2-layer predictor that outputs a distribution over the 11 λ values. It decodes the expected λ |
| `cl/strategies/` | Naive, EWC, LwF, ER, Joint, kNN, and the fixed-λ baselines |
| `cl/train_cl.py`, `cl/metrics.py`, `cl/analyze.py` | Trains one run over the task stream, then computes ACC / FM / recall matrices |
| `cl/diagnose_lambda.py`, `cl/exp_perquery.py` | Diagnostics: λ flatness, and which inputs help |
| `cl/run_downstream_qa.py`, `cl/analyze_qa.py` | Q4 downstream QA with Qwen3-VL-8B, plus paired McNemar and bootstrap tests |
| `scripts/` | Environment setup, video download, Slurm jobs for each stage |
| `results_v4/` | **Final grid** (after the 24 fps annotation fix): one JSON and one checkpoint per run |
| `results_qa/` | Final QA answers and analysis |
| `results/`, `results_task/`, `results_v2/`, `results_v3/` | Earlier grids, kept for the "what changed" figures (JSON only) |
| `results_diag/` | λ diagnostics |
| `figures_v4/` | Per-grid plots from `cl/analyze.py` |
| `report/make_figures.py` | Builds every paper/talk figure (`report/figs/`) and `report/numbers.json` |
| `report/paper/` | CVPR-format paper (LaTeX + PDF) |
| `report/slides/` | MBZUAI Beamer talks: a 12-minute full talk and three 5-minute talks (one per member), each with a speaker script |
| `report/index.html`, `report/draft_zh.html` | Exploration report and Chinese draft |

## Reproduce

```bash
# environment (VSI stack; see README_VSI_upstream.md for the torch/mmcv notes)
bash scripts/setup_env.sh
pip install -r cl/requirements.txt

# 1. cache score curves on LongVideoBench val (GPU, ~12 h, resumable)
sbatch scripts/run_cache.sh
sbatch scripts/run_cache_resume_and_oracle.sh

# 2. final continual learning grid: 6 methods x 3 orders x 3 seeds (CPU, minutes)
CACHE=<cache_dir> sbatch scripts/run_fps24_slurm.sh      # -> results_v4/, figures_v4/

# 3. downstream QA with Qwen3-VL-8B (GPU)
PY=<python with transformers>=4.57> sbatch scripts/run_qa_slurm.sh   # -> results_qa/
python cl/analyze_qa.py --qa_dir results_qa --results results_v4

# 4. figures, paper, slides
CACHE_OLD=<cache_before_fix> CACHE_NEW=<cache_dir> python report/make_figures.py
cd report/paper  && latexmk -pdf main.tex
cd report/slides && latexmk -pdf talk_jianxiang.tex
```

Submit the Slurm scripts from the repository root, or set `PROJ=<repo>`. The
default data paths in the scripts point to our cluster (`/l/users/...`), so
override them for your own setup.

A single run looks like this:

```bash
python cl/train_cl.py --cache_dir <cache_dir> --strategy lwf --order default \
    --loss soft_topk --decode expected --features stats10_pq44 --standardize \
    --epochs 50 --seed 0 --out results_v4/lwf_default_seed0.json
```

## Pitfalls we hit

Each of these runs without error and gives believable but wrong numbers.

1. **Frames vs. seconds.** Annotations are frame numbers, but the score curves
   are indexed by second. `cl/oracle.gt_to_score_index` converts between them.
2. **24 fps annotations.** LongVideoBench counts annotated frames at a fixed
   24 fps, not the video's real frame rate. 38 of 39 subtitle-answer
   annotations line up at 24 fps, but only 14 at the real rate.
3. **Short videos dropped.** VSI's 4×4 detection grid fails on clips of 16 s
   or less. `cl/cache_scores.adaptive_grid` shrinks the grid instead.
4. **Train/test mismatch.** The predictor was trained on the expected λ but
   tested on the most likely λ. Learned weights then looked 3–5 points worse
   than the default.

Also note that VSI's fusion uses min–max normalisation, not Z-score.

## Licence

Upstream VSI code keeps its MIT licence ([`LICENSE`](LICENSE)). The Beamer
theme in `report/slides/` keeps its own licence (see `THEME_SOURCE.txt`).
