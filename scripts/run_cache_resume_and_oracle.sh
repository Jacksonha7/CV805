#!/bin/bash
#SBATCH --job-name=vsi-cache-resume
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# Fills any cache gaps left by the first pass (cache_scores.py skips existing
# .npz, so this is a safe re-run), then gates oracle.py behind a coverage check.
#
# The gate exists because oracle.py writes a lambda that every downstream CL
# experiment consumes; computing it on a partial cache poisons them silently.

set -uo pipefail

PYTHON=/home/jianxiang.he/.conda/envs/vsi/bin/python
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
CACHE_DIR=/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench
DATA_DIR=/l/users/jianxiang.he/continual-vsi/data/LongVideoBench

# Fraction of on-disk-answerable videos that must be cached before oracle runs.
COVERAGE_MIN=${COVERAGE_MIN:-0.98}

cd "$PROJ" || exit 1

echo "=== [$(date '+%F %T')] cache resume pass ==="
$PYTHON cl/cache_scores.py \
    --dataset LongVideoBench \
    --input_json "$DATA_DIR/lvb_val_grounded.json" \
    --video_root "$DATA_DIR/videos" \
    --subtitle_root "$DATA_DIR/subtitles" \
    --config_path third_party/YOLO-World/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
    --checkpoint_path /l/users/jianxiang.he/YOLO-World/weights/s_stage1-d1c1d7d8.pth \
    --cache_dir "$CACHE_DIR" \
    --device cuda:0 \
    --text_weight 0.3

echo "=== [$(date '+%F %T')] coverage check ==="
$PYTHON - "$DATA_DIR" "$CACHE_DIR" "$COVERAGE_MIN" <<'PY'
import json, os, sys
data_dir, cache_dir, cov_min = sys.argv[1], sys.argv[2], float(sys.argv[3])

items = json.load(open(f"{data_dir}/lvb_val_grounded.json"))
on_disk = set(os.listdir(f"{data_dir}/videos"))
cached = {f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npz")}

# Only videos actually present on disk can ever be cached; judge coverage
# against that set, not against the full JSON.
answerable = set()
for it in items:
    vid = it.get("video_id")
    vp = it.get("video_path", "")
    name = os.path.basename(vp) if vp else None
    if name and (name in on_disk or f"{name}.mp4" in on_disk):
        answerable.add(vid)

missing = answerable - cached
cov = len(answerable & cached) / len(answerable) if answerable else 0.0

print(f"answerable={len(answerable)} cached={len(answerable & cached)} "
      f"missing={len(missing)} coverage={cov:.4f} threshold={cov_min}")
if missing:
    print("missing sample:", sorted(missing)[:10])

sys.exit(0 if cov >= cov_min else 1)
PY

if [ $? -ne 0 ]; then
    echo "!!! coverage below threshold — oracle SKIPPED to avoid a poisoned lambda."
    echo "!!! inspect the log, then rerun this script to fill the remainder."
    exit 1
fi

echo "=== [$(date '+%F %T')] oracle ==="
$PYTHON cl/oracle.py --cache_dir "$CACHE_DIR" --k 8 --threshold 5.0
echo "=== [$(date '+%F %T')] done ==="
