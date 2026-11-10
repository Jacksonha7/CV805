#!/bin/bash
#SBATCH --job-name=vsi-qa
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/qa_%j.out
#SBATCH --error=logs/qa_%j.err

# Stage 6 downstream VideoQA (Qwen3-VL-8B, 8 frames + interleaved subtitles)
# on the GT-fixed cache. Heads come from results_v4 (default task order); each
# is scored on the test split of its own seed. Resumable: answers are cached
# per (query, frame set) in results_qa/answers.jsonl.

set -uo pipefail
cd "${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}" || exit 1
PY=${PY:-/home/jianxiang.he/.conda/envs/starVLA/bin/python}
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
$PY -c "import torch; assert torch.cuda.device_count() >= 1" || { echo "no GPU"; exit 1; }

HEADS=""
for st in lwf naive; do
  for s in 0 1 2; do HEADS="$HEADS results_v4/${st}_default_seed${s}.pt"; done
done

$PY cl/run_downstream_qa.py \
    --cache_dir /l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench \
    --data_root /l/users/jianxiang.he/continual-vsi/data/LongVideoBench \
    --methods uniform fixed:0.0 fixed:0.3 fixed:0.7 fixed:1.0 oracle $HEADS \
    --out_dir results_qa
echo "=== [$(date '+%F %T')] done ==="
