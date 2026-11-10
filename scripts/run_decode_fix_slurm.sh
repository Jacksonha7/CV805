#!/bin/bash
#SBATCH --job-name=vsi-decode
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/decode_fix_%j.out
#SBATCH --error=logs/decode_fix_%j.err

# Re-run the full grid after the eval-decode fix (eval λ = softmax-weighted
# mean over bins, matching what soft_topk trains on). Writes to new dirs so
# results/ and results_task/ (argmax decode) stay for comparison.

set -uo pipefail
cd "${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}" || exit 1

echo "=== [$(date '+%F %T')] query_oracle grid -> results_v2 ==="
LABEL=query_oracle RESULTS=results_v2 FIGS=figures_v2 bash scripts/run_all.sh
# task_oracle runs use ce_oracle + argmax, which this fix does not change, so
# results_task/ stays valid. (Running it here without LOSS=ce_oracle would
# silently train soft_topk and ignore the task labels.)
echo "=== [$(date '+%F %T')] done ==="
