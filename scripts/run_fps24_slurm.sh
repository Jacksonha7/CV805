#!/bin/bash
#SBATCH --job-name=vsi-fps24
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fps24_%j.out
#SBATCH --error=logs/fps24_%j.err

# Re-run the v3 configuration (stats10_pq44 features, z-scored) after the GT
# frame-rate fix (oracle.GT_FPS=24). Uses cache_fps24/, a copy of the score
# cache whose recall curves / oracle λ were recomputed with the fix, so the
# original cache and results_v2/ v3/ stay as they were.

set -uo pipefail
cd "${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}" || exit 1

echo "=== [$(date '+%F %T')] v4 grid -> results_v4 ==="
CACHE=${CACHE:-/l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench} \
FEATURES=stats10_pq44 EXTRA=--standardize \
RESULTS=results_v4 FIGS=figures_v4 bash scripts/run_all.sh
echo "=== [$(date '+%F %T')] done ==="
