#!/bin/bash
#SBATCH --job-name=vsi-cl
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cl_%j.out
#SBATCH --error=logs/cl_%j.err

# SLURM wrapper for run_cl_experiments.sh (CPU-only, 8 strategies x 2 orders).
#
# The guard below is load-bearing. oracle.py writes oracle_lambda_idx back into
# each cached .npz in place, and cl/dataset.py reads it as
# data.get("oracle_lambda_idx", 0) — a MISSING key silently becomes label 0
# rather than raising. Running these experiments before oracle therefore
# produces a full set of plausible-looking, entirely meaningless results.

set -uo pipefail

PYTHON=/home/jianxiang.he/.conda/envs/vsi/bin/python
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
CACHE=/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench

cd "$PROJ" || exit 1

echo "=== [$(date '+%F %T')] oracle-key guard ==="
$PYTHON - "$CACHE" <<'PY'
import glob, os, sys
import numpy as np

cache = sys.argv[1]
files = sorted(glob.glob(os.path.join(cache, "*.npz")))
if not files:
    print(f"!!! no .npz under {cache}")
    sys.exit(1)

missing = []
for p in files:
    with np.load(p, allow_pickle=True) as d:
        if "oracle_lambda_idx" not in d:
            missing.append(os.path.basename(p))

print(f"cached={len(files)} without_oracle_key={len(missing)}")
if missing:
    print("sample:", missing[:10])
    sys.exit(1)
sys.exit(0)
PY

if [ $? -ne 0 ]; then
    echo "!!! cache is missing oracle_lambda_idx — refusing to run."
    echo "!!! dataset.py would silently label every sample 0."
    echo "!!! run scripts/run_cache_resume_and_oracle.sh first."
    exit 1
fi

echo "=== [$(date '+%F %T')] running CL experiments ==="
bash scripts/run_cl_experiments.sh
echo "=== [$(date '+%F %T')] done ==="
