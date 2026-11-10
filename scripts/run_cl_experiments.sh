#!/bin/bash
# Run ALL CL experiments (CPU-only, no GPU needed).
# Call after caching + oracle computation.

PYTHON=/home/jianxiang.he/.conda/envs/vsi/bin/python
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
CACHE=/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench
CONFIG=$PROJ/configs/tasks_lvb4.yaml

cd $PROJ
mkdir -p results logs figures

STRATEGIES="naive ewc lwf er joint fixed oracle knn"
ORDERS="default grouped"

for order in $ORDERS; do
    for strat in $STRATEGIES; do
        OUT="results/${strat}_${order}.json"
        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT already exists"
            continue
        fi
        echo "[RUN] strategy=$strat order=$order"
        $PYTHON cl/train_cl.py \
            --cache_dir $CACHE \
            --config $CONFIG \
            --strategy $strat \
            --order $order \
            --loss ce_oracle \
            --epochs 50 \
            --lr 1e-3 \
            --seed 0 \
            --out $OUT \
            2>&1 | tee "logs/${strat}_${order}.log"
    done
done

echo ""
echo "=== All experiments done, running analysis ==="
$PYTHON cl/analyze.py --results_dir results/ --fig_dir figures/
