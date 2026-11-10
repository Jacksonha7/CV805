#!/bin/bash
# Full experiment grid: 8 strategies x 3 orders x 3 seeds = 72 runs.
# CPU-only — everything runs on the cached score curves, so the whole grid is
# minutes, not hours. Requires the cache to have been through cl/oracle.py.
#
#   bash scripts/run_all.sh              # default soft_topk loss
#   LOSS=ce_oracle bash scripts/run_all.sh
#
# Re-running is safe: existing result files are skipped.

set -uo pipefail

PYTHON=${PYTHON:-/home/jianxiang.he/.conda/envs/vsi/bin/python}
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
CACHE=${CACHE:-/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench}
CONFIG=$PROJ/configs/tasks_lvb4.yaml

# soft_topk is the default: with ~1.8 GT frames and k=8, recall@8 is flat in λ
# for ~60% of queries, which collapses every oracle λ* onto the 0.5 tie-break
# and makes every strategy score identically. Optimising the fused curve
# directly restores a usable gradient. See README "Known pitfalls".
LOSS=${LOSS:-soft_topk}

# query_oracle = per-query λ* (noisy); task_oracle = per-task optimum.
LABEL=${LABEL:-query_oracle}
FEATURES=${FEATURES:-base778}
EXTRA=${EXTRA:-}
RESULTS=${RESULTS:-results}
FIGS=${FIGS:-figures}

STRATEGIES=${STRATEGIES:-"naive ewc lwf er joint fixed oracle knn"}
ORDERS=${ORDERS:-"default grouped random_seed0"}
SEEDS=${SEEDS:-"0 1 2"}

cd "$PROJ" || exit 1
mkdir -p "$RESULTS" logs "$FIGS"

total=0
skipped=0
failed=0
for order in $ORDERS; do
    for strat in $STRATEGIES; do
        for seed in $SEEDS; do
            OUT="$RESULTS/${strat}_${order}_seed${seed}.json"
            total=$((total + 1))
            if [ -f "$OUT" ]; then
                skipped=$((skipped + 1))
                continue
            fi
            echo "[RUN] strategy=$strat order=$order seed=$seed loss=$LOSS"
            if ! $PYTHON cl/train_cl.py \
                    --cache_dir "$CACHE" \
                    --config "$CONFIG" \
                    --strategy "$strat" \
                    --order "$order" \
                    --loss "$LOSS" \
                    --label "$LABEL" \
                    --features "$FEATURES" $EXTRA \
                    --epochs 50 \
                    --lr 1e-3 \
                    --seed "$seed" \
                    --out "$OUT" \
                    > "logs/$(basename "$RESULTS")_${strat}_${order}_seed${seed}.log" 2>&1; then
                echo "  !! failed — see logs/$(basename "$RESULTS")_${strat}_${order}_seed${seed}.log"
                failed=$((failed + 1))
            fi
        done
    done
done

echo ""
echo "=== grid done: $total configs, $skipped skipped, $failed failed ==="
[ "$failed" -gt 0 ] && echo "!! some runs failed; analysis will be incomplete"

echo "=== analysis ==="
$PYTHON cl/analyze.py --results_dir "$RESULTS/" --fig_dir "$FIGS/"
