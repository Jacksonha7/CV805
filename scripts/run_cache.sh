#!/bin/bash
#SBATCH --job-name=vsi-cache
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cache_%j.out
#SBATCH --error=logs/cache_%j.err

# Stage 1: Cache VSI scores for all LongVideoBench queries.
# Run once; all CL experiments use the cached data.

PYTHON=/home/jianxiang.he/.conda/envs/vsi/bin/python
PROJ=${PROJ:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
CACHE_DIR=/l/users/jianxiang.he/continual-vsi/cache/LongVideoBench
DATA_DIR=/l/users/jianxiang.he/continual-vsi/data/LongVideoBench

cd $PROJ

$PYTHON cl/cache_scores.py \
    --dataset LongVideoBench \
    --input_json $DATA_DIR/lvb_val_grounded.json \
    --video_root $DATA_DIR/videos \
    --subtitle_root $DATA_DIR/subtitles \
    --config_path third_party/YOLO-World/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
    --checkpoint_path /l/users/jianxiang.he/YOLO-World/weights/s_stage1-d1c1d7d8.pth \
    --cache_dir $CACHE_DIR \
    --device cuda:0 \
    --text_weight 0.3

echo "=== Caching done ==="

# Stage 2a: Compute oracle lambda
$PYTHON cl/oracle.py --cache_dir $CACHE_DIR --k 8 --threshold 5.0

echo "=== Oracle lambda done ==="
