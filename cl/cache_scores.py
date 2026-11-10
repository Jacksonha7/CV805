#!/usr/bin/env python3
"""
Stage 1: Cache VSI dual-branch scores for all queries.

For each query in LongVideoBench val (or VideoMME), runs the VSI pipeline
and saves the raw s_obj, s_text score curves, question embedding, and
ground-truth keyframe indices to a .npz file.  All downstream CL
experiments operate on these caches — no GPU needed after this step.

Usage:
    python cl/cache_scores.py \
        --dataset LongVideoBench \
        --input_json  data/processed/LongVideoBench/annotations.json \
        --video_root  data/processed/LongVideoBench/videos \
        --subtitle_root data/processed/LongVideoBench/subtitles \
        --config_path third_party/YOLO-World/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py \
        --checkpoint_path /l/users/jianxiang.he/YOLO-World/weights/s_stage1-d1c1d7d8.pth \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache/LongVideoBench \
        --device cuda:0

Supports --shard i/n for multi-GPU parallelism and --limit N for smoke tests.
Already-cached queries are skipped (resume-safe).
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from VSI_keyframe_search import (
    MultimodalTStarFramework,
    ScoreFusionManager,
    SubtitleProcessor,
    TextSimilarityCalculator,
    VideoSearchConfig,
)
from TStar.interface_yolo import YoloWorldInterface
from TStar.interface_searcher import TStarSearcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_question_embedding(text_calc, question: str) -> np.ndarray:
    """Extract the 768-d sentence embedding for a question string."""
    with torch.no_grad():
        emb = text_calc.text_encoder.encode(question, convert_to_tensor=False)
    return np.asarray(emb, dtype=np.float32)


def run_branches(
    config: VideoSearchConfig,
    yolo_scorer: YoloWorldInterface,
    grounding_objects: dict,
) -> dict:
    """
    Run both VSI branches on one query and extract raw scores.

    Strategy: run the visual search WITHOUT text fusion so that
    score_distribution stays as pure visual scores (s_obj).
    Text scores (s_text) are computed separately.  Downstream CL
    experiments re-fuse them with different λ values.
    """
    framework = MultimodalTStarFramework(config, None, yolo_scorer)

    target_objects = grounding_objects["target_objects"]
    cue_objects = grounding_objects["cue_objects"]

    searcher = TStarSearcher(
        video_path=config.video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=config.search_nframes,
        image_grid_shape=(config.grid_rows, config.grid_cols),
        output_dir=config.output_dir,
        confidence_threshold=config.confidence_threshold,
        search_budget=config.search_budget,
        yolo_scorer=yolo_scorer,
        update_method="spline",
    )
    searcher.relation_alpha = config.relation_alpha

    # Text branch: compute frame-level text similarity scores
    text_scores = framework.text_calculator.compute_similarity_scores(
        config.question, framework.subtitles, framework.total_frame_num, framework.fps
    )

    # Question embedding (768-d)
    q_emb = get_question_embedding(framework.text_calculator, config.question)

    # Visual branch: run search WITHOUT text fusion hook.
    # This keeps score_distribution as pure visual (YOLO spline) scores.
    all_frames, timestamps = framework._perform_search(searcher)

    # s_obj is the pure visual score distribution after iterative search
    s_obj = np.array(searcher.score_distribution, dtype=np.float32)

    # Resample text scores to match the visual distribution length
    fusion_mgr = ScoreFusionManager(config.text_weight)
    s_text = fusion_mgr._resample_text_scores(
        text_scores, len(s_obj)
    ).astype(np.float32)

    return {
        "s_obj": s_obj,
        "s_text": s_text,
        "q_emb": q_emb,
        "total_frames": framework.total_frame_num,
        "fps": framework.fps,
        "keyframe_timestamps": timestamps,
    }


def adaptive_grid(video_path: str, rows: int, cols: int) -> tuple:
    """Largest square grid that the video can actually fill, capped at rows×cols.

    Returns (rows, cols) unchanged for any video long enough. VSI samples at
    1 fps, so a clip of D seconds yields ~D frames; a 4×4 grid needs 16.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        n_sampled = int(n / fps) if fps > 0 else n
    except Exception:                                    # noqa: BLE001
        return rows, cols

    if n_sampled >= rows * cols:
        return rows, cols
    side = max(1, int(math.isqrt(max(n_sampled, 1))))
    side = min(side, rows, cols)
    log.info(f"  short video ({n_sampled} sampled frames): "
             f"grid {rows}x{cols} -> {side}x{side}")
    return side, side


def get_gt_frames(data_item: dict) -> np.ndarray:
    """Extract ground-truth keyframe indices from a data item."""
    if "position" in data_item:
        return np.array(data_item["position"], dtype=np.int64)
    if "frame_indexes" in data_item:
        return np.array(data_item["frame_indexes"], dtype=np.int64)
    return np.array([], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LongVideoBench")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--video_root", default=None)
    ap.add_argument("--subtitle_root", default=None)
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--search_nframes", type=int, default=8)
    ap.add_argument("--grid_rows", type=int, default=4)
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--text_weight", type=float, default=0.3)
    ap.add_argument("--shard", default=None, help="i/n for multi-GPU sharding")
    ap.add_argument("--limit", type=int, default=0, help="Limit queries for smoke test")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.input_json) as f:
        dataset = json.load(f)

    if args.limit > 0:
        dataset = dataset[: args.limit]

    # Sharding
    if args.shard:
        shard_i, shard_n = map(int, args.shard.split("/"))
        dataset = [d for i, d in enumerate(dataset) if i % shard_n == shard_i]
        log.info(f"Shard {shard_i}/{shard_n}: {len(dataset)} queries")

    log.info(f"Total queries to cache: {len(dataset)}")

    # Init YOLO-World
    yolo_scorer = YoloWorldInterface(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )

    cached, skipped, failed = 0, 0, 0
    for idx, item in enumerate(dataset):
        qid = item.get("video_id", f"q{idx}")
        out_path = Path(args.cache_dir) / f"{qid}.npz"

        if out_path.exists():
            skipped += 1
            continue

        log.info(f"[{idx+1}/{len(dataset)}] {qid}")

        try:
            video_path = item.get("video_path", "")
            if args.video_root and not os.path.isabs(video_path):
                video_path = os.path.join(args.video_root, video_path)

            subtitle_path = ""
            if args.subtitle_root:
                if args.dataset == "VideoMME":
                    subtitle_path = os.path.join(
                        args.subtitle_root, f"{qid}.srt"
                    )
                else:
                    subtitle_path = os.path.join(
                        args.subtitle_root, f"{qid}_en.json"
                    )

            options = item.get("options", item.get("candidates", ""))
            if isinstance(options, list):
                options = "\n".join(f"{chr(65+i)}) {o}"
                                    for i, o in enumerate(options))

            # VSI samples one frame per second and packs them into a
            # grid_rows x grid_cols mosaic for YOLO-World. A video shorter than
            # the grid cannot fill one, and the packer raises "Frame count does
            # not match grid dimensions". That excludes every clip <= 16s --
            # 197 of 1337 LongVideoBench val queries, and unevenly by task
            # (20.9% of T1 vs 12.2% of T4). Shrink the grid for those instead.
            grid_r, grid_c = adaptive_grid(video_path, args.grid_rows,
                                           args.grid_cols)

            config = VideoSearchConfig(
                video_path=video_path,
                subtitle_path=subtitle_path,
                question=item.get("question", ""),
                options=options,
                text_weight=args.text_weight,
                search_nframes=args.search_nframes,
                grid_rows=grid_r,
                grid_cols=grid_c,
                device=args.device,
            )

            result = run_branches(config, yolo_scorer, item["grounding_objects"])

            gt_frames = get_gt_frames(item)
            category = item.get("question_category", item.get("category", ""))

            np.savez_compressed(
                out_path,
                s_obj=result["s_obj"],
                s_text=result["s_text"],
                q_emb=result["q_emb"],
                gt_frames=gt_frames,
                n_frames=result["total_frames"],
                category=str(category),
                duration=float(item.get("duration", 0)),
            )
            cached += 1

        except Exception as e:
            log.error(f"Failed {qid}: {e}")
            failed += 1

        if (idx + 1) % 50 == 0:
            log.info(f"Progress: cached={cached} skipped={skipped} failed={failed}")

    log.info(f"Done. cached={cached} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
