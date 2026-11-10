#!/usr/bin/env python3
"""
Stage 6: downstream VideoQA. Does a better λ actually help a VLM answer?

For every cached LongVideoBench query, each method picks a λ, the cached
branch curves are fused, the top-k seconds become the k input frames, and
Qwen3-VL-8B-Instruct answers the multiple-choice question from those frames
plus the subtitles shown at each frame's time (LongVideoBench's interleaved
format). Methods:

    uniform        k evenly spaced frames, ignores λ
    fixed:<λ>      constant λ (0.3 = VSI default)
    oracle         per-query λ maximising keyframe recall (uses GT; ceiling)
    <ckpt>.pt      fusion head from cl/train_cl.py, evaluated only on the test
                   split of the seed it was trained with

Different methods often select the same frames for a query, so the VLM is
called once per distinct (query, frame set) and answers are cached in
<out_dir>/answers.jsonl (resumable). Per-method rows go to
<out_dir>/<method>.jsonl and accuracies to <out_dir>/summary.json.

The cache maps one query per video (the first annotation of that video);
annotations are matched back by video_id and GT positions.

Run with an env that has transformers >= 4.57 (Qwen3-VL), e.g. starVLA:
    python cl/run_downstream_qa.py \
        --cache_dir /l/users/jianxiang.he/continual-vsi/cache_fps24/LongVideoBench \
        --data_root /l/users/jianxiang.he/continual-vsi/data/LongVideoBench \
        --methods uniform fixed:0.3 oracle results_v4/lwf_default_seed0.pt \
        --out_dir results_qa
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cl.dataset import CLDataset
from cl.features import extract_features
from cl.fusion_head import FusionHead, decode_lambda
from cl.oracle import LAMBDA_GRID, normalize_minmax, score_fps

MODEL = "/l/users/jianxiang.he/ego-oracle-gap/models/Qwen3-VL-8B-Instruct"


def select_seconds(sample, lam: float, k: int) -> np.ndarray:
    """Top-k score-curve indices (sampled seconds) for one query, in time order."""
    fused = (1.0 - lam) * normalize_minmax(np.asarray(sample["s_obj"], float)) \
        + lam * normalize_minmax(np.asarray(sample["s_text"], float))
    return np.sort(np.argsort(fused)[-k:])


def oracle_lambda(sample) -> float:
    """Best-recall λ; among ties, the one closest to the VSI default 0.3."""
    r = np.asarray(sample["recall_curve"])
    best = np.flatnonzero(r == r.max())
    return float(LAMBDA_GRID[best[np.argmin(np.abs(LAMBDA_GRID[best] - 0.3))]])


def head_lambdas(ckpt_path, samples) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    head = FusionHead(in_dim=ckpt["in_dim"], n_bins=ckpt["n_bins"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    feature_set = ckpt.get("features", "base778")
    # CLDataset samples carry base778 features in "feat" (q_emb is not kept),
    # so only other feature sets are recomputed from the curves.
    X = np.stack([s["feat"] if feature_set == "base778"
                  else extract_features(s, feature_set) for s in samples])
    if ckpt.get("feat_mean") is not None:
        X = (X - ckpt["feat_mean"].numpy()) / ckpt["feat_std"].numpy()
    # Checkpoints from before the decode fix have no "decode" key; fall back
    # to the rule train_cl.py now applies.
    decode = ckpt.get("decode") or (
        "expected" if ckpt.get("loss") == "soft_topk" else "argmax")
    with torch.no_grad():
        lam = decode_lambda(head(torch.tensor(X, dtype=torch.float32)), decode)
    return lam.numpy().astype(float)


def _ts(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_subtitles(path, offset):
    """[(start, end, text)] in clip time. Two formats ship with LongVideoBench;
    clipped videos need the subtitle times shifted by their start offset."""
    if not os.path.exists(path):
        return []
    out = []
    for u in json.load(open(path)):
        if "line" in u:
            st, en, txt = _ts(u["start"]), _ts(u["end"]), u["line"]
        elif "timestamp" in u:
            t = u["timestamp"]
            if t is None or t[0] is None:
                continue
            st, en, txt = float(t[0]), float(t[1] if t[1] is not None else t[0]), u["text"]
        else:
            continue
        txt = str(txt).replace("&nbsp;", " ").replace("\n", " ").strip()
        if txt:
            out.append((st - offset, en - offset, txt))
    return out


def subtitle_at(subs, t, pad=1.0):
    return " ".join(txt for st, en, txt in subs if st - pad <= t <= en + pad)


class QA:
    def __init__(self, model_path, max_pixels):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model_path,
                                                  max_pixels=max_pixels)

    def __call__(self, images, times, subs, question, candidates):
        content = []
        for i, (t, img) in enumerate(zip(times, images)):
            content.append({"type": "text", "text": f"Frame {i + 1} ({t:.0f}s):"})
            content.append({"type": "image"})
            sub = subtitle_at(subs, t)
            if sub:
                content.append({"type": "text", "text": f"Subtitle: {sub}"})
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(candidates))
        content.append({"type": "text", "text":
                        f"Question: {question}\n{opts}\n"
                        "Answer with the option's letter from the given "
                        "choices directly."})
        msgs = [{"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images,
                           return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8,
                                      do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="configs/tasks_lvb4.yaml")
    ap.add_argument("--data_root", required=True,
                    help="has lvb_val_grounded.json, videos/, subtitles/")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max_pixels", type=int, default=448 * 448)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ds = CLDataset(args.cache_dir, args.config)
    samples = ds.samples[:args.limit] if args.limit else ds.samples
    by_video = defaultdict(list)
    for a in json.load(open(os.path.join(args.data_root,
                                         "lvb_val_grounded.json"))):
        by_video[a["video_id"]].append(a)
    meta = {}
    for s in samples:
        gt = list(np.atleast_1d(s["gt_frames"]))
        hit = [a for a in by_video[s["qid"]] if list(a["position"]) == gt]
        if len(hit) != 1:
            raise ValueError(f"{s['qid']}: {len(hit)} matching annotations")
        meta[s["qid"]] = hit[0]

    # method -> {qid: λ (nan = uniform)}
    plans = {}
    test_of = {}
    for m in args.methods:
        if m == "uniform":
            plans[m] = {s["qid"]: np.nan for s in samples}
        elif m.startswith("fixed:"):
            plans[m] = {s["qid"]: float(m[6:]) for s in samples}
        elif m == "oracle":
            plans[m] = {s["qid"]: oracle_lambda(s) for s in samples}
        else:
            seed = torch.load(m, map_location="cpu")["seed"]
            split = CLDataset(args.cache_dir, args.config, split_seed=seed)
            test = [s for t in split.task_names
                    for s in split.get_task_splits(t)[1]]
            keep = {s["qid"] for s in samples}
            test = [s for s in test if s["qid"] in keep]
            name = Path(m).parent.name + "/" + Path(m).stem
            plans[name] = dict(zip([s["qid"] for s in test],
                                   head_lambdas(m, test)))
            test_of[name] = seed

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "answers.jsonl")
    answers = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            if l.strip():
                r = json.loads(l)
                answers[r["key"]] = r

    by_qid = {s["qid"]: s for s in samples}

    def frames_for(s, lam):
        """(score-curve seconds, raw video frame indices) for one plan."""
        n_raw = int(s["n_frames"])
        if np.isnan(lam):
            raw = np.linspace(0, n_raw, num=args.k, endpoint=False).astype(int)
            return raw / (n_raw / float(s["duration"])), raw
        sec = select_seconds(s, lam, args.k)
        t = sec / score_fps(s)
        raw = np.round(t * n_raw / float(s["duration"])).astype(int)
        return t, np.clip(raw, 0, n_raw - 1)

    jobs = {}
    for name, plan in plans.items():
        for qid, lam in plan.items():
            t, raw = frames_for(by_qid[qid], lam)
            key = qid + "|" + ",".join(map(str, raw))
            jobs.setdefault(key, (qid, t, raw))
    todo = [k for k in jobs if k not in answers]
    print(f"{len(jobs)} distinct (query, frames), {len(todo)} to run",
          flush=True)

    if todo:
        import decord
        from PIL import Image
        qa = QA(args.model, args.max_pixels)
        with open(cache_path, "a") as fout:
            for i, key in enumerate(todo):
                qid, t, raw = jobs[key]
                a = meta[qid]
                try:
                    vr = decord.VideoReader(os.path.join(
                        args.data_root, "videos", a["video_path"]))
                    raw = np.clip(raw, 0, len(vr) - 1)
                    imgs = [Image.fromarray(f)
                            for f in vr.get_batch(list(raw)).asnumpy()]
                    subs = load_subtitles(
                        os.path.join(args.data_root, "subtitles",
                                     a["subtitle_path"]),
                        float(a.get("starting_timestamp_for_subtitles", 0)))
                    pred = qa(imgs, t, subs, a["question"], a["candidates"])
                except Exception as e:                  # noqa: BLE001
                    print(f"[skip] {key}: {e}", flush=True)
                    continue
                gold = chr(65 + int(a["correct_choice"]))
                r = {"key": key, "qid": qid, "pred": pred, "gold": gold,
                     "correct": pred[:1].upper() == gold}
                answers[key] = r
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                if (i + 1) % 50 == 0:
                    print(f"[{i + 1}/{len(todo)}]", flush=True)

    summary = {}
    for name, plan in plans.items():
        rows = []
        for qid, lam in plan.items():
            t, raw = frames_for(by_qid[qid], lam)
            r = answers.get(qid + "|" + ",".join(map(str, raw)))
            if r is None:
                continue
            rows.append({"qid": qid, "task": by_qid[qid]["task"],
                         "lambda": None if np.isnan(lam) else float(lam),
                         "pred": r["pred"], "gold": r["gold"],
                         "correct": r["correct"]})
        with open(os.path.join(args.out_dir,
                               name.replace("/", "__") + ".jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        per_task = defaultdict(list)
        for r in rows:
            per_task[r["task"]].append(r["correct"])
        summary[name] = {
            "n": len(rows), "acc": float(np.mean([r["correct"] for r in rows]))
            if rows else None,
            "split_seed": test_of.get(name),
            "per_task": {t: float(np.mean(v)) for t, v in per_task.items()}}
        print(f"{name:<40} n={len(rows):<4} acc={summary[name]['acc']}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
