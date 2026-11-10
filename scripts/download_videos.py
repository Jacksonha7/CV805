#!/usr/bin/env python3
"""Download and extract LongVideoBench videos from HuggingFace."""
import os
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download, login

token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
login(token=token)

DATA_DIR = "/l/users/jianxiang.he/continual-vsi/data/LongVideoBench"
os.makedirs(DATA_DIR, exist_ok=True)

api = HfApi()
files = api.list_repo_files("longvideobench/LongVideoBench", repo_type="dataset")
video_parts = sorted([f for f in files if f.startswith("videos.tar.part.")])
print(f"Found {len(video_parts)} video part files")

for part in video_parts:
    part_path = os.path.join(DATA_DIR, part)
    if os.path.exists(part_path):
        print(f"[SKIP] {part} already downloaded")
        continue
    print(f"[DOWNLOAD] {part} ...")
    hf_hub_download(
        "longvideobench/LongVideoBench",
        part,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print(f"[DONE] {part}")

# Merge and extract
merged = os.path.join(DATA_DIR, "videos.tar")
if not os.path.exists(merged):
    print("Merging parts...")
    parts_glob = os.path.join(DATA_DIR, "videos.tar.part.*")
    cmd = f"cat {parts_glob} > {merged}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Merged -> {merged}")

videos_dir = os.path.join(DATA_DIR, "videos")
# Check if already extracted
existing = len([f for f in os.listdir(videos_dir) if f.endswith(".mp4")]) \
    if os.path.isdir(videos_dir) else 0
if existing > 100:
    print(f"[SKIP] Already {existing} videos extracted")
else:
    print("Extracting videos...")
    subprocess.run(["tar", "xf", merged, "-C", DATA_DIR], check=True)
    new_count = len([f for f in os.listdir(videos_dir) if f.endswith(".mp4")])
    print(f"Extracted {new_count} videos")

print("Done!")
