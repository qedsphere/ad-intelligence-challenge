#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures as futures
import importlib
import traceback
import time
import numpy as np
import json
# Serial videos, parallel tasks per video
PARALLEL_TASKS = True
MAX_TASK_WORKERS = max(1, (os.cpu_count() or 8) - 3)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ads" / "videos"
OUTPUT_CSV = ROOT / "video_features.csv"

# Analysis knobs
TARGET_FPS = 6.0
SHORT_EDGE = 576
MAX_FRAMES = 300

# Scoring weights (tweakable)
W_NUM_SHOTS = 0.10
W_START_STRENGTH = 0.20
W_CTA_PRESENT = 0.10
W_ENDCARD_READABILITY = 0.10
W_LOOP_READINESS = 0.10
W_MOTION_MEAN = 0.10
W_PRODUCT_AREA = 0.10
W_TEXT_TOPK = 0.05
W_MONTAGE_DENSITY = 0.05
# New metrics weights
W_THUMB_TOP = 0.05
W_NOVELTY_PEAK = -0.05
W_NOVELTY_ANOMALY = -0.05
W_FACE_TIME_RATIO = 0.05
W_FACE_AREA = 0.05
W_WARM_COOL = 0.00
W_PALETTE_STABILITY = -0.05
W_CLUTTER = -0.05
W_ISOLATION = 0.05
W_AR_CLASS = 0.02
W_LETTERBOX = -0.05
W_SIMPLICITY = 0.05
W_CENTER_SALIENCY = 0.03
W_SAFE_TEXT = 0.03

# Metrics modules map
# wave 1: frame-only
WAVE1 = [
    ("shot_detection", "metrics.shot_detection"),
    ("first3s", "metrics.first3s"),
    ("cta_spans", "metrics.cta_spans"),
    ("product_reveal", "metrics.product_reveal"),
    ("motion_bursts", "metrics.motion_bursts"),
    ("text_dense", "metrics.text_dense"),
    ("end_card", "metrics.end_card"),
    ("loop_readiness", "metrics.loop_readiness"),
    ("faces_gaze", "metrics.faces_gaze"),
    ("color_palette", "metrics.color_palette"),
    ("gestalt", "metrics.gestalt"),
    ("platform_norms", "metrics.platform_norms"),
    ("semantic", "metrics.semantic"),
    ("safe_area", "metrics.safe_area"),
]
# wave 2: depends on shot_detection and/or other wave1 outputs
WAVE2 = [
    ("montage_fastcuts", "metrics.montage_fastcuts"),
    ("novelty", "metrics.novelty"),
    ("thumbnails", "metrics.thumbnails"),
]

DEPS_KEYS = {"montage_fastcuts", "novelty", "thumbnails", "cta_spans"}


def list_videos(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".avi")])


def _import(modpath: str):
    return importlib.import_module(modpath)


def run_wave(frames, wave, deps: Dict[str, Any], pool: futures.Executor) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    def run_task(key, modpath):
        try:
            mod = _import(modpath)
            if key in DEPS_KEYS:
                out = mod.compute(frames, deps)
            else:
                out = mod.compute(frames)
            return key, out
        except Exception as e:
            return key, {"error": str(e), "traceback": traceback.format_exc()}

    if PARALLEL_TASKS and len(wave) > 1:
        futures_list = [pool.submit(run_task, key, modpath) for (key, modpath) in wave]
        for fut in futures.as_completed(futures_list):
            k, out = fut.result()
            results[k] = out
    else:
        for (key, modpath) in wave:
            k, out = run_task(key, modpath)
            results[k] = out
    return results


def extract_frames(path: Path):
    # Lazy import loader from metrics.common to avoid circular imports
    cm = _import("metrics.common")
    frames = cm.load_video_frames(path, target_fps=TARGET_FPS, short_edge=SHORT_EDGE, max_frames=MAX_FRAMES)
    cm.reset_gray_cache()
    return frames


def main() -> List[Dict[str, Any]]:
    videos = list_videos(DATA_DIR)
    if not videos:
        print(f"No videos found in {DATA_DIR}")
        return []
    print(f"Processing {len(videos)} videos serially; tasks parallel={PARALLEL_TASKS} (workers={MAX_TASK_WORKERS})")

    results: List[Dict[str, Any]] = []

    with futures.ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS) as pool:
        for path in videos:
            print(f"\n=== {path.name} ===")
            t0 = time.perf_counter()
            frames = extract_frames(path)
            # Wave 1: frame-only tasks
            wave1_out = run_wave(frames, WAVE1, deps={}, pool=pool)
            # Wave 2: tasks that depend on shot_detection
            deps = {
                "shot_detection": wave1_out.get("shot_detection", {}),
                "end_card": wave1_out.get("end_card", {}),
                "text_dense": wave1_out.get("text_dense", {}),
            }
            wave2_out = run_wave(frames, WAVE2, deps=deps, pool=pool)
            all_out = {**wave1_out, **wave2_out}
            elapsed_s = time.perf_counter() - t0
            per_video: Dict[str, Any] = {"file": path.name, "elapsed_s": elapsed_s}
            per_video.update(all_out)
            results.append(per_video)
            print(f"Processed in {elapsed_s:.2f}s")

    return results


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
    sys.exit(0)
