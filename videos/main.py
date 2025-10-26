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

# Serial videos, parallel tasks per video
PARALLEL_TASKS = True
MAX_TASK_WORKERS = max(1, (os.cpu_count() or 4) - 1)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ads" / "videos"
OUTPUT_CSV = ROOT / "video_features.csv"

# Analysis knobs
TARGET_FPS = 12.0
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
]
# wave 2: depends on shot_detection
WAVE2 = [
    ("montage_fastcuts", "metrics.montage_fastcuts"),
]


def list_videos(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".avi")])


def _import(modpath: str):
    return importlib.import_module(modpath)


def run_wave(frames, wave, deps: Dict[str, Any], pool: futures.Executor) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    def run_task(key, modpath):
        try:
            mod = _import(modpath)
            if key == "montage_fastcuts":
                out = mod.compute(frames, deps)
            elif key in ("cta_spans",):
                out = mod.compute(frames, deps)
            else:
                # metric modules with compute(frames)
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


def score_video(all_feats: Dict[str, Any]) -> float:
    # Weighted composite (adjust weights above)
    s = 0.0
    sd = all_feats.get("shot_detection", {})
    f3 = all_feats.get("first3s", {})
    cta = all_feats.get("cta_spans", {})
    endc = all_feats.get("end_card", {})
    loopr = all_feats.get("loop_readiness", {})
    textd = all_feats.get("text_dense", {})
    motion = all_feats.get("motion_bursts", {})
    reveal = all_feats.get("product_reveal", {})
    montage = all_feats.get("montage_fastcuts", {})

    s += W_NUM_SHOTS * float(sd.get("num_shots", 0))
    s += W_START_STRENGTH * float(f3.get("start_strength", 0.0))
    s += W_CTA_PRESENT * (1.0 if bool(cta.get("cta_present", False)) else 0.0)
    s += W_ENDCARD_READABILITY * float(endc.get("endcard_readability", 0.0))
    s += W_LOOP_READINESS * float(loopr.get("loop_readiness_score", 0.0))
    s += W_MOTION_MEAN * float(motion.get("motion_intensity_mean", 0.0))
    s += W_PRODUCT_AREA * float(reveal.get("max_product_area_ratio", 0.0))
    s += W_TEXT_TOPK * float(len(textd.get("top_text", [])))
    s += W_MONTAGE_DENSITY * float(montage.get("peak_cut_density", 0.0))
    return float(s)


def main() -> int:
    videos = list_videos(DATA_DIR)
    if not videos:
        print(f"No videos found in {DATA_DIR}")
        return 1
    print(f"Processing {len(videos)} videos serially; tasks parallel={PARALLEL_TASKS} (workers={MAX_TASK_WORKERS})")

    rows: List[Dict[str, Any]] = []
    total_elapsed = 0.0

    with futures.ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS) as pool:
        for path in videos:
            print(f"\n=== {path.name} ===")
            t0 = time.perf_counter()
            frames = extract_frames(path)
            # Wave 1: frame-only tasks
            wave1_out = run_wave(frames, WAVE1, deps={}, pool=pool)
            # Wave 2: tasks that depend on shot_detection
            deps = {"shot_detection": wave1_out.get("shot_detection", {})}
            wave2_out = run_wave(frames, WAVE2, deps=deps, pool=pool)
            all_out = {**wave1_out, **wave2_out}
            score = score_video(all_out)
            elapsed_s = time.perf_counter() - t0
            total_elapsed += elapsed_s
            flat: Dict[str, Any] = {"file": path.name, "score": score, "elapsed_s": elapsed_s}
            # flatten some keys (shallow)
            for k, v in all_out.items():
                if isinstance(v, dict):
                    for ik, iv in v.items():
                        if isinstance(iv, (str, int, float, bool)):
                            flat[f"{k}.{ik}"] = iv
            rows.append(flat)
            print(f"Processed in {elapsed_s:.2f}s")

    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nVideos in order of score:")
    for r in rows:
        print(f"  {r['file']}: score={r['score']:.3f} | {r['elapsed_s']:.2f}s")
    print(f"Total processing time: {total_elapsed:.2f}s")
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
