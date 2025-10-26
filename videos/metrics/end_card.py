from typing import List, Dict, Any
import numpy as np
import cv2
from .common import Frame, get_gray

WINDOW_S = 3.0


def _last_seconds(frames: List[Frame], seconds: float = WINDOW_S) -> List[Frame]:
    if not frames:
        return []
    end_t = frames[-1].t
    start_t = max(0.0, end_t - seconds)
    return [f for f in frames if f.t >= start_t]


def _staticity_series(frames_subset: List[Frame]):
    diffs = []
    for i in range(1, len(frames_subset)):
        a = get_gray(frames_subset[i-1].img)
        b = get_gray(frames_subset[i].img)
        d = float(np.mean(cv2.absdiff(a, b)) / 255.0)
        diffs.append((frames_subset[i].t, d))
    return diffs


def _text_presence(img_bgr) -> float:
    gray = get_gray(img_bgr)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    h, w = gray.shape[:2]
    area = 0
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 60:
            continue
        area += cw * ch
    return float(area) / float(max(1, h*w))


def compute(frames: List[Frame]) -> Dict[str, Any]:
    subset = _last_seconds(frames, WINDOW_S)
    if len(subset) < 2:
        return {"endcard_present": False, "endcard_duration": 0.0, "endcard_idx": None, "endcard_readability": 0.0}
    diffs = _staticity_series(subset)
    if not diffs:
        return {"endcard_present": False, "endcard_duration": 0.0, "endcard_idx": None, "endcard_readability": 0.0}
    min_idx = int(np.argmin([d for (_, d) in diffs])) + 1
    best_frame = subset[min_idx]
    thr = float(np.median([d for (_, d) in diffs]) * 0.8 + 0.02)
    start_t = best_frame.t
    end_t = best_frame.t
    for i in range(min_idx, 0, -1):
        if diffs[i-1][1] < thr:
            start_t = subset[i-1].t
        else:
            break
    for i in range(min_idx+1, len(subset)):
        if diffs[i-1][1] < thr:
            end_t = subset[i].t
        else:
            break
    readability = _text_presence(best_frame.img)
    return {
        "endcard_present": True,
        "endcard_duration": float(max(0.0, end_t - start_t)),
        "endcard_idx": frames.index(best_frame),
        "endcard_readability": float(readability),
    }
