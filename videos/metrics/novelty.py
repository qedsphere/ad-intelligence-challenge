from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame

SSIM_DOWNSCALE = 112


def _hist_bhatt(a_bgr, b_bgr) -> float:
    a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([a_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    h2 = cv2.calcHist([b_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def _frame_ssim(a_bgr, b_bgr) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        if SSIM_DOWNSCALE:
            a_bgr = cv2.resize(a_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
            b_bgr = cv2.resize(b_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
        a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        return float(ssim(a_gray, b_gray))
    except Exception:
        return 0.0


def compute(frames: List[Frame], deps: Dict[str, Any] = None) -> Dict[str, Any]:
    deps = deps or {}
    key_idxs = deps.get("shot_detection", {}).get("keyframe_indices", [])
    if not key_idxs:
        step = max(1, len(frames)//6 or 1)
        key_idxs = list(range(0, len(frames), step))[:6]
    if len(key_idxs) < 2:
        return {"novelty_peak": 0.0, "anomaly_count": 0}
    scores = []
    anomalies = 0
    for a, b in zip(key_idxs[:-1], key_idxs[1:]):
        img_a = frames[a].img
        img_b = frames[b].img
        hdist = _hist_bhatt(img_a, img_b)
        ssim_val = _frame_ssim(img_a, img_b)
        novelty = 0.5 * hdist + 0.5 * (1.0 - ssim_val)
        scores.append(novelty)
        if novelty > 0.5:
            anomalies += 1
    peak = max(scores) if scores else 0.0
    return {"novelty_peak": float(peak), "anomaly_count": int(anomalies)}
