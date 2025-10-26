from typing import List, Dict, Any
import numpy as np
import cv2
from .common import Frame

NUM_SAMPLES = 8


def _warm_cool_balance(img_bgr) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0
    warm = ((h <= 30) | (h >= 330)).astype(np.float32).mean()
    cool = ((h >= 90) & (h <= 150)).astype(np.float32).mean()
    denom = max(1e-6, warm + cool)
    return float((warm - cool) / denom)


def _palette_stability(frames: List[Frame], indices: List[int]) -> float:
    if not indices:
        return 0.0
    vals = []
    for i in indices:
        hsv = cv2.cvtColor(frames[i].img, cv2.COLOR_BGR2HSV)
        vals.append(float(np.mean(hsv[..., 0])))
    return float(np.std(vals) / 90.0)


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"warm_cool": 0.0, "palette_stability": 0.0}
    step = max(1, len(frames) // NUM_SAMPLES)
    sample_idx = list(range(0, len(frames), step))[:NUM_SAMPLES]
    wc_vals = [_warm_cool_balance(frames[i].img) for i in sample_idx]
    warm_cool = float(np.mean(wc_vals)) if wc_vals else 0.0
    palette_var = _palette_stability(frames, sample_idx)
    return {
        "warm_cool": warm_cool,
        "palette_stability": float(palette_var),
    }
