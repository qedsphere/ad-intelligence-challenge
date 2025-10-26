from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from .common import Frame

SSIM_DOWNSCALE = 112
WINDOW_S = 1.0


def _avg_frame(frames_subset: List[Frame]) -> Optional[np.ndarray]:
    if not frames_subset:
        return None
    acc = None
    for f in frames_subset:
        img = f.img.astype(np.float32)
        if acc is None:
            acc = img
        else:
            acc += img
    acc /= max(1, len(frames_subset))
    return acc.astype(np.uint8)


def _frame_ssim(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
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


def _hist_distance(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([a_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    h2 = cv2.calcHist([b_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"loop_readiness_score": 0.0, "ssim_first_last": 0.0, "hist_first_last": 1.0}
    start_t = frames[0].t
    end_t = frames[-1].t
    first = [f for f in frames if start_t <= f.t <= (start_t + WINDOW_S)]
    last = [f for f in frames if (end_t - WINDOW_S) <= f.t <= end_t]
    a = _avg_frame(first)
    b = _avg_frame(last)
    if a is None or b is None:
        return {"loop_readiness_score": 0.0, "ssim_first_last": 0.0, "hist_first_last": 1.0}
    ssim_val = _frame_ssim(a, b)
    hdist = _hist_distance(a, b)
    # higher ssim and lower hist distance => better
    score = float(max(0.0, ssim_val - 0.5) * (1.0 - min(1.0, hdist)))
    return {
        "ssim_first_last": float(ssim_val),
        "hist_first_last": float(hdist),
        "loop_readiness_score": float(score),
    }
