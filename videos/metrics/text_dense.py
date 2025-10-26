from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame, get_gray

TOP_K = 3
MSER_MIN_AREA = 60


def _text_cov_contrast(img_bgr) -> Tuple[float, float]:
    gray = get_gray(img_bgr)
    mser = cv2.MSER_create(min_area=MSER_MIN_AREA) if hasattr(cv2.MSER_create, "__call__") else cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    h, w = gray.shape[:2]
    area = 0
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < MSER_MIN_AREA:
            continue
        area += cw * ch
    coverage = float(area) / float(max(1, h*w))
    contrast = float(np.std(gray) / 255.0)
    return coverage, contrast


def compute(frames: List[Frame]) -> Dict[str, Any]:
    stats = []  # (coverage, contrast, idx, t)
    for i, f in enumerate(frames):
        cov, contr = _text_cov_contrast(f.img)
        stats.append((cov, contr, i, f.t))
    stats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = stats[:min(TOP_K, len(stats))]
    coverage_stats = [(t, cov, contr, idx) for (cov, contr, idx, t) in top]
    return {
        "top_text": coverage_stats,
    }
