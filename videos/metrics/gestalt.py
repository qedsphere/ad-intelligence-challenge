from typing import List, Dict, Any
import numpy as np
import cv2
from .common import Frame


def _edge_clutter(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean()) / 255.0


def _isolation_ratio(img_bgr) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if not ok:
            return 0.0
        t = np.quantile(smap, 0.8)
        mask = (smap >= t).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = float(areas.max())
        total = float(mask.sum())
        return float(min(1.0, largest / max(1e-6, total)))
    except Exception:
        return 0.0


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"clutter": 0.0, "isolation": 0.0}
    step = max(1, len(frames)//6 or 1)
    samples = [frames[i].img for i in range(0, len(frames), step)[:6]]
    clutter = float(np.mean([_edge_clutter(img) for img in samples])) if samples else 0.0
    isolation = float(np.mean([_isolation_ratio(img) for img in samples])) if samples else 0.0
    return {"clutter": clutter, "isolation": isolation}
