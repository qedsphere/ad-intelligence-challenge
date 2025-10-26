from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame

SAMPLE_STEP = 10


def _center_saliency_ratio(img_bgr) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if not ok:
            return 0.0
        h, w = smap.shape[:2]
        cx0, cy0 = int(w * 0.33), int(h * 0.33)
        cx1, cy1 = int(w * 0.67), int(h * 0.67)
        center = smap[cy0:cy1, cx0:cx1]
        return float(center.mean() / max(1e-6, smap.mean()))
    except Exception:
        return 0.0


def _text_boxes(img_bgr) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    boxes = []
    for cnt in regions:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 60:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _text_safe_area_ratio(img_bgr) -> float:
    h, w = img_bgr.shape[:2]
    sx0, sy0 = int(w * 0.1), int(h * 0.1)
    sx1, sy1 = int(w * 0.9), int(h * 0.9)
    boxes = _text_boxes(img_bgr)
    if not boxes:
        return 1.0
    total = 0
    inside = 0
    for (x, y, bw, bh) in boxes:
        total += 1
        cx, cy = x + bw // 2, y + bh // 2
        if sx0 <= cx <= sx1 and sy0 <= cy <= sy1:
            inside += 1
    return float(inside / max(1, total))


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"center_saliency_ratio": 0.0, "safe_text_ratio": 1.0}
    step = max(1, len(frames)//6 or 1)
    sample_imgs = [frames[i].img for i in range(0, len(frames), step)[:6]]
    center_sal = float(np.mean([_center_saliency_ratio(img) for img in sample_imgs])) if sample_imgs else 0.0
    # Sample fewer frames for text boxes
    sample_imgs_text = [frames[i].img for i in range(0, len(frames), max(1, SAMPLE_STEP))]
    safe_text = float(np.mean([_text_safe_area_ratio(img) for img in sample_imgs_text])) if sample_imgs_text else 1.0
    return {
        "center_saliency_ratio": center_sal,
        "safe_text_ratio": safe_text,
    }
