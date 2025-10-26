from typing import List, Dict, Any
import numpy as np
import cv2
from .common import Frame

SALIENCY_DOWNSCALE = 192
AREA_THRESH = 0.12


def _salient_area_ratio(img_bgr, q: float = 0.6) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        img = img_bgr
        w = img.shape[1]
        if w > SALIENCY_DOWNSCALE:
            s = SALIENCY_DOWNSCALE / w
            img = cv2.resize(img, (int(w*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        ok, smap = sal.computeSaliency(img)
        if not ok:
            return 0.0
        t = np.quantile(smap, q)
        return float((smap >= t).mean())
    except Exception:
        return 0.0


def compute(frames: List[Frame]) -> Dict[str, Any]:
    ratios = [(f.t, _salient_area_ratio(f.img)) for f in frames]
    t_first = None
    max_ratio = 0.0
    t_peak = None
    dwell = 0.0
    for i, (t, r) in enumerate(ratios):
        if t_first is None and r >= AREA_THRESH:
            t_first = t
        if r > max_ratio:
            max_ratio = r
            t_peak = t
    # crude dwell: count frames over thresh * avg dt
    if len(ratios) >= 2:
        dt = (ratios[-1][0] - ratios[0][0]) / max(1, len(ratios)-1)
        dwell = float(sum(1 for (_, r) in ratios if r >= AREA_THRESH) * max(0.0, dt))
    return {
        "t_first_product": t_first,
        "t_peak_product": t_peak,
        "max_product_area_ratio": float(max_ratio),
        "product_dwell": float(dwell),
    }
