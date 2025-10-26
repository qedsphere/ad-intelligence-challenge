from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame


def _brightness_score(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray) / 128.0)


def _sharpness(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _saliency_mean(img_bgr) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if ok:
            return float(np.mean(smap))
    except Exception:
        pass
    return 0.0


def _text_presence(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    h, w = gray.shape[:2]
    area = 0
    count = 0
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 20:
            continue
        area += cw * ch
        count += 1
    area_ratio = min(1.0, area / float(max(1, h * w)))
    return float(0.6 * area_ratio + 0.4 * (np.log1p(count) / 6.0))


def _thumb_score(img_bgr) -> float:
    sharp = _sharpness(img_bgr)
    sal = _saliency_mean(img_bgr)
    txt = _text_presence(img_bgr)
    br = _brightness_score(img_bgr)
    return 0.35 * sharp + 0.25 * sal + 0.20 * (txt * 100.0) + 0.20 * br


def compute(frames: List[Frame], deps: Dict[str, Any]) -> Dict[str, Any]:
    shot = deps.get("shot_detection", {})
    endc = deps.get("end_card", {})
    textd = deps.get("text_dense", {})

    key_idxs = shot.get("keyframe_indices", []) or []
    end_idx = endc.get("endcard_idx", None)
    text_idxs = [idx for (_, _, _, idx) in textd.get("top_text", [])]

    candidates: List[Tuple[float, float, int]] = []  # (score, t, idx)
    for idx in key_idxs:
        if 0 <= idx < len(frames):
            candidates.append((_thumb_score(frames[idx].img), frames[idx].t, idx))
    if isinstance(end_idx, int) and 0 <= end_idx < len(frames):
        candidates.append((_thumb_score(frames[end_idx].img), frames[end_idx].t, end_idx))
    for idx in text_idxs:
        if 0 <= idx < len(frames):
            candidates.append((_thumb_score(frames[idx].img), frames[idx].t, idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top3 = candidates[:min(3, len(candidates))]
    top_score = top3[0][0] if top3 else 0.0
    return {
        "thumbnail_top_score": float(top_score),
        "thumbnail_candidates": [(float(s), float(t), int(i)) for (s, t, i) in top3],
    }
