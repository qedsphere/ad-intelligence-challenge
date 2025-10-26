from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from .common import Frame, get_gray

TEXT_THRESH = 0.02
AVG_SALIENCY_SAMPLES = 8


def _saliency_mean(img_bgr) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        w = img_bgr.shape[1]
        if w > 192:
            s = 192 / w
            img_bgr = cv2.resize(img_bgr, (int(w*s), int(img_bgr.shape[0]*s)), interpolation=cv2.INTER_AREA)
        ok, smap = sal.computeSaliency(img_bgr)
        if ok:
            return float(np.mean(smap))
    except Exception:
        pass
    return 0.0


def _motion_diff(a_bgr, b_bgr) -> float:
    a = get_gray(a_bgr)
    b = get_gray(b_bgr)
    return float(np.mean(cv2.absdiff(a, b)) / 255.0)


def _text_coverage(img_bgr) -> float:
    gray = get_gray(img_bgr)
    mser = cv2.MSER_create(min_area=60) if hasattr(cv2.MSER_create, "__call__") else cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    h, w = gray.shape[:2]
    area = 0
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 60:
            continue
        area += cw * ch
    return float(area) / float(max(1, h*w))


def compute(frames: List[Frame], prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    window = [f for f in frames if 0.0 <= f.t <= 3.0]
    if not window:
        return {"start_strength": 0.0, "t_first_action": None, "t_first_text": None, "opening_indices": []}

    # first action
    diffs = []
    for i in range(1, min(4, len(window))):
        diffs.append(_motion_diff(window[i-1].img, window[i].img))
    baseline = np.median(diffs) if diffs else 0.02
    threshold = baseline + 0.03
    t_first_action = None
    for i in range(1, len(window)):
        if _motion_diff(window[i-1].img, window[i].img) > threshold:
            t_first_action = window[i].t
            break

    # first text
    t_first_text = None
    covs = []
    for i, f in enumerate(window):
        c = _text_coverage(f.img)
        covs.append(c)
        if t_first_text is None and c > TEXT_THRESH:
            t_first_text = f.t

    # rank top opening indices by sharpness+saliency+text
    scores = []
    for i, f in enumerate(window):
        sharp = float(cv2.Laplacian(get_gray(f.img), cv2.CV_64F).var())
        sal = _saliency_mean(f.img)
        score = 0.5*sharp + 0.3*sal + 0.2*(covs[i]*100.0)
        scores.append((score, i))
    scores.sort(reverse=True)
    opening_indices = [i for (_, i) in scores[:min(3, len(scores))]]

    # start strength
    has_action_early = 1.0 if (t_first_action is not None and t_first_action <= 2.0) else 0.0
    has_text_early = 1.0 if (t_first_text is not None and t_first_text <= 2.0) else 0.0
    avg_saliency = float(np.mean([_saliency_mean(f.img) for f in window[:min(AVG_SALIENCY_SAMPLES, len(window))]])) if window else 0.0
    start_strength = float(np.clip(0.4*has_action_early + 0.3*has_text_early + 0.3*avg_saliency, 0.0, 1.0))

    return {
        "start_strength": start_strength,
        "t_first_action": t_first_action,
        "t_first_text": t_first_text,
        "opening_indices": opening_indices,
    }
