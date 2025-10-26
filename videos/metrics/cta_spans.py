from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame, get_gray

COVERAGE_THRESH = 0.01
MSER_MIN_AREA = 60


def _text_cov_contrast(img_bgr) -> Tuple[float, float]:
    gray = get_gray(img_bgr)
    mser = cv2.MSER_create(_min_area=MSER_MIN_AREA) if hasattr(cv2.MSER_create, "__call__") else cv2.MSER_create()
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


def compute(frames: List[Frame], deps: Dict[str, Any] = None) -> Dict[str, Any]:
    timeline = []  # (t, cov, contrast)
    for f in frames:
        cov, contr = _text_cov_contrast(f.img)
        timeline.append((f.t, cov, contr))
    spans = []
    in_span = False
    start_t = 0.0
    for i, (t, cov, _) in enumerate(timeline):
        if cov >= COVERAGE_THRESH and not in_span:
            in_span = True
            start_t = t
        if cov < COVERAGE_THRESH and in_span:
            in_span = False
            spans.append((start_t, timeline[i-1][0]))
    if in_span and timeline:
        spans.append((start_t, timeline[-1][0]))

    present = len(spans) > 0
    t_first = spans[0][0] if present else None
    dwell = sum(max(0.0, e - s) for (s, e) in spans)
    longest = max((e - s) for (s, e) in spans) if present else 0.0
    avg_contrast = float(np.mean([c for (_, _, c) in timeline])) if timeline else 0.0

    return {
        "cta_timeline": timeline,
        "cta_spans": spans,
        "cta_present": present,
        "t_first_cta": t_first,
        "cta_dwell": dwell,
        "cta_longest": longest,
        "cta_contrast": avg_contrast,
    }
