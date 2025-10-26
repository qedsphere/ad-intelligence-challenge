from typing import List, Dict, Any
import numpy as np
import cv2
from .common import Frame


def _aspect_ratio_class(width: int, height: int) -> str:
    r = float(width) / float(max(1, height))
    if r < 0.9:
        return "portrait"
    if r > 1.2:
        return "landscape"
    return "square"


def _letterbox_ratio(img_bgr, thresh_dark: int = 20) -> float:
    h, w = img_bgr.shape[:2]
    row_top = img_bgr[0: max(1, h // 50), :]
    row_bottom = img_bgr[h - max(1, h // 50):, :]
    col_left = img_bgr[:, 0: max(1, w // 50)]
    col_right = img_bgr[:, w - max(1, w // 50):]
    def dark_ratio(block):
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        return float((gray < thresh_dark).mean())
    return float(np.mean([dark_ratio(row_top), dark_ratio(row_bottom), dark_ratio(col_left), dark_ratio(col_right)]))


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"aspect_ratio_class": "unknown", "letterbox_ratio": 0.0}
    h, w = frames[0].img.shape[:2]
    return {
        "aspect_ratio_class": _aspect_ratio_class(w, h),
        "letterbox_ratio": _letterbox_ratio(frames[0].img),
    }
