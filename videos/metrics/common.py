import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class Frame:
    t: float
    img: np.ndarray  # BGR


def resize_to_short_edge(img: np.ndarray, se: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == se:
        return img
    if h < w:
        new_h = se
        new_w = int(w * (se / h))
    else:
        new_w = se
        new_h = int(h * (se / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_video_frames(path: Path, target_fps: float, short_edge: int, max_frames: Optional[int]) -> List[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / target_fps)))
    frames: List[Frame] = []
    idx = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = resize_to_short_edge(frame, short_edge)
            frames.append(Frame(t=t, img=frame))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

# Simple per-process gray cache
_GRAY_CACHE = {}

def get_gray(img_bgr: np.ndarray) -> np.ndarray:
    key = id(img_bgr)
    if key in _GRAY_CACHE:
        return _GRAY_CACHE[key]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _GRAY_CACHE[key] = gray
    return gray


def reset_gray_cache():
    _GRAY_CACHE.clear()
