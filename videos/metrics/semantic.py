from typing import List, Dict, Any
import numpy as np
from .common import Frame
import cv2


def _edge_clutter(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean()) / 255.0


def _scene_simplicity(clutter_values: List[float]) -> float:
    if not clutter_values:
        return 0.0
    mean_c = float(np.mean(clutter_values))
    std_c = float(np.std(clutter_values))
    return float(np.clip(1.0 - (0.6 * mean_c + 0.4 * std_c), 0.0, 1.0))


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if not frames:
        return {"simplicity": 0.0}
    step = max(1, len(frames)//6 or 1)
    samples = [frames[i].img for i in range(0, len(frames), step)[:6]]
    clutter_vals = [_edge_clutter(img) for img in samples] if samples else []
    simplicity = _scene_simplicity(clutter_vals)
    return {"simplicity": float(simplicity)}
