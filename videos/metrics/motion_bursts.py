from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from .common import Frame, get_gray

FLOW_DOWNSCALE = 224
FLOW_STRIDE = 3


def _flow_magnitude(prev_bgr, curr_bgr) -> Tuple[float, float, float]:
    a = get_gray(prev_bgr)
    b = get_gray(curr_bgr)
    if a.shape[1] > FLOW_DOWNSCALE:
        s = FLOW_DOWNSCALE / a.shape[1]
        a = cv2.resize(a, (int(a.shape[1]*s), int(a.shape[0]*s)), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (int(b.shape[1]*s), int(b.shape[0]*s)), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = float(np.mean(mag))
    mean_dx = float(np.mean(flow[...,0]))
    mean_dy = float(np.mean(flow[...,1]))
    return mean_mag, mean_dx, mean_dy


def _classify_camera_motion(mean_dx: float, mean_dy: float, mag_thresh: float = 0.2) -> str:
    if abs(mean_dx) + abs(mean_dy) < mag_thresh:
        return "static"
    if abs(mean_dx) > 2 * abs(mean_dy):
        return "pan"
    if abs(mean_dy) > 2 * abs(mean_dx):
        return "tilt"
    return "mixed"


def compute(frames: List[Frame]) -> Dict[str, Any]:
    if len(frames) < 2:
        return {"motion_intensity_mean": 0.0, "motion_intensity_peak": 0.0, "burst_timestamps": [], "camera_motion_type": "static"}
    series = []  # (t, mag, dx, dy)
    step = max(1, FLOW_STRIDE)
    for i in range(step, len(frames), step):
        m, dx, dy = _flow_magnitude(frames[i-step].img, frames[i].img)
        t = 0.5 * (frames[i-step].t + frames[i].t)
        series.append((t, m, dx, dy))
    if not series:
        return {"motion_intensity_mean": 0.0, "motion_intensity_peak": 0.0, "burst_timestamps": [], "camera_motion_type": "static"}
    mags = [m for (_, m, _, _) in series]
    mean_mag = float(np.mean(mags))
    peak_mag = float(np.max(mags))
    # pick top 2 peaks
    top = sorted(series, key=lambda x: x[1], reverse=True)[:2]
    bursts = [t for (t, _, _, _) in top]
    mean_dx = float(np.mean([dx for (_, _, dx, _) in series]))
    mean_dy = float(np.mean([dy for (_, _, _, dy) in series]))
    cam = _classify_camera_motion(mean_dx, mean_dy)
    return {
        "motion_intensity_mean": mean_mag,
        "motion_intensity_peak": peak_mag,
        "burst_timestamps": bursts,
        "camera_motion_type": cam,
    }
