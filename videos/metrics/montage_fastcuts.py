from typing import List, Dict, Any, Tuple

from .common import Frame

WINDOW_S = 1.0


def _cut_times(segments: List[Tuple[int, int]], frames: List[Frame]) -> List[float]:
    times = []
    for (s, e) in segments:
        if 0 <= s < len(frames):
            times.append(frames[s].t)
    return times


def _peak_density(cut_ts: List[float], window_s: float) -> Tuple[float, Tuple[float, float]]:
    if not cut_ts:
        return 0.0, (0.0, 0.0)
    cut_ts = sorted(cut_ts)
    best = (0.0, 0.0, 0.0)
    for t in cut_ts:
        start = t
        end = t + window_s
        count = sum(1 for ct in cut_ts if start <= ct <= end)
        density = count / window_s
        if density > best[0]:
            best = (density, start, end)
    return best[0], (best[1], best[2])


def compute(frames: List[Frame], deps: Dict[str, Any]) -> Dict[str, Any]:
    segs = deps.get("shot_detection", {}).get("shot_segments", [])
    cut_ts = _cut_times(segs, frames)
    peak_density, montage_window = _peak_density(cut_ts, WINDOW_S)
    return {
        "peak_cut_density": float(peak_density),
        "montage_window": montage_window,
        "montage_snippet": montage_window,
    }
