from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from .common import Frame, get_gray

# Tunables
HIST_DOWNSCALE = 80
SSIM_DOWNSCALE = 112
KEY_W_SHARP = 0.7
KEY_W_SAL = 0.3
SHOT_CHECK_STRIDE = 2


def frame_sharpness(img):
    gray = get_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_saliency_score(img):
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        w = img.shape[1]
        if w > 192:
            scale = 192 / w
            img = cv2.resize(img, (int(w * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, smap = sal.computeSaliency(img)
        if ok:
            return float(np.mean(smap))
    except Exception:
        pass
    return 0.0


def hist_distance(a_bgr, b_bgr) -> float:
    if HIST_DOWNSCALE:
        a_bgr = cv2.resize(a_bgr, (HIST_DOWNSCALE, HIST_DOWNSCALE), interpolation=cv2.INTER_AREA)
        b_bgr = cv2.resize(b_bgr, (HIST_DOWNSCALE, HIST_DOWNSCALE), interpolation=cv2.INTER_AREA)
    a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([a_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    hist_b = cv2.calcHist([b_hsv], [0,1,2], None, [32,32,32], [0,180,0,256,0,256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def frame_ssim(a_bgr, b_bgr) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        if SSIM_DOWNSCALE:
            a_bgr = cv2.resize(a_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
            b_bgr = cv2.resize(b_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
        a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        return float(ssim(a_gray, b_gray))
    except Exception:
        return 0.0


def detect_shots(frames: List[Frame], thresh_hist: float = 0.3, thresh_ssim_delta: float = 0.35) -> List[Tuple[int,int]]:
    n = len(frames)
    if n == 0:
        return []
    cuts = [0]
    i = 1
    while i < n:
        prev = frames[i-1].img
        curr = frames[i].img
        hdist = hist_distance(prev, curr)
        ssim_val = frame_ssim(prev, curr)
        if hdist > thresh_hist or (1.0 - ssim_val) > thresh_ssim_delta:
            cuts.append(i)
        i += SHOT_CHECK_STRIDE
    if cuts[-1] != n:
        cuts.append(n)
    segs = []
    for j in range(len(cuts)-1):
        s = cuts[j]
        e = cuts[j+1]-1
        if e >= s:
            segs.append((s,e))
    return segs


def select_keyframe(frames: List[Frame], s: int, e: int) -> int:
    best_idx = s
    best_score = -1.0
    for i in range(s, e+1):
        img = frames[i].img
        score = KEY_W_SHARP * frame_sharpness(img) + KEY_W_SAL * frame_saliency_score(img)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def compute(frames: List[Frame]) -> Dict[str, Any]:
    segs = detect_shots(frames)
    keys = [select_keyframe(frames, s, e) for (s,e) in segs]
    return {
        "shot_segments": segs,
        "keyframe_indices": keys,
        "num_shots": len(segs),
    }
