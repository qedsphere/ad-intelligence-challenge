# %% [markdown]
# # Visual Feature Extraction Playground (Videos)
# 
# This notebook loads ad videos, lets you select one by index or at random, and provides placeholders for step-by-step visual analysis (no audio):
# 
# %%
# Imports & setup
import os
import random
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path("/Users/elshroomster/Desktop/Calhacks x Applovin -- Ad Intelligence")
VIDEOS_DIR = PROJECT_ROOT / "ads" / "videos"

# Discover dataset
video_exts = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
VIDEO_PATHS = sorted([p for p in VIDEOS_DIR.glob("*.*") if p.suffix.lower() in video_exts])
print(f"Found {len(VIDEO_PATHS)} videos in {VIDEOS_DIR}")

# Matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = False


# caching stuff to avoid recomputing things
TEXT_COVERAGE_CACHE = None
TEXT_CONTRAST_CACHE = None
SAL_MEAN_CACHE = {}
EDGE_CLUTTER_CACHE = {}
CENTER_SAL_CACHE = {}
GRAY_CACHE = {}
SHARPNESS_CACHE = {}
TEXT_CC_BY_ID = {}

def reset_caches(num_frames: int = 0):
    global TEXT_COVERAGE_CACHE, TEXT_CONTRAST_CACHE, SAL_MEAN_CACHE, EDGE_CLUTTER_CACHE, CENTER_SAL_CACHE, GRAY_CACHE, SHARPNESS_CACHE, TEXT_CC_BY_ID
    TEXT_COVERAGE_CACHE = [None] * num_frames
    TEXT_CONTRAST_CACHE = [None] * num_frames
    SAL_MEAN_CACHE = {}
    EDGE_CLUTTER_CACHE = {}
    CENTER_SAL_CACHE = {}
    GRAY_CACHE = {}
    SHARPNESS_CACHE = {}
    TEXT_CC_BY_ID = {}

# speedup settings - downsample heavy operations
FAST_MODE = True
SALIENCY_DOWNSCALE = 192
MSER_DOWNSCALE = 384
HIST_DOWNSCALE = 80
SSIM_DOWNSCALE = 112
FLOW_DOWNSCALE = 224
FLOW_STRIDE = 3
TEXT_SAMPLE_STEP = 3
# New knobs
SHOT_CHECK_STRIDE = 2  # stride inside shot boundary scan when FAST_MODE
AVG_SALIENCY_SAMPLES = 8  # opening saliency avg samples


def get_gray(img_bgr: np.ndarray) -> np.ndarray:
    key = id(img_bgr)
    if key in GRAY_CACHE:
        return GRAY_CACHE[key]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    GRAY_CACHE[key] = gray
    return gray

# %%
# Video selection utilities

@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def probe_video(path: Path) -> Optional[VideoMeta]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Failed to open: {path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return VideoMeta(path, fps, frame_count, duration_s, width, height)


def pick_video(index: Optional[int] = None, seed: Optional[int] = None) -> VideoMeta:
    """
    Pick a video by index or randomly.
    
    Args:
        index: Specific video index (0-based). If None, picks randomly.
        seed: Random seed for reproducibility when index is None.
    
    Returns:
        VideoMeta with video information
    """
    if not VIDEO_PATHS:
        raise RuntimeError("No videos found")
    if index is None:
        rng = random.Random(seed)
        p = rng.choice(VIDEO_PATHS)
    else:
        p = VIDEO_PATHS[index % len(VIDEO_PATHS)]
    meta = probe_video(p)
    if not meta:
        raise RuntimeError(f"Could not probe video: {p}")
    print(f"Selected: {meta.path.name}")
    print(f"  Resolution: {meta.width}x{meta.height}")
    print(f"  FPS: {meta.fps:.2f}")
    print(f"  Duration: {meta.duration_s:.2f}s")
    print(f"  Frames: {meta.frame_count}")
    return meta

# (Selection happens in Step 0 below)


# %%
# Video loader with fps resample & resize

@dataclass
class Frame:
    t: float  # timestamp in seconds
    img: np.ndarray  # BGR image


def load_video_frames(path: Path,
                      target_fps: float = 12.0,
                      short_edge: int = 576,
                      max_frames: Optional[int] = 500) -> List[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {path}")
    
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    stride = max(1, int(round(src_fps / target_fps)))
    frames: List[Frame] = []

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

    idx = 0
    est_total = frame_count if frame_count > 0 else int(src_fps * 60)
    est_used = min(est_total, (max_frames or est_total))
    reset_caches(num_frames=int(est_used // max(1, stride)) + 2)

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
    print(f"Loaded {len(frames)} frames at ~{target_fps}fps (stride={stride})")
    return frames

# (Loading happens in Step 0 below)

# %%
# Visualization helpers

def show_frame(img_bgr: np.ndarray, title: Optional[str] = None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def show_frames_grid(frames_list: List[np.ndarray], cols: int = 4, titles: Optional[List[str]] = None):
    n = len(frames_list)
    if n == 0:
        print("No frames to display")
        return
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, img in enumerate(frames_list):
        plt.subplot(rows, cols, i + 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        if titles and i < len(titles):
            plt.title(titles[i], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def extract_clip(frames_seq: List[Frame], start_s: float, end_s: float) -> List[Frame]:
    return [f for f in frames_seq if start_s <= f.t <= end_s]

# (Preview happens in Step 0 below)

# %%
# %% [markdown]
# ## 0) Load and Normalize (Selection + Sampling)
# - Params: INDEX, TARGET_FPS, SHORT_EDGE, MAX_FRAMES
# - Preview: grid of first N frames

# Configure Step 0
INDEX = 19
SEED = 42
TARGET_FPS = 12.0
SHORT_EDGE = 576
MAX_FRAMES = 300
PREVIEW_COUNT = 8

# Select video
SELECTED = pick_video(index=INDEX, seed=SEED)

# Load frames
frames = load_video_frames(
    SELECTED.path,
    target_fps=TARGET_FPS,
    short_edge=SHORT_EDGE,
    max_frames=MAX_FRAMES,
)

# Summarize
if frames:
    print(
        f"Video: {SELECTED.path.name} | {SELECTED.width}x{SELECTED.height} @ {SELECTED.fps:.2f}fps | "
        f"Loaded: {len(frames)} frames | {frames[0].t:.2f}s-{frames[-1].t:.2f}s"
    )
else:
    print("No frames loaded")

# Preview
if frames:
    preview_imgs = [f.img for f in frames[:min(PREVIEW_COUNT, len(frames))]]
    preview_titles = [f"t={frames[i].t:.2f}s" for i in range(len(preview_imgs))]
    show_frames_grid(preview_imgs, cols=4, titles=preview_titles)


# %% [markdown]
# ## 1) Shot Detection and Keyframe Selection
# 
# %%
# Step 1: Shot detection

def frame_sharpness(img: np.ndarray) -> float:
    key = id(img)
    if key in SHARPNESS_CACHE:
        return SHARPNESS_CACHE[key]
    gray = get_gray(img)
    val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    SHARPNESS_CACHE[key] = val
    return val


def frame_saliency_score(img: np.ndarray) -> float:
    # cache saliency computation since it's slow
    key = id(img)
    if key in SAL_MEAN_CACHE:
        return SAL_MEAN_CACHE[key]
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        if FAST_MODE and SALIENCY_DOWNSCALE and img.shape[1] > SALIENCY_DOWNSCALE:
            scale = SALIENCY_DOWNSCALE / img.shape[1]
            small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            ok, sal_map = saliency.computeSaliency(small)
        else:
            ok, sal_map = saliency.computeSaliency(img)
        if ok:
            val = float(np.mean(sal_map))
            SAL_MEAN_CACHE[key] = val
            return val
    except Exception:
        pass
    SAL_MEAN_CACHE[key] = 0.0
    return 0.0


def hist_distance_bhattacharyya(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    # downsample before comparing
    if FAST_MODE and HIST_DOWNSCALE:
        a_bgr = cv2.resize(a_bgr, (HIST_DOWNSCALE, HIST_DOWNSCALE), interpolation=cv2.INTER_AREA)
        b_bgr = cv2.resize(b_bgr, (HIST_DOWNSCALE, HIST_DOWNSCALE), interpolation=cv2.INTER_AREA)
    a_hsv = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    bins = (32, 32, 32)
    ranges = [0, 180, 0, 256, 0, 256]
    hist_a = cv2.calcHist([a_hsv], [0, 1, 2], None, bins, ranges)
    hist_b = cv2.calcHist([b_hsv], [0, 1, 2], None, bins, ranges)
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def frame_ssim(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        # downsample for speed
        if FAST_MODE and SSIM_DOWNSCALE:
            a_bgr = cv2.resize(a_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
            b_bgr = cv2.resize(b_bgr, (SSIM_DOWNSCALE, SSIM_DOWNSCALE), interpolation=cv2.INTER_AREA)
        a_gray = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
        score = ssim(a_gray, b_gray)
        return float(score)
    except Exception:
        return 0.0


def detect_shots(frames_seq: List[Frame],
                 thresh_hist: float = 0.3,
                 thresh_ssim_delta: float = 0.35,
                 min_shot_len: int = 3) -> List[Tuple[int, int]]:
    n = len(frames_seq)
    if n == 0:
        return []
    cut_indices = [0]
    i = 1
    while i < n:
        prev = frames_seq[i - 1].img
        curr = frames_seq[i].img
        hdist = hist_distance_bhattacharyya(prev, curr)
        ssim_val = frame_ssim(prev, curr)
        ssim_delta = 1.0 - ssim_val
        if hdist > thresh_hist or ssim_delta > thresh_ssim_delta:
            cut_indices.append(i)
        i += (SHOT_CHECK_STRIDE if FAST_MODE else 1)
    if cut_indices[-1] != n:
        cut_indices.append(n)
    segments: List[Tuple[int, int]] = []
    for i in range(len(cut_indices) - 1):
        s = cut_indices[i]
        e = cut_indices[i + 1] - 1
        if e >= s:
            segments.append((s, e))
    # merge short segments
    merged: List[Tuple[int, int]] = []
    for seg in segments:
        if (seg[1] - seg[0] + 1) < min_shot_len and merged:
            prev_s, prev_e = merged[-1]
            merged[-1] = (prev_s, seg[1])
        else:
            merged.append(seg)
    return merged


def select_keyframe(frames_seq: List[Frame], start: int, end: int,
                    w_sharp: float = 0.7, w_sal: float = 0.3) -> int:
    best_idx = start
    best_score = -1.0
    for i in range(start, end + 1):
        img = frames_seq[i].img
        score = w_sharp * frame_sharpness(img) + w_sal * frame_saliency_score(img)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# Execute Step 1
shot_segments: List[Tuple[int, int]] = detect_shots(frames)
keyframe_indices: List[int] = [select_keyframe(frames, s, e) for (s, e) in shot_segments]
keyframe_frames: List[np.ndarray] = [frames[i].img for i in keyframe_indices]

print(f"Found {len(shot_segments)} shots, showing {len(keyframe_frames)} keyframes")
if keyframe_frames:
    titles = [f"t={frames[i].t:.2f}s" for i in keyframe_indices]
    show_frames_grid(keyframe_frames, cols=4, titles=titles)


# %% [markdown]
# ## 2) First 3 Seconds Focus
# 

# %%
# Step 2: First 3 seconds

def slice_frames_by_time(frames_seq: List[Frame], start_s: float, end_s: float) -> List[Frame]:
    return [f for f in frames_seq if start_s <= f.t <= end_s]


def text_presence_score(img_bgr: np.ndarray) -> float:
    # quick text detection using MSER
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
    area_ratio = min(1.0, area / float(h * w))
    score = 0.6 * area_ratio + 0.4 * (np.log1p(count) / 6.0)
    return float(max(0.0, min(1.0, score)))


def get_text_cov_contrast(img_bgr: np.ndarray) -> Tuple[float, float]:
    key = id(img_bgr)
    if key in TEXT_CC_BY_ID:
        return TEXT_CC_BY_ID[key]
    gray = get_gray(img_bgr)
    h, w = gray.shape[:2]
    mser = cv2.MSER_create(min_area=60)
    regions, _ = mser.detectRegions(gray)
    area = 0
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 60:
            continue
        area += cw * ch
    coverage = float(area) / float(max(1, h * w))
    contrast = float(np.std(gray) / 255.0)
    val = (coverage, contrast)
    TEXT_CC_BY_ID[key] = val
    return val


def motion_diff_score(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    # fast motion via grayscale diff
    a = get_gray(prev_bgr)
    b = get_gray(curr_bgr)
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff) / 255.0)


def compute_first_action_time(frames_subset: List[Frame]) -> Optional[float]:
    if len(frames_subset) < 2:
        return None
    diffs = []
    for i in range(1, min(4, len(frames_subset))):
        diffs.append(motion_diff_score(frames_subset[i - 1].img, frames_subset[i].img))
    baseline = np.median(diffs) if diffs else 0.02
    threshold = baseline + 0.03
    for i in range(1, len(frames_subset)):
        s = motion_diff_score(frames_subset[i - 1].img, frames_subset[i].img)
        if s > threshold:
            return frames_subset[i].t
    return None


def rank_opening_frames(frames_subset: List[Frame], top_k: int = 3, text_cov_list: Optional[List[float]] = None) -> List[int]:
    scores = []
    for i, f in enumerate(frames_subset):
        sharp = frame_sharpness(f.img)
        sal = frame_saliency_score(f.img)
        cov = text_cov_list[i] if text_cov_list is not None and i < len(text_cov_list) else get_text_cov_contrast(f.img)[0]
        score = 0.5 * sharp + 0.3 * sal + 0.2 * (cov * 100.0)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [idx for (_, idx) in scores[:min(top_k, len(scores))]]


# Execute Step 2
opening_window = slice_frames_by_time(frames, 0.0, 3.0)
if not opening_window:
    print("No frames in first 3 seconds")
    start_strength = 0.0
    t_first_action = None
    t_first_text = None
    opening_frames = []
    opening_snippet = (0.0, 0.0)
else:
    t_first_action = compute_first_action_time(opening_window)
    t_first_text = None
    text_covs_opening = [get_text_cov_contrast(f.img)[0] for f in opening_window]
    for idx, f in enumerate(opening_window):
        if text_covs_opening[idx] > 0.02:
            t_first_text = f.t
            break

    top_indices_rel = rank_opening_frames(opening_window, top_k=3, text_cov_list=text_covs_opening)
    opening_frames = [opening_window[i] for i in top_indices_rel]

    has_action_early = 1.0 if (t_first_action is not None and t_first_action <= 2.0) else 0.0
    has_text_early = 1.0 if (t_first_text is not None and t_first_text <= 2.0) else 0.0
    avg_saliency = float(np.mean([
        frame_saliency_score(f.img) for f in opening_window[:min(AVG_SALIENCY_SAMPLES, len(opening_window))]
    ])) if opening_window else 0.0
    start_strength = float(np.clip(0.4 * has_action_early + 0.3 * has_text_early + 0.3 * avg_saliency, 0.0, 1.0))

    center_t = opening_frames[0].t if opening_frames else 1.5
    center_t = float(np.clip(center_t, 0.75, 2.25))
    opening_snippet = (max(0.0, center_t - 0.75), min(3.0, center_t + 0.75))

    if opening_frames:
        show_frames_grid([f.img for f in opening_frames], cols=3,
                         titles=[f"t={f.t:.2f}s" for f in opening_frames])
    print(
        f"Start strength={start_strength:.2f} | first_action={t_first_action} | first_text={t_first_text} | "
        f"snippet={opening_snippet[0]:.2f}-{opening_snippet[1]:.2f}s"
    )


# %% [markdown]
# ## 3) CTA Text Spans
# 

# %%
# Step 3: CTA spans using MSER

def detect_text_mser_stats(img_bgr: np.ndarray):
    img = img_bgr
    if FAST_MODE and MSER_DOWNSCALE and img.shape[1] > MSER_DOWNSCALE:
        scale = MSER_DOWNSCALE / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    gray = get_gray(img)
    h, w = gray.shape[:2]
    mser = cv2.MSER_create(min_area=60)
    regions, _ = mser.detectRegions(gray)
    area = 0
    boxes = []
    for cnt in regions:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < 60:
            continue
        boxes.append((x, y, cw, ch))
        area += cw * ch
    coverage = float(area) / float(h * w)
    contrast = float(np.std(gray) / 255.0)
    return coverage, boxes, contrast


def compute_cta_spans(frames_seq: List[Frame], coverage_thresh: float = 0.01):
    timeline = []
    for idx, f in enumerate(frames_seq):
        if TEXT_COVERAGE_CACHE is not None and idx < len(TEXT_COVERAGE_CACHE) and TEXT_COVERAGE_CACHE[idx] is not None:
            cov = TEXT_COVERAGE_CACHE[idx]
            contr = TEXT_CONTRAST_CACHE[idx]
        else:
            cov, _, contr = detect_text_mser_stats(f.img)
            if TEXT_COVERAGE_CACHE is not None and idx < len(TEXT_COVERAGE_CACHE):
                TEXT_COVERAGE_CACHE[idx] = cov
                TEXT_CONTRAST_CACHE[idx] = contr
        timeline.append((f.t, cov, contr))
    spans = []
    in_span = False
    span_start_t = 0.0
    for i, (t, cov, _) in enumerate(timeline):
        if cov >= coverage_thresh and not in_span:
            in_span = True
            span_start_t = t
        if cov < coverage_thresh and in_span:
            in_span = False
            spans.append((span_start_t, timeline[i - 1][0]))
    if in_span:
        spans.append((span_start_t, timeline[-1][0]))
    return timeline, spans


# Execute Step 3
cta_timeline, cta_spans = compute_cta_spans(frames, coverage_thresh=0.01)
cta_present = len(cta_spans) > 0
t_first_cta = cta_spans[0][0] if cta_present else None
cta_dwell = sum(max(0.0, e - s) for (s, e) in cta_spans)
cta_longest = max((e - s) for (s, e) in cta_spans) if cta_present else 0.0
cta_contrast = float(np.mean([c for (_, _, c) in cta_timeline])) if cta_timeline else 0.0

cta_snippets = []
if cta_present:
    first_s, first_e = cta_spans[0]
    long_s, long_e = max(cta_spans, key=lambda se: se[1] - se[0])
    cta_snippets.append((max(0.0, first_s - 0.75), first_e + 0.75))
    cta_snippets.append((max(0.0, long_s - 0.75), long_e + 0.75))

print(
    f"CTA present={cta_present} | t_first_cta={t_first_cta} | dwell={cta_dwell:.2f}s | longest={cta_longest:.2f}s | contrast={cta_contrast:.2f}"
)


# %% [markdown]
# ## 4) Logo/Product Reveal
# 

# %%
# Step 4: Product/Logo reveals using saliency

def salient_area_ratio(img_bgr: np.ndarray, thresh: float = 0.6) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        img = img_bgr
        if FAST_MODE and SALIENCY_DOWNSCALE and img.shape[1] > SALIENCY_DOWNSCALE:
            scale = SALIENCY_DOWNSCALE / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        ok, smap = sal.computeSaliency(img)
        if not ok:
            return 0.0
        t = np.quantile(smap, thresh)
        return float((smap >= t).mean())
    except Exception:
        return 0.0


def compute_product_reveals(frames_seq: List[Frame], area_thresh: float = 0.12):
    ratios = [(f.t, salient_area_ratio(f.img)) for f in frames_seq]
    t_first = None
    dwell = 0.0
    prev_above = False
    prev_t = ratios[0][0] if ratios else 0.0
    max_ratio = 0.0
    t_at_max = None
    for (t, r) in ratios:
        if r > max_ratio:
            max_ratio = r
            t_at_max = t
        above = r >= area_thresh
        if above and t_first is None:
            t_first = t
        if prev_above:
            dwell += max(0.0, t - prev_t)
        prev_above = above
        prev_t = t
    return t_first, t_at_max, max_ratio, dwell


t_first_product, t_peak_product, max_product_area_ratio, product_dwell = compute_product_reveals(frames)
t_first_logo = None  # no logo detector in fallback
reveal_snippets = []
if t_first_product is not None:
    reveal_snippets.append((max(0.0, t_first_product - 1.0), t_first_product + 1.0))
if t_peak_product is not None:
    reveal_snippets.append((max(0.0, t_peak_product - 1.0), t_peak_product + 1.0))

print(
    f"Product reveal t_first={t_first_product} | t_peak={t_peak_product} | max_area_ratio={max_product_area_ratio:.3f} | dwell={product_dwell:.2f}s"
)


# %% [markdown]
# ## 5) High-Motion Burst Detection
# 

# %%
# Step 5: Motion bursts

def flow_magnitude(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    a_full = get_gray(prev_bgr)
    b_full = get_gray(curr_bgr)
    if FAST_MODE and FLOW_DOWNSCALE and a_full.shape[1] > FLOW_DOWNSCALE:
        scale = FLOW_DOWNSCALE / a_full.shape[1]
        a = cv2.resize(a_full, (int(a_full.shape[1]*scale), int(a_full.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b_full, (int(b_full.shape[1]*scale), int(b_full.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    else:
        a, b = a_full, b_full
    # Smaller window/levels for speed
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))
    mean_dx = float(np.mean(flow[..., 0]))
    mean_dy = float(np.mean(flow[..., 1]))
    return mean_mag, (mean_dx, mean_dy)


def classify_camera_motion(mean_dx: float, mean_dy: float, mag_thresh: float = 0.2) -> str:
    if abs(mean_dx) + abs(mean_dy) < mag_thresh:
        return "static"
    if abs(mean_dx) > 2 * abs(mean_dy):
        return "pan"
    if abs(mean_dy) > 2 * abs(mean_dx):
        return "tilt"
    return "mixed"


def compute_flow_series(frames_seq: List[Frame]):
    series = []
    step = FLOW_STRIDE if FAST_MODE else 1
    for i in range(step, len(frames_seq), step):
        mag, (dx, dy) = flow_magnitude(frames_seq[i - step].img, frames_seq[i].img)
        t = 0.5 * (frames_seq[i - 1].t + frames_seq[i].t)
        series.append((t, mag, dx, dy))
    return series


def top_motion_bursts(flow_series: List[Tuple[float, float, float, float]], window_s: float = 1.0, top_n: int = 2):
    if not flow_series:
        return [], 0.0, 0.0, "static"
    mags = np.array([m for (_, m, _, _) in flow_series], dtype=np.float32)
    ts = np.array([t for (t, _, _, _) in flow_series], dtype=np.float32)
    # rolling energy
    peaks = []
    for i in range(len(flow_series)):
        t0 = ts[i]
        mask = (ts >= t0) & (ts <= t0 + window_s)
        energy = float(np.mean(mags[mask])) if np.any(mask) else 0.0
        peaks.append((energy, t0))
    peaks.sort(reverse=True)
    top = peaks[:top_n]
    mean_mag = float(np.mean(mags))
    peak_mag = float(top[0][0]) if top else 0.0
    # camera motion from global averages
    mean_dx = float(np.mean([dx for (_, _, dx, _) in flow_series]))
    mean_dy = float(np.mean([dy for (_, _, _, dy) in flow_series]))
    cam = classify_camera_motion(mean_dx, mean_dy)
    return top, mean_mag, peak_mag, cam


flow_series = compute_flow_series(frames)
bursts, motion_intensity_mean, motion_intensity_peak, camera_motion_type = top_motion_bursts(flow_series)
burst_timestamps = [t for (_, t) in bursts]
burst_clips = [(max(0.0, t - 0.5), t + 0.5) for t in burst_timestamps]

print(
    f"Motion mean={motion_intensity_mean:.3f} | peak={motion_intensity_peak:.3f} | bursts={burst_timestamps} | camera={camera_motion_type}"
)


# %% [markdown]
# ## 6) Text-Dense Frames
# 

# %%
# Step 6: Text-dense frames

def rank_text_dense_frames(frames_seq: List[Frame], top_k: int = 3):
    stats = []
    for i, f in enumerate(frames_seq):
        cov, _, contr = detect_text_mser_stats(f.img)
        stats.append((cov, contr, i))
    stats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = stats[:min(top_k, len(stats))]
    return top


top_text = rank_text_dense_frames(frames, top_k=3)
top_text_frames = [frames[i].img for (_, _, i) in top_text]
text_coverage_stats = [(frames[i].t, cov, contr) for (cov, contr, i) in top_text]
if top_text_frames:
    show_frames_grid(top_text_frames, cols=3,
                     titles=[f"t={frames[i].t:.2f}s cov={cov:.3f}" for (cov, _, i) in top_text])

print("Top text frames:", text_coverage_stats)


# %% [markdown]
# ## 7) End-Card Extraction
# 

# %%
# Step 7: End-card

def last_seconds(frames_seq: List[Frame], seconds: float = 3.0) -> List[Frame]:
    if not frames_seq:
        return []
    end_t = frames_seq[-1].t
    start_t = max(0.0, end_t - seconds)
    return [f for f in frames_seq if f.t >= start_t]


def staticity_series(frames_subset: List[Frame]):
    diffs = []
    for i in range(1, len(frames_subset)):
        a = cv2.cvtColor(frames_subset[i - 1].img, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames_subset[i].img, cv2.COLOR_BGR2GRAY)
        diffs.append((frames_subset[i].t, float(np.mean(cv2.absdiff(a, b)) / 255.0)))
    return diffs


def find_endcard(frames_seq: List[Frame]):
    subset = last_seconds(frames_seq, 3.0)
    if len(subset) < 2:
        return False, 0.0, None, 0.0
    diffs = staticity_series(subset)
    if not diffs:
        return False, 0.0, None, 0.0
    # staticity is inverse of diff; pick min-diff frame
    min_idx = int(np.argmin([d for (_, d) in diffs])) + 1
    best_frame = subset[min_idx]
    # duration: consecutive low-diff region (< threshold)
    thr = float(np.median([d for (_, d) in diffs]) * 0.8 + 0.02)
    start_t = best_frame.t
    end_t = best_frame.t
    # expand backward
    for i in range(min_idx, 0, -1):
        if diffs[i - 1][1] < thr:
            start_t = subset[i - 1].t
        else:
            break
    # forward
    for i in range(min_idx + 1, len(subset)):
        if diffs[i - 1][1] < thr:
            end_t = subset[i].t
        else:
            break
    endcard_readability = text_presence_score(best_frame.img)
    return True, max(0.0, end_t - start_t), best_frame, endcard_readability


endcard_present, endcard_duration, endcard_frame, endcard_readability = find_endcard(frames)
if endcard_present and endcard_frame is not None:
    show_frame(endcard_frame.img, title=f"End-card t={endcard_frame.t:.2f}s")
print(
    f"End-card present={endcard_present} | duration={endcard_duration:.2f}s | readability={endcard_readability:.2f}"
)


# %% [markdown]
# ## 8) Thumbnail Candidate Ranking
# 

# %%
# Step 8: Thumbnails

def brightness_score(img_bgr: np.ndarray) -> float:
    gray = get_gray(img_bgr)
    return float(np.std(gray) / 128.0)


def thumbnail_score(img_bgr: np.ndarray) -> float:
    sharp = frame_sharpness(img_bgr)
    sal = frame_saliency_score(img_bgr)
    txt = text_presence_score(img_bgr)
    br = brightness_score(img_bgr)
    return 0.35 * sharp + 0.25 * sal + 0.20 * (txt * 100.0) + 0.20 * br


def collect_candidates() -> List[Tuple[float, float, np.ndarray]]:
    cands = []
    for idx in keyframe_indices:
        img = frames[idx].img
        t = frames[idx].t
        cands.append((thumbnail_score(img), t, img))
    if endcard_present and endcard_frame is not None:
        cands.append((thumbnail_score(endcard_frame.img), endcard_frame.t, endcard_frame.img))
    try:
        for of in opening_frames:
            cands.append((thumbnail_score(of.img), of.t, of.img))
    except Exception:
        pass
    for (_, _, i) in top_text:
        img = frames[i].img
        t = frames[i].t
        cands.append((thumbnail_score(img), t, img))
    return cands


thumb_candidates = collect_candidates()
thumb_candidates.sort(key=lambda x: x[0], reverse=True)
top_thumbs = thumb_candidates[:min(3, len(thumb_candidates))]
if top_thumbs:
    show_frames_grid([img for (_, _, img) in top_thumbs], cols=3,
                     titles=[f"t={t:.2f}s score={s:.1f}" for (s, t, _) in top_thumbs])
print("Top thumbnail timestamps:", [(t, s) for (s, t, _) in top_thumbs])


# %% [markdown]
# ## 9) Montage/Fast-Cut Section
# 

# %%
# Step 9: Montage

def shot_cut_times(segments: List[Tuple[int, int]], frames_seq: List[Frame]) -> List[float]:
    times = []
    for (s, e) in segments:
        times.append(frames_seq[s].t)
    return times


def peak_cut_density(cut_ts: List[float], window_s: float = 1.0):
    if not cut_ts:
        return 0.0, (0.0, 0.0)
    best = (0.0, 0.0, 0.0)  # (density, start, end)
    for t in cut_ts:
        start = t
        end = t + window_s
        density = sum(1 for ct in cut_ts if start <= ct <= end) / window_s
        if density > best[0]:
            best = (density, start, end)
    return best[0], (best[1], best[2])


cut_ts = shot_cut_times(shot_segments, frames)
peak_density, montage_window = peak_cut_density(cut_ts, window_s=1.0)
montage_snippet = montage_window
print(f"Montage peak density={peak_density:.2f} cuts/s | window={montage_window}")


# %% [markdown]
# ## 10) Loop Readiness
# 

# %%
# Step 10: Loop readiness

def average_frame(frames_subset: List[Frame]) -> Optional[np.ndarray]:
    if not frames_subset:
        return None
    acc = None
    for f in frames_subset:
        img = f.img.astype(np.float32)
        if acc is None:
            acc = img
        else:
            acc += img
    acc /= float(len(frames_subset))
    return acc.astype(np.uint8)


def ssim_between(img_a: np.ndarray, img_b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b))


def hist_bhatt_between(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return hist_distance_bhattacharyya(img_a, img_b)


def window_frames(frames_seq: List[Frame], start_s: float, end_s: float) -> List[Frame]:
    return [f for f in frames_seq if start_s <= f.t <= end_s]


if frames:
    total_end = frames[-1].t
    first_win = window_frames(frames, 0.0, min(1.0, total_end))
    last_win = window_frames(frames, max(0.0, total_end - 1.0), total_end)
    avg_first = average_frame(first_win)
    avg_last = average_frame(last_win)
    if avg_first is not None and avg_last is not None:
        ssim_fl = ssim_between(avg_first, avg_last)
        hist_fl = hist_bhatt_between(avg_first, avg_last)
        loop_readiness_score = float(np.clip(0.7 * ssim_fl + 0.3 * (1.0 - hist_fl), 0.0, 1.0))
        print(
            f"Loop readiness={loop_readiness_score:.2f} | SSIM={ssim_fl:.2f} | 1-Bhatt={1.0 - hist_fl:.2f}"
        )
    else:
        print("Insufficient frames for loop readiness")
else:
    print("No frames for loop readiness")

# %%
# %% [markdown]
# ## 11) Faces and Gaze (lightweight)
# - Detect faces on sampled frames, compute first-appearance time and size prominence

# %%
# Step 11: Faces and Gaze (proxy)

def compute_faces_stats(frames_seq: List[Frame], step: int = 5):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    t_first_face = None
    max_face_area_ratio = 0.0
    face_frames = 0
    samples = 0
    for i in range(0, len(frames_seq), max(1, step)):
        samples += 1
        f = frames_seq[i]
        gray = cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces) > 0:
            face_frames += 1
            if t_first_face is None:
                t_first_face = f.t
            h, w = gray.shape[:2]
            for (x, y, fw, fh) in faces:
                area_ratio = (fw * fh) / float(w * h)
                if area_ratio > max_face_area_ratio:
                    max_face_area_ratio = area_ratio
    face_present = face_frames > 0
    face_time_ratio = float(face_frames) / float(max(1, samples))
    return face_present, t_first_face, max_face_area_ratio, face_time_ratio


face_present, t_first_face, max_face_area_ratio, face_time_ratio = compute_faces_stats(frames, step=5)
print(f"Faces present={face_present} | t_first_face={t_first_face} | max_face_area={max_face_area_ratio:.3f} | face_time_ratio={face_time_ratio:.2f}")


# %% [markdown]
# ## 12) Color Tone and Palette
# - Warm/cool balance and palette stability across keyframes

# %%
# Step 12: Color tone & palette

def warm_cool_balance(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0  # approx 0..360
    warm = ((h <= 30) | (h >= 330)).astype(np.float32)
    cool = ((h >= 90) & (h <= 150)).astype(np.float32)
    w_sum = warm.mean()
    c_sum = cool.mean()
    denom = max(1e-6, (w_sum + c_sum))
    return float((w_sum - c_sum) / denom)  # -1 cool .. +1 warm


def palette_stability(frames_seq: List[Frame], indices: List[int]) -> float:
    if not indices:
        return 0.0
    hues = []
    for i in indices:
        img = frames_seq[i].img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hues.append(np.mean(hsv[..., 0]))
    return float(np.std(hues) / 90.0)


wc_values = [warm_cool_balance(frames[i].img) for i in (keyframe_indices[:min(8, len(keyframe_indices))] if 'keyframe_indices' in globals() else [])] or [warm_cool_balance(f.img) for f in frames[:min(8, len(frames))]]
warm_cool = float(np.mean(wc_values)) if wc_values else 0.0
palette_var = palette_stability(frames, keyframe_indices if 'keyframe_indices' in globals() else list(range(min(8, len(frames)))))
print(f"Warm/cool balance={warm_cool:.2f} | Palette stability (std)={palette_var:.2f}")


# %% [markdown]
# ## 13) Novelty and Surprise (drift across keyframes)
# - Measure appearance drift between consecutive keyframes

# %%
# Step 13: Novelty/surprise

def novelty_from_keyframes(frames_seq: List[Frame], key_idxs: List[int]):
    if not key_idxs or len(key_idxs) < 2:
        return 0.0, 0
    scores = []
    anomalies = 0
    for a, b in zip(key_idxs[:-1], key_idxs[1:]):
        img_a = frames_seq[a].img
        img_b = frames_seq[b].img
        hdist = hist_distance_bhattacharyya(img_a, img_b)
        ssim_val = frame_ssim(img_a, img_b)
        novelty = 0.5 * hdist + 0.5 * (1.0 - ssim_val)
        scores.append(novelty)
        if novelty > 0.5:
            anomalies += 1
    peak = max(scores) if scores else 0.0
    return float(peak), int(anomalies)


novelty_peak, anomaly_count = novelty_from_keyframes(frames, keyframe_indices if 'keyframe_indices' in globals() else [])
print(f"Novelty peak={novelty_peak:.2f} | anomaly_count={anomaly_count}")


# %% [markdown]
# ## 14) Gestalt: Clutter and Isolation
# - Edge density (clutter) and largest salient blob ratio (isolation)

# %%
# Step 14: Gestalt/clutter/isolation

def edge_clutter_score(img_bgr: np.ndarray) -> float:
    key = id(img_bgr)
    if key in EDGE_CLUTTER_CACHE:
        return EDGE_CLUTTER_CACHE[key]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    val = float(edges.mean()) / 255.0
    EDGE_CLUTTER_CACHE[key] = val
    return val


def isolation_ratio(img_bgr: np.ndarray) -> float:
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if not ok:
            return 0.0
        t = np.quantile(smap, 0.8)
        mask = (smap >= t).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = float(areas.max())
        total = float(mask.sum())
        return float(min(1.0, largest / max(1e-6, total)))
    except Exception:
        return 0.0


key_imgs = [frames[i].img for i in (keyframe_indices if 'keyframe_indices' in globals() else [])]
if not key_imgs:
    key_imgs = [f.img for f in frames[::max(1, len(frames)//6 or 1)]]
clutter = float(np.mean([edge_clutter_score(img) for img in key_imgs])) if key_imgs else 0.0
isolation = float(np.mean([isolation_ratio(img) for img in key_imgs])) if key_imgs else 0.0
print(f"Clutter score={clutter:.2f} | Isolation ratio={isolation:.2f}")


# %% [markdown]
# ## 15) Platform Norms (format proxies)
# - Aspect ratio class, letterbox/pillarbox ratio

# %%
# Step 15: Platform norms

def aspect_ratio_class(width: int, height: int) -> str:
    r = float(width) / float(max(1, height))
    if r < 0.9:
        return "portrait"
    if r > 1.2:
        return "landscape"
    return "square"


def letterbox_ratio(img_bgr: np.ndarray, thresh_dark: int = 20) -> float:
    h, w = img_bgr.shape[:2]
    row_top = img_bgr[0: max(1, h // 50), :]
    row_bottom = img_bgr[h - max(1, h // 50):, :]
    col_left = img_bgr[:, 0: max(1, w // 50)]
    col_right = img_bgr[:, w - max(1, w // 50):]
    def dark_ratio(block):
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        return float((gray < thresh_dark).mean())
    return float(np.mean([dark_ratio(row_top), dark_ratio(row_bottom), dark_ratio(col_left), dark_ratio(col_right)]))


ar_class = aspect_ratio_class(SELECTED.width, SELECTED.height)
lb_ratio = letterbox_ratio(frames[0].img) if frames else 0.0
print(f"Aspect ratio={ar_class} | letterbox_ratio={lb_ratio:.2f}")


# %% [markdown]
# ## 16) Semantic Coherence (simplicity) and Palette Stability
# - Scene simplicity from low clutter variance; palette stability

# %%
# Step 16: Semantic coherence proxy

def scene_simplicity_score(clutter_values: List[float]) -> float:
    if not clutter_values:
        return 0.0
    mean_c = float(np.mean(clutter_values))
    std_c = float(np.std(clutter_values))
    return float(np.clip(1.0 - (0.6 * mean_c + 0.4 * std_c), 0.0, 1.0))


simplicity = scene_simplicity_score([edge_clutter_score(img) for img in key_imgs]) if key_imgs else 0.0
print(f"Scene simplicity={simplicity:.2f} | Palette stability={palette_var:.2f}")


# %% [markdown]
# ## 17) Center-weighted Saliency and Text Safe-area
# - Center saliency ratio; fraction of text inside safe area

# %%
# Step 17: Center saliency & safe-area for text

def center_saliency_ratio(img_bgr: np.ndarray) -> float:
    key = id(img_bgr)
    if key in CENTER_SAL_CACHE:
        return CENTER_SAL_CACHE[key]
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if not ok:
            CENTER_SAL_CACHE[key] = 0.0
            return 0.0
        h, w = smap.shape[:2]
        cx0, cy0 = int(w * 0.33), int(h * 0.33)
        cx1, cy1 = int(w * 0.67), int(h * 0.67)
        center = smap[cy0:cy1, cx0:cx1]
        val = float(center.mean() / max(1e-6, smap.mean()))
        CENTER_SAL_CACHE[key] = val
        return val
    except Exception:
        CENTER_SAL_CACHE[key] = 0.0
        return 0.0


def text_safe_area_ratio(frames_seq: List[Frame], sample_step: int = 10) -> float:
    total_boxes = 0
    inside = 0
    for i in range(0, len(frames_seq), max(1, sample_step)):
        f = frames_seq[i]
        _, boxes, _ = detect_text_mser_stats(f.img)
        if not boxes:
            continue
        h, w = f.img.shape[:2]
        sx0, sy0 = int(w * 0.1), int(h * 0.1)
        sx1, sy1 = int(w * 0.9), int(h * 0.9)
        for (x, y, bw, bh) in boxes:
            total_boxes += 1
            cx, cy = x + bw // 2, y + bh // 2
            if sx0 <= cx <= sx1 and sy0 <= cy <= sy1:
                inside += 1
    return float(inside / max(1, total_boxes))


center_sal = float(np.mean([center_saliency_ratio(img) for img in key_imgs])) if key_imgs else 0.0
safe_text_ratio = text_safe_area_ratio(frames, sample_step=10)
print(f"Center saliency ratio={center_sal:.2f} | text safe-area ratio={safe_text_ratio:.2f}")


# %%
# %% [markdown]
# ## 18) Aggregate, Score, and Rank Videos
# - Iterate all videos, extract features via Steps 1–17, compute a composite score, and rank
# - Writes a CSV summary and prints top results (no images displayed)

# %%
# Aggregate + rank

def extract_features_for_video(path: Path,
                               target_fps: float = 12.0,
                               short_edge: int = 576,
                               max_frames: int = 300) -> dict:
    meta = probe_video(path)
    v_frames = load_video_frames(path, target_fps=target_fps, short_edge=short_edge, max_frames=max_frames)
    if not v_frames:
        return {
            "file": path.name,
            "error": "no_frames"
        }

    # Step 1
    v_shots = detect_shots(v_frames)
    v_key_idx = [select_keyframe(v_frames, s, e) for (s, e) in v_shots]
    # shot stats
    shot_durations = [max(0.0, v_frames[e].t - v_frames[s].t) for (s, e) in v_shots] if v_shots else []

    # Step 2
    v_opening = slice_frames_by_time(v_frames, 0.0, 3.0)
    v_t_first_action = compute_first_action_time(v_opening) if v_opening else None
    v_t_first_text = None
    if v_opening:
        for f in v_opening:
            if text_presence_score(f.img) > 0.02:
                v_t_first_text = f.t
                break
    v_opening_top_rel = rank_opening_frames(v_opening, top_k=3) if v_opening else []
    v_opening_frames = [v_opening[i] for i in v_opening_top_rel] if v_opening_top_rel else []
    has_action_early = 1.0 if (v_t_first_action is not None and v_t_first_action <= 2.0) else 0.0
    has_text_early = 1.0 if (v_t_first_text is not None and v_t_first_text <= 2.0) else 0.0
    v_avg_sal = float(np.mean([frame_saliency_score(f.img) for f in v_opening[:min(10, len(v_opening))]])) if v_opening else 0.0
    v_start_strength = float(np.clip(0.4 * has_action_early + 0.3 * has_text_early + 0.3 * v_avg_sal, 0.0, 1.0))

    # Step 3
    v_cta_timeline, v_cta_spans = compute_cta_spans(v_frames, coverage_thresh=0.01)
    v_cta_present = len(v_cta_spans) > 0
    v_t_first_cta = v_cta_spans[0][0] if v_cta_present else None
    v_cta_dwell = sum(max(0.0, e - s) for (s, e) in v_cta_spans)
    v_cta_longest = max((e - s) for (s, e) in v_cta_spans) if v_cta_present else 0.0
    v_cta_contrast = float(np.mean([c for (_, _, c) in v_cta_timeline])) if v_cta_timeline else 0.0

    # Step 4
    v_t_first_product, v_t_peak_product, v_max_product_area_ratio, v_product_dwell = compute_product_reveals(v_frames)

    # Step 5
    v_flow_series = compute_flow_series(v_frames)
    v_bursts, v_motion_mean, v_motion_peak, v_camera_motion = top_motion_bursts(v_flow_series)
    v_burst_ts = [t for (_, t) in v_bursts]

    # Step 6
    v_top_text = rank_text_dense_frames(v_frames, top_k=3)

    # Step 7
    v_endcard_present, v_endcard_duration, v_endcard_frame, v_endcard_readability = find_endcard(v_frames)

    # Step 8 (thumbnails)
    def v_thumbnail_candidates():
        cands = []
        for idx in v_key_idx:
            cands.append((thumbnail_score(v_frames[idx].img), v_frames[idx].t))
        if v_endcard_present and v_endcard_frame is not None:
            cands.append((thumbnail_score(v_endcard_frame.img), v_endcard_frame.t))
        for f in v_opening_frames:
            cands.append((thumbnail_score(f.img), f.t))
        for (_, _, i) in v_top_text:
            cands.append((thumbnail_score(v_frames[i].img), v_frames[i].t))
        return cands
    v_thumb_cands = v_thumbnail_candidates()
    v_thumb_cands.sort(key=lambda x: x[0], reverse=True)
    v_top_thumb_score = v_thumb_cands[0][0] if v_thumb_cands else 0.0

    # Step 9
    def v_shot_cut_times():
        return [v_frames[s].t for (s, e) in v_shots] if v_shots else []
    v_cut_ts = v_shot_cut_times()
    v_peak_density, v_montage_window = peak_cut_density(v_cut_ts, window_s=1.0)

    # Step 10
    total_end = v_frames[-1].t
    v_first_win = window_frames(v_frames, 0.0, min(1.0, total_end))
    v_last_win = window_frames(v_frames, max(0.0, total_end - 1.0), total_end)
    v_avg_first = average_frame(v_first_win)
    v_avg_last = average_frame(v_last_win)
    if v_avg_first is not None and v_avg_last is not None:
        v_ssim_fl = ssim_between(v_avg_first, v_avg_last)
        v_hist_fl = hist_bhatt_between(v_avg_first, v_avg_last)
        v_loop_readiness = float(np.clip(0.7 * v_ssim_fl + 0.3 * (1.0 - v_hist_fl), 0.0, 1.0))
    else:
        v_ssim_fl, v_hist_fl, v_loop_readiness = 0.0, 1.0, 0.0

    # Step 11: Faces and Gaze (proxy)
    v_face_present, v_t_first_face, v_max_face_area_ratio, v_face_time_ratio = compute_faces_stats(v_frames, step=5)

    # Step 12: Color Tone and Palette
    if v_key_idx:
        sample_idxs = v_key_idx[:min(8, len(v_key_idx))]
    else:
        sample_idxs = list(range(0, min(8, len(v_frames))))
    v_wc_values = [warm_cool_balance(v_frames[i].img) for i in sample_idxs]
    v_warm_cool = float(np.mean(v_wc_values)) if v_wc_values else 0.0
    v_palette_var = palette_stability(v_frames, sample_idxs)

    # Step 13: Novelty and Surprise
    v_novelty_peak, v_anomaly_count = novelty_from_keyframes(v_frames, v_key_idx)

    # Step 14: Gestalt (clutter, isolation)
    v_key_imgs = [v_frames[i].img for i in v_key_idx] if v_key_idx else [f.img for f in v_frames[::max(1, len(v_frames)//6 or 1)]]
    v_clutter_vals = [edge_clutter_score(img) for img in v_key_imgs] if v_key_imgs else []
    v_clutter = float(np.mean(v_clutter_vals)) if v_clutter_vals else 0.0
    v_isolation_vals = [isolation_ratio(img) for img in v_key_imgs] if v_key_imgs else []
    v_isolation = float(np.mean(v_isolation_vals)) if v_isolation_vals else 0.0

    # Step 15: Platform Norms
    v_ar_class = aspect_ratio_class(meta.width, meta.height)
    v_letterbox = letterbox_ratio(v_frames[0].img) if v_frames else 0.0

    # Step 16: Semantic Coherence
    v_simplicity = scene_simplicity_score(v_clutter_vals)

    # Step 17: Center-weighted Saliency & Text Safe-area
    v_center_sal = float(np.mean([center_saliency_ratio(img) for img in v_key_imgs])) if v_key_imgs else 0.0
    v_safe_text_ratio = text_safe_area_ratio(v_frames, sample_step=10)

    return {
        "file": path.name,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "duration": meta.duration_s,
        "num_shots": len(v_shots),
        "avg_shot_s": float(np.mean(shot_durations)) if shot_durations else 0.0,
        "start_strength": v_start_strength,
        "t_first_action": v_t_first_action,
        "t_first_text": v_t_first_text,
        "cta_present": v_cta_present,
        "t_first_cta": v_t_first_cta,
        "cta_dwell": v_cta_dwell,
        "cta_longest": v_cta_longest,
        "cta_contrast": v_cta_contrast,
        "t_first_product": v_t_first_product,
        "max_product_area_ratio": v_max_product_area_ratio,
        "product_dwell": v_product_dwell,
        "motion_mean": v_motion_mean,
        "motion_peak": v_motion_peak,
        "camera_motion": v_camera_motion,
        "endcard_present": v_endcard_present,
        "endcard_duration": v_endcard_duration,
        "endcard_readability": v_endcard_readability,
        "thumbnail_top_score": v_top_thumb_score,
        "peak_cut_density": v_peak_density,
        "loop_readiness": v_loop_readiness,
        # New heuristics 11–17
        "face_present": v_face_present,
        "t_first_face": v_t_first_face,
        "max_face_area_ratio": v_max_face_area_ratio,
        "face_time_ratio": v_face_time_ratio,
        "warm_cool": v_warm_cool,
        "palette_var": v_palette_var,
        "novelty_peak": v_novelty_peak,
        "anomaly_count": v_anomaly_count,
        "clutter": v_clutter,
        "isolation": v_isolation,
        "aspect_ratio_class": v_ar_class,
        "letterbox_ratio": v_letterbox,
        "simplicity": v_simplicity,
        "center_saliency_ratio": v_center_sal,
        "safe_text_ratio": v_safe_text_ratio,
    }


def _nz(x, default=0.0):
    try:
        return float(x) if x is not None else float(default)
    except Exception:
        return float(default)


def score_video(feat: dict) -> float:
    # Composite score combining core steps and new heuristics
    s = 0.0
    s += 0.25 * _nz(feat.get("start_strength"), 0.0)
    # earlier product reveal better
    if feat.get("t_first_product") is not None and _nz(feat.get("duration"), 0.0) > 0:
        s += 0.15 * (1.0 - min(1.0, _nz(feat.get("t_first_product"), 0.0) / max(1e-6, feat["duration"])) )
    # CTA presence and earliness
    s += 0.10 * (1.0 if feat.get("cta_present") else 0.0)
    if feat.get("t_first_cta") is not None and _nz(feat.get("duration"), 0.0) > 0:
        s += 0.05 * (1.0 - min(1.0, _nz(feat.get("t_first_cta"), 0.0) / max(1e-6, feat["duration"])) )
    s += 0.10 * _nz(feat.get("endcard_readability"), 0.0)
    s += 0.05 * _nz(feat.get("loop_readiness"), 0.0)
    s += 0.05 * min(1.0, _nz(feat.get("peak_cut_density"), 0.0) / 5.0)
    # motion moderate weight
    s += 0.10 * min(1.0, _nz(feat.get("motion_mean"), 0.0))
    # thumbnail top score scaled
    s += 0.15 * min(1.0, _nz(feat.get("thumbnail_top_score"), 0.0) / 100.0)
    # new heuristics small weights
    s += 0.05 * (1.0 if feat.get("face_present") else 0.0)
    if feat.get("t_first_face") is not None and _nz(feat.get("duration"), 0.0) > 0:
        s += 0.02 * (1.0 - min(1.0, _nz(feat.get("t_first_face"), 0.0) / max(1e-6, feat["duration"])) )
    s += 0.03 * np.clip(_nz(feat.get("warm_cool"), 0.0), -1.0, 1.0)
    s += 0.03 * (1.0 - min(1.0, _nz(feat.get("palette_var"), 0.0)))
    s += 0.03 * min(1.0, _nz(feat.get("novelty_peak"), 0.0))
    s += 0.03 * (1.0 - min(1.0, _nz(feat.get("clutter"), 0.0)))
    s += 0.03 * min(1.0, _nz(feat.get("isolation"), 0.0))
    s += 0.02 * _nz(feat.get("simplicity"), 0.0)
    s += 0.02 * min(1.0, _nz(feat.get("center_saliency_ratio"), 0.0))
    s += 0.02 * _nz(feat.get("safe_text_ratio"), 0.0)
    return float(s)


# %%
# Run aggregation
all_results = []
for path in VIDEO_PATHS:
    try:
        feats = extract_features_for_video(path, target_fps=TARGET_FPS, short_edge=SHORT_EDGE, max_frames=MAX_FRAMES)
        feats["score"] = score_video(feats)
        all_results.append(feats)
        print(f"Processed {path.name} -> score={feats['score']:.3f}")
    except Exception as e:
        print(f"Error processing {path.name}: {e}")

# %%
# Rank and save
all_results.sort(key=lambda d: d.get("score", 0.0), reverse=True)

try:
    import csv
    out_csv = PROJECT_ROOT / "video_features.csv"
    if all_results:
        keys = [
            "file","width","height","fps","duration","num_shots","avg_shot_s","start_strength",
            "t_first_action","t_first_text","cta_present","t_first_cta","cta_dwell","cta_longest","cta_contrast",
            "t_first_product","max_product_area_ratio","product_dwell","motion_mean","motion_peak","camera_motion",
            "endcard_present","endcard_duration","endcard_readability","thumbnail_top_score","peak_cut_density","loop_readiness",
            # new heuristics 11–17
            "face_present","t_first_face","max_face_area_ratio","face_time_ratio",
            "warm_cool","palette_var","novelty_peak","anomaly_count","clutter","isolation",
            "aspect_ratio_class","letterbox_ratio","simplicity","center_saliency_ratio","safe_text_ratio",
            "score"
        ]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in all_results:
                w.writerow({k: row.get(k) for k in keys})
        print(f"Saved features to {out_csv}")
except Exception as e:
    print(f"CSV save error: {e}")

print("Top 5 videos:")
for r in all_results[:5]:
    print(f"  {r['file']}: score={r['score']:.3f}, start={_nz(r.get('start_strength')):.2f}, cta={r.get('cta_present')}, prod_t={r.get('t_first_product')}")




# %%
print("Videos in order of score:")
for r in all_results:
    print(f"  {r['file']}: score={r['score']:.3f}")

# %% [markdown]
# ## Heuristic Notes (One-Liners)
# - Color/Contrast: high contrast; warm/cool balance; brand palette early
# - Motion/Salience: rapid onset; center saliency; purposeful camera motion
# - Faces/Gaze: early faces; direct gaze; gaze cueing to product/logo
# - Timing/Rhythm: strong first 1–2s; pacing ramp; stabilize for message
# - Text/Overlays: minimal, high-contrast; early brand; safe-area
# - Novelty/Surprise: 1–2 spikes; avoid fatigue; align with motion/edits
# - Gestalt/Grouping: low clutter; clear figure-ground; isolate subject
# - Platform Norms: UGC style when apt; aspect ratio; watermarks/stickers
# - Emotional Visuals: expressive close-ups; color grading intent
# - Semantic Coherence: simple scenes; visual metaphors
# 

# %% [markdown]
# (See `videos/Workflow.md` for full details.)

# %%
