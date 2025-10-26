import os
import io
import cv2
import math
import json
import torch
import tempfile
import numpy as np
import random

def _maybe_seed_everything():
    """Seed Python, NumPy, and PyTorch if TPPGAZE_SEED is set (>=0)."""
    seed_env = os.getenv("TPPGAZE_SEED")
    if seed_env is None:
        return  # no seeding requested
    seed = int(seed_env)
    if seed < 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # make cuDNN deterministic (at some perf cost)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # stricter determinism (PyTorch 1.12+). If an op is nondeterministic, it warns.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # helps some BLAS kernels be deterministic
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# --- tppgaze imports
from tppgaze.tppgaze import TPPGaze

# ---------------------------
# Module-level singleton model
# ---------------------------
_TPP_MODEL: Optional[TPPGaze] = None

def _resolve_paths() -> Tuple[str, str]:
    cfg_env = os.getenv("TPPGAZE_CFG")
    ckpt_env = os.getenv("TPPGAZE_CKPT")
    if cfg_env and ckpt_env and Path(cfg_env).is_file() and Path(ckpt_env).is_file():
        return cfg_env, ckpt_env

    base = Path(__file__).resolve().parents[1]  # repo root
    candidates = [
        (base / "tppgaze/data/config.yaml", base / "tppgaze/data/model_transformer.pth"),
        (Path("tppgaze/data/config.yaml"), Path("tppgaze/data/model_transformer.pth")),
    ]
    for cfg, ckpt in candidates:
        if cfg.is_file() and ckpt.is_file():
            return str(cfg), str(ckpt)

    raise FileNotFoundError(
        "Could not locate TPPGaze config/weights. Set TPPGAZE_CFG/TPPGAZE_CKPT or place files under tppgaze/data/."
    )


def _get_model() -> TPPGaze:
    global _TPP_MODEL
    if _TPP_MODEL is not None:
        return _TPP_MODEL

    cfg_path, ckpt_path = _resolve_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TPPGaze(cfg_path, ckpt_path, device)
    model.load_model()
    _TPP_MODEL = model
    return _TPP_MODEL


# ---------------------------
# Attention map utilities
# ---------------------------
def _stamp_gaussian(
    heatmap: np.ndarray,
    x: float,
    y: float,
    weight: float,
    sigma_px: float = 16.0
) -> None:
    """
    Add a Gaussian blob centered at (x,y) with given weight into 'heatmap'.
    Assumes heatmap is float32 HxW in [0, +inf).
    """
    h, w = heatmap.shape
    # Kernel size: 6*sigma rounded to odd
    k = int(max(3, round(6.0 * sigma_px)))
    if k % 2 == 0:
        k += 1
    r = k // 2

    # Bounding window
    x0 = int(round(x)) - r
    y0 = int(round(y)) - r
    x1 = x0 + k
    y1 = y0 + k

    # Overlap with heatmap
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    xs0 = max(0, x0); ys0 = max(0, y0)
    xs1 = min(w, x1); ys1 = min(h, y1)

    # Build coordinates for the visible kernel region
    xx = np.arange(xs0, xs1) - x
    yy = np.arange(ys0, ys1) - y
    X, Y = np.meshgrid(xx, yy)
    g = np.exp(-(X**2 + Y**2) / (2.0 * sigma_px**2))

    # Normalize kernel to unit sum, then scale by weight
    s = g.sum()
    if s > 0:
        g *= (weight / s)
        heatmap[ys0:ys1, xs0:xs1] += g.astype(np.float32)


def _build_heatmap_from_scanpaths(
    scanpaths, img_h, img_w, *,
    sigma_px=10.0,          # ↓ was 16.0
    duration_gamma=0.5,     # NEW: compress extreme durations
    per_path_norm=True,     # NEW: normalize per path
    use_duration=True       # NEW: toggle to ignore durations
) -> np.ndarray:
    heat = np.zeros((img_h, img_w), dtype=np.float32)
    for sp in scanpaths:
        if sp is None or len(sp) == 0: continue
        durs = np.clip(sp[:, 2], 0.0, None)
        if not use_duration:
            weights = np.ones_like(durs, dtype=np.float32)
        else:
            weights = np.power(np.maximum(durs, 1e-3), duration_gamma).astype(np.float32)
            if per_path_norm:
                s = float(weights.sum())
                if s > 0: weights /= s
        for (x, y), w in zip(sp[:, :2], weights):
            if not (np.isnan(x) or np.isnan(y)):
                _stamp_gaussian(heat, float(np.clip(x, 0, img_w-1)),
                                      float(np.clip(y, 0, img_h-1)),
                                      float(w), sigma_px=sigma_px)
    m = float(heat.max())
    if m > 0: heat /= m
    return heat



def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy (base e) of a normalized heatmap.
    """
    q = p.astype(np.float64)
    s = q.sum()
    if s <= 0:
        return 0.0
    q = q / s
    q = np.clip(q, eps, 1.0)
    return float(-(q * np.log(q)).sum())


def _topk_peaks(
    heat: np.ndarray,
    k: int = 5,
    nms_radius: int = 12
) -> List[Tuple[int,int,float]]:
    """
    Return up to k peak (x,y,value) with simple NMS (Chebyshev radius).
    """
    h, w = heat.shape
    flat_idx = np.argpartition(heat.ravel(), -k*10)[-k*10:]  # oversample, then NMS
    cand = [(int(i % w), int(i // w), float(heat.ravel()[i])) for i in flat_idx]
    cand.sort(key=lambda t: t[2], reverse=True)

    kept: List[Tuple[int,int,float]] = []
    for x, y, v in cand:
        suppress = False
        for xx, yy, vv in kept:
            if max(abs(x - xx), abs(y - yy)) <= nms_radius:
                suppress = True
                break
        if not suppress:
            kept.append((x, y, v))
            if len(kept) >= k:
                break
    return kept


def _summarize_scanpaths(scanpaths: List[np.ndarray]) -> Dict[str, Any]:
    """
    Aggregate basic scanpath statistics.
    """
    fix_durations = []
    n_fix = 0
    total_duration = 0.0
    for sp in scanpaths:
        if sp is None or len(sp) == 0:
            continue
        n_fix += sp.shape[0]
        fix_durations.extend(sp[:, 2].tolist())

    if len(fix_durations) == 0:
        return {
            "num_scanpaths": len(scanpaths),
            "num_fixations": 0,
            "total_fixation_time_sec": 0.0,
            "mean_fixation_dur_sec": 0.0,
            "median_fixation_dur_sec": 0.0,
        }
    arr = np.array(fix_durations, dtype=np.float32)
    return {
        "num_scanpaths": len(scanpaths),
        "num_fixations": int(n_fix),
        "total_fixation_time_sec": float(arr.clip(min=0).sum()),
        "mean_fixation_dur_sec": float(arr.mean()),
        "median_fixation_dur_sec": float(np.median(arr)),
    }


# ---------------------------
# Public API
# ---------------------------
def extract(image: np.ndarray, image_id: str) -> Dict[str, Any]:
    """
    Extract attention map features from image using tppgaze.

    Args:
        image: Standardized image (1024x1024) array (H, W, 3) with values in [0, 1]
        image_id: Unique identifier for the image

    Returns:
        Dict with attention heatmap, peaks, scanpath stats, and raw scanpaths.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a (H,W,3) numpy array.")

    H, W, _ = image.shape
    if (H, W) != (1024, 1024):
        raise ValueError("image must be 1024x1024 as specified.")

    # Convert to uint8 BGR and save to a temp file (tppgaze expects a path)
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        model = _get_model()

        _maybe_seed_everything()

        # Inference parameters (tune if desired)
        n_simulations = int(os.getenv("TPPGAZE_N_SIM", "1")) #multiple will give you a heat map!
        sample_duration = float(os.getenv("TPPGAZE_DURATION_SEC", "2.0"))
        
        # Generate scanpaths (list of arrays [x, y, fix_duration])
        scanpaths: List[np.ndarray] = model.generate_predictions(
            tmp_path, sample_duration, n_simulations
        )
        
        # --- FIX 1: coords scaling (normalized vs pixels) and axis order ---
        H, W = image.shape[:2]

        def _ensure_pixels(paths):
            fixed = []
            for sp in paths:
                if sp is None or len(sp) == 0:
                    fixed.append(sp); continue
                sp = sp.astype(np.float32)
                # If coords look normalized, scale to pixels
                if np.nanmax(sp[:, :2]) <= 2.0:
                    sp[:, 0] *= W  # x
                    sp[:, 1] *= H  # y
                fixed.append(sp)
            return fixed

        def _swap_xy(paths):
            swapped = []
            for sp in paths:
                if sp is None or len(sp) == 0:
                    swapped.append(sp); continue
                q = sp.copy()
                q[:, [0, 1]] = q[:, [1, 0]]  # swap x<->y
                swapped.append(q)
            return swapped

        def _score_sharpness(paths, sigma=6.0):
            # quick “how peaky is this?” score to choose xy vs yx
            agg = np.zeros((H, W), dtype=np.float32)
            for sp in paths:
                if sp is None: continue
                for i in range(sp.shape[0]):
                    x, y = float(sp[i, 0]), float(sp[i, 1])
                    # unit weight for scoring
                    _stamp_gaussian(agg, x, y, 1.0, sigma_px=sigma)
            mean = float(agg.mean()) + 1e-8
            return float(agg.max() / mean)

        paths_xy = _ensure_pixels(scanpaths)
        paths_yx = _ensure_pixels(_swap_xy(scanpaths))

        score_xy = _score_sharpness(paths_xy)
        score_yx = _score_sharpness(paths_yx)
        scanpaths = paths_yx if score_yx > score_xy else paths_xy

        # --- FIX 2: duration units (ms vs sec) ---
        # The repo returns fixation duration most commonly in **milliseconds**.
        # Convert to seconds if values look like ms.
        for i, sp in enumerate(scanpaths):
            if sp is None or len(sp) == 0:
                continue
            dur = sp[:, 2]
            med = float(np.nanmedian(dur))
            # Heuristic: if median in [50, 2000], treat as milliseconds and convert to seconds
            if 50.0 <= med <= 2000.0:
                sp[:, 2] = dur / 1000.0
                scanpaths[i] = sp

        # --- runtime knobs via env ---
        USE_DURATION = os.getenv("TPPGAZE_USE_DURATION", "1") != "0"
        DURATION_GAMMA = float(os.getenv("TPPGAZE_DURATION_GAMMA", "0.5"))   # 0..1
        SIGMA_PX = float(os.getenv("TPPGAZE_SIGMA_PX", "10"))                # e.g., 6..12
        NMS_RADIUS = int(os.getenv("TPPGAZE_NMS_RADIUS", "6"))               # e.g., 4..8
        FORCE_ORDER = os.getenv("TPPGAZE_FORCE_ORDER", "").strip().lower()   # "", "xy", "yx"
        JITTER_PX = float(os.getenv("TPPGAZE_JITTER_PX", "0"))               # e.g., 2.0
        
        if FORCE_ORDER == "xy":
            pass
        elif FORCE_ORDER == "yx":
            scanpaths = _ensure_pixels(_swap_xy(scanpaths))
            # else: keep the auto-chosen one
        
        if JITTER_PX > 0:
            for sp in scanpaths:
                if sp is not None and len(sp) > 0:
                    sp[:, 0] += np.random.normal(0, JITTER_PX, size=sp.shape[0])
                    sp[:, 1] += np.random.normal(0, JITTER_PX, size=sp.shape[0])

        tot_sec = sum(s[:,2].sum() for s in scanpaths if s is not None)
        p0 = scanpaths[0] if scanpaths and len(scanpaths[0]) else None
        if p0 is not None:
            xs, ys, ds = p0[:,0], p0[:,1], p0[:,2]
            print(f"[TPPGaze] order={FORCE_ORDER or 'auto'} "
                f"x_std={xs.std():.1f} y_std={ys.std():.1f} "
                f"dur_med={np.median(ds):.3f}s tot≈{tot_sec:.2f}s")


        
        
        # Build attention heatmap from fixations (duration-weighted)
        heatmap = _build_heatmap_from_scanpaths(
            scanpaths, img_h=H, img_w=W,
            sigma_px=SIGMA_PX,
            duration_gamma=DURATION_GAMMA,
            per_path_norm=True,
            use_duration=USE_DURATION,
        )
        peaks = _topk_peaks(heatmap, k=5, nms_radius=NMS_RADIUS)
        
        # Normalize heatmap for stats & return
        heat_sum = float(heatmap.sum())
        entropy = _entropy(heatmap if heat_sum > 0 else np.zeros_like(heatmap))
        peaks = _topk_peaks(heatmap, k=5, nms_radius=6)  # [(x,y,val), ...]

        # Center-of-mass (saliency centroid)
        if heatmap.max() > 0:
            y_idx, x_idx = np.indices(heatmap.shape)
            denom = heatmap.sum()
            cx = float((x_idx * heatmap).sum() / denom)
            cy = float((y_idx * heatmap).sum() / denom)
        else:
            cx, cy = float(W/2), float(H/2)

        # Basic scanpath statistics
        sp_stats = _summarize_scanpaths(scanpaths)

        return {
            "status": "ok",
            "image_id": image_id,
            "heatmap": heatmap.astype(np.float32),          # 1024x1024, [0,1]
            "peaks_top5": [{"x": int(x), "y": int(y), "value": float(v)} for (x, y, v) in peaks],
            "saliency_centroid": {"x": cx, "y": cy},
            "entropy": entropy,
            "scanpath_stats": sp_stats,
            # Optional: include lightweight representation of scanpaths
            "scanpaths": [
                {"fixations": sp.astype(np.float32)} for sp in scanpaths if sp is not None
            ],
            "params": {
                "sigma_px": 16.0,
                "n_simulations": n_simulations,
                "sample_duration_sec": sample_duration,
            },
        }

    finally:
        # Clean up the temp image
        try:
            os.remove(tmp_path)
        except Exception:
            pass