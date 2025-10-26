"""
Image Batch Extraction Runner (Parallel, Deadline-Aware)
- Stage A: Standardize images (CPU-heavy) in a pool
- Stage B: Extract features (YOLO+CLIP+OCR, TPPGaze) in a pool
- As soon as an image is standardized, it flows to extraction (pipeline)
- Saves interpretable artifacts per image + project index CSV

CLI:
    python -m image.extraction \
        --images_dir ads/images \
        --out_dir extracted_features \
        --size 1024 1024 \
        --std_workers 6 \
        --ext_workers auto \
        --deadline_seconds 300 \
        --per_image_timeout 120 \
        [--force]
"""

from __future__ import annotations

import os
import csv
import json
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ====== project extractors (must exist in image/) ======
from image import visual_semantics as VS
from image import attention_map as AM
try:
    from image import llm_description as LLM  # optional
except Exception:
    LLM = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extraction")


# ----------------- helpers & data -----------------
@dataclass
class Meta:
    image_id: str
    original_path: str
    original_size: Tuple[int, int]  # (W, H)
    standardized_size: Tuple[int, int]  # (H, W)
    file_size_bytes: int
    format: str
    processing_time_ms: float = 0.0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def load_and_standardize(path: Path, target_size=(1024, 1024), keep_aspect=True) -> Tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    if keep_aspect:
        tw, th = target_size
        w, h = img.size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        x_off = (tw - nw) // 2
        y_off = (th - nh) // 2
        canvas.paste(img_resized, (x_off, y_off))
        std_pil = canvas
    else:
        std_pil = img.resize(target_size, Image.LANCZOS)
    arr = np.asarray(std_pil).astype(np.float32) / 255.0  # HxWx3 [0,1]
    return std_pil, arr


def draw_detections(pil_img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets or []:
        x1, y1, x2, y2 = [int(x) for x in d.get("bbox", [0, 0, 0, 0])]
        label = f"{d.get('class_name','')} {d.get('confidence',0):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        if font:
            draw.text((x1 + 4, y1 + 4), label, fill=(0, 255, 0), font=font)
    return out


def overlay_heatmap(pil_img: Image.Image, heatmap01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm
    base = np.array(pil_img).astype(np.float32) / 255.0
    hm_rgb = cm.get_cmap("jet")(np.clip(heatmap01, 0, 1))[:, :, :3]
    mix = (1.0 - alpha) * base + alpha * hm_rgb
    return Image.fromarray((np.clip(mix, 0, 1) * 255).astype(np.uint8))


def _torch_accel():
    t = getattr(VS, "_torch", None)
    if t is None:
        return {"cuda": False, "mps": False}
    cuda = bool(t.cuda.is_available())
    mps = False
    try:
        mps = bool(getattr(t.backends, "mps", None) and t.backends.mps.is_available())
    except Exception:
        pass
    return {"cuda": cuda, "mps": mps}


# ----------------- core runner -----------------
class Runner:
    def __init__(
        self,
        images_dir: str,
        out_dir: str,
        target_size: Tuple[int, int],
        std_workers: int,
        ext_workers: int,
        force: bool,
        deadline_seconds: int,
        per_image_timeout: Optional[int],
    ):
        self.images_dir = Path(images_dir)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.std_workers = std_workers
        self.ext_workers = ext_workers
        self.force = force
        self.deadline_seconds = deadline_seconds
        self.per_image_timeout = per_image_timeout
        self.start_time = time.time()

    # ---- Stage A: standardize ----
    def standardize_one(self, path: Path):
        image_id = path.stem
        img_dir = self.out_dir / image_id
        img_dir.mkdir(parents=True, exist_ok=True)

        summary_json = img_dir / f"{image_id}_summary.json"
        if summary_json.exists() and not self.force:
            # Already processed; just return sentinel to skip Stage B
            return {"skip": True, "image_id": image_id, "img_dir": img_dir}

        std_pil, std_arr = load_and_standardize(path, self.target_size, keep_aspect=True)
        std_pil.save(img_dir / f"{image_id}_standardized.png")

        md = Meta(
            image_id=image_id,
            original_path=str(path),
            original_size=Image.open(path).size,
            standardized_size=std_arr.shape[:2],
            file_size_bytes=os.path.getsize(path),
            format=(Image.open(path).format or "unknown"),
        )
        return {"skip": False, "image_id": image_id, "img_dir": img_dir, "std_pil": std_pil, "std_arr": std_arr, "meta": md}

    # ---- Stage B: extract ----
    def extract_one(self, payload) -> Optional[Dict[str, Any]]:
        if payload.get("skip"):
            # read and return existing summary (already processed)
            img_dir = payload["img_dir"]; image_id = payload["image_id"]
            try:
                with open(img_dir / f"{image_id}_summary.json", "r") as f:
                    return json.load(f)
            except Exception:
                return None

        image_id = payload["image_id"]
        img_dir: Path = payload["img_dir"]
        std_pil: Image.Image = payload["std_pil"]
        std_arr: np.ndarray = payload["std_arr"]
        meta: Meta = payload["meta"]

        t0 = time.time()
        try:
            # visual_semantics
            vs_out = VS.extract(std_arr, image_id)
            with open(img_dir / f"{image_id}_visual_semantics.json", "w") as f:
                json.dump(vs_out, f, indent=2, default=_json_default)
            draw_detections(std_pil, vs_out.get("detections", [])).save(img_dir / f"{image_id}_detections.png")
            ocr = vs_out.get("ocr", {}) or {}
            with open(img_dir / f"{image_id}_ocr.txt", "w") as f:
                f.write((ocr.get("full_text") or "").strip())

            # attention_map
            am_out = AM.extract(std_arr, image_id)
            np.savez_compressed(img_dir / f"{image_id}_attention_heatmap.npz", heatmap=am_out.get("heatmap"))
            hm = am_out.get("heatmap")
            if isinstance(hm, np.ndarray) and hm.ndim == 2:
                mmin, mmax = float(hm.min()), float(hm.max())
                hm01 = (hm - mmin) / (mmax - mmin + 1e-8) if mmax > mmin else np.zeros_like(hm)
                overlay_heatmap(std_pil, hm01, 0.45).save(img_dir / f"{image_id}_attention_overlay.png")
            with open(img_dir / f"{image_id}_attention_map.json", "w") as f:
                json.dump(am_out, f, indent=2, default=_json_default)

            # llm_description (optional)
            llm_out: Dict[str, Any] = {}
            if LLM and hasattr(LLM, "extract"):
                try:
                    llm_out = LLM.extract(std_arr, image_id)
                except Exception as e:
                    llm_out = {"error": str(e)}
            with open(img_dir / f"{image_id}_llm_description.json", "w") as f:
                json.dump(llm_out, f, indent=2, default=_json_default)

            # summary
            meta.processing_time_ms = (time.time() - t0) * 1000.0
            quick = {
                "num_detections": len(vs_out.get("detections", [])),
                "scene_top": (vs_out.get("scene_labels") or [{}])[0:1],
                "product_category": vs_out.get("product_category"),
                "ocr_snippet": (ocr.get("full_text") or "")[:160],
                "attention_entropy": am_out.get("entropy"),
            }
            files = {
                "standardized_png": f"{image_id}_standardized.png",
                "detections_png": f"{image_id}_detections.png",
                "attention_overlay_png": f"{image_id}_attention_overlay.png",
                "attention_heatmap_npz": f"{image_id}_attention_heatmap.npz",
                "ocr_txt": f"{image_id}_ocr.txt",
                "visual_semantics_json": f"{image_id}_visual_semantics.json",
                "attention_map_json": f"{image_id}_attention_map.json",
                "llm_description_json": f"{image_id}_llm_description.json",
            }
            summary = {"image_id": image_id, "metadata": asdict(meta), "files": files, "quicklook": quick}
            with open(img_dir / f"{image_id}_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=_json_default)
            return summary
        except Exception as e:
            log.exception(f"[{image_id}] extraction failed: {e}")
            return None

    # ---- pipeline orchestrator ----
    def run(self) -> List[Dict[str, Any]]:
        img_paths = sorted([p for p in self.images_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not img_paths:
            log.warning(f"No images found in {self.images_dir}")
            return []

        # Auto ext_workers: if GPU/MPS detected, default to 1 to avoid contention; else 2*cpu
        if self.ext_workers <= 0:
            accel = _torch_accel()
            self.ext_workers = 1 if (accel["cuda"] or accel["mps"]) else max(2, os.cpu_count() or 4)

        log.info(f"[config] std_workers={self.std_workers} ext_workers={self.ext_workers} deadline={self.deadline_seconds}s force={self.force}")

        deadline_at = self.start_time + float(self.deadline_seconds)
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.std_workers) as std_pool, \
             ThreadPoolExecutor(max_workers=self.ext_workers) as ext_pool:

            # kick off all standardization tasks
            std_futs: Dict[Future, Path] = {std_pool.submit(self.standardize_one, p): p for p in img_paths}
            ext_futs: List[Future] = []

            for fut in as_completed(std_futs):
                now = time.time()
                if now >= deadline_at:
                    log.warning("Global deadline reached; skipping remaining extractions.")
                    break

                path = std_futs[fut]
                try:
                    payload = fut.result()
                except Exception as e:
                    log.exception(f"[std] {path.name} failed: {e}")
                    continue

                # schedule extraction if time remains
                if payload is None:
                    continue

                # If already processed and not --force, load summary quickly
                if payload.get("skip"):
                    try:
                        with open(payload["img_dir"] / f"{payload['image_id']}_summary.json", "r") as f:
                            results.append(json.load(f))
                    except Exception:
                        pass
                    continue

                # check deadline before scheduling extraction
                if time.time() + 1.0 >= deadline_at:
                    log.warning(f"[skip] {payload['image_id']} (would exceed deadline)")
                    continue

                ext_fut = ext_pool.submit(self.extract_one, payload)
                ext_futs.append(ext_fut)

            # collect finished extractions with optional per-image timeout
            for ef in as_completed(ext_futs, timeout=max(0.1, deadline_at - time.time()) if deadline_at > time.time() else None):
                try:
                    res = ef.result(timeout=self.per_image_timeout) if self.per_image_timeout else ef.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    log.warning(f"[extract] a task timed out or failed: {e}")

        # project-level CSV index
        index_csv = self.out_dir / "features_index.csv"
        try:
            keys = [
                "image_id","original_path","original_size","standardized_size","file_size_bytes","format",
                "processing_time_ms","num_detections","product_category_from","product_category_top",
                "scene_top_label","scene_top_score","ocr_chars","attention_entropy",
            ]
            with open(index_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
                for s in results:
                    md = s["metadata"]; q = s["quicklook"]
                    pc = (q.get("product_category") or {})
                    pc_top = (pc.get("labels") or [{}])[0] if pc else {}
                    scene_top = (q.get("scene_top") or [{}])
                    row = {
                        "image_id": s["image_id"],
                        "original_path": md["original_path"],
                        "original_size": tuple(md["original_size"]),
                        "standardized_size": tuple(md["standardized_size"]),
                        "file_size_bytes": md["file_size_bytes"],
                        "format": md["format"],
                        "processing_time_ms": f"{md['processing_time_ms']:.1f}",
                        "num_detections": q.get("num_detections", 0),
                        "product_category_from": pc.get("from"),
                        "product_category_top": pc_top.get("label"),
                        "scene_top_label": (scene_top[0].get("label") if scene_top else None),
                        "scene_top_score": (f"{scene_top[0].get('score',0):.3f}" if scene_top else None),
                        "ocr_chars": len(q.get("ocr_snippet") or ""),
                        "attention_entropy": q.get("attention_entropy"),
                    }
                    w.writerow(row)
            log.info(f"[OK] Wrote index: {index_csv}")
        except Exception as e:
            log.warning(f"Could not write index CSV: {e}")

        return results


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default="ads/images")
    ap.add_argument("--out_dir", type=str, default="extracted_features")
    ap.add_argument("--size", nargs=2, type=int, default=(1024, 1024), help="Target size W H")
    ap.add_argument("--std_workers", type=int, default=max(2, (os.cpu_count() or 4)//2), help="Parallel workers for standardization")
    ap.add_argument("--ext_workers", type=str, default="auto", help="'auto' or integer for extraction pool size")
    ap.add_argument("--deadline_seconds", type=int, default=300, help="Global time budget (seconds)")
    ap.add_argument("--per_image_timeout", type=int, default=0, help="Optional timeout per extraction (seconds, 0=disabled)")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = ap.parse_args()

    # parse ext_workers
    if args.ext_workers.strip().lower() == "auto":
        ext_workers = 0  # auto inside Runner.run()
    else:
        try:
            ext_workers = max(1, int(args.ext_workers))
        except Exception:
            ext_workers = 1

    runner = Runner(
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        target_size=(args.size[0], args.size[1]),
        std_workers=max(1, int(args.std_workers)),
        ext_workers=ext_workers,
        force=args.force,
        deadline_seconds=max(30, int(args.deadline_seconds)),
        per_image_timeout=(int(args.per_image_timeout) if int(args.per_image_timeout) > 0 else None),
    )
    t0 = time.time()
    summaries = runner.run()
    dt = time.time() - t0
    log.info(f"Processed {len(summaries)} images in {dt:.1f}s")
    log.info(f"Artifacts folder: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
