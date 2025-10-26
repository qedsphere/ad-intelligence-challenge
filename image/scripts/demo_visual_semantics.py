"""
Demo runner for image.visual_semantics.extract()

- Standardizes an input image (keep aspect, pad to target size)
- Calls visual_semantics.extract (your module)
- Prints JSON summary and saves:
    1) standardized image with YOLO detections overlay (if any)
    2) simple saliency heatmap overlay (gradient magnitude)
    3) dominant color palette strip
    4) raw JSON output
    5) OCR overlay image + plaintext dump (if OCR available)

Run:
  python -m image.scripts.demo_visual_semantics \
    --image ads/images/i0006.png \
    --size 1024 1024 \
    --out viz_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Import your extractor
from image import visual_semantics as vs


# ---------------------------
# Utilities
# ---------------------------
def load_and_standardize(path: Path, target_size=(1024, 1024), keep_aspect=True) -> tuple[Image.Image, np.ndarray]:
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
    arr = np.asarray(std_pil).astype(np.float32) / 255.0  # float [0,1], HxWx3
    return std_pil, arr


def kmeans_palette(arr: np.ndarray, k=5, seed=42):
    """Very small k-means for palette extraction."""
    H, W, _ = arr.shape
    X = arr.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(X.shape[0], k, replace=False)].copy()
    labels = np.zeros((X.shape[0],), dtype=np.int32)
    for _ in range(10):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    pops = np.array([(labels == i).sum() for i in range(k)])
    order = np.argsort(-pops)
    return centers[order], (pops[order] / pops.sum())


def palette_strip(colors: np.ndarray, weights: np.ndarray, width=900, height=80) -> Image.Image:
    strip = np.zeros((height, width, 3), dtype=np.float32)
    x = 0
    for c, w in zip(colors, weights):
        wpx = int(round(width * float(w)))
        strip[:, x:x+wpx, :] = c
        x += wpx
    strip = (np.clip(strip, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(strip)


def simple_saliency(arr: np.ndarray) -> np.ndarray:
    """Gradient magnitude of luminance as a quick saliency proxy."""
    Y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    Ypad = np.pad(Y, 1, mode='edge')
    gx = np.zeros_like(Y); gy = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            patch = Ypad[i:i+3, j:j+3]
            gx[i, j] = float((patch * sobel_x).sum())
            gy[i, j] = float((patch * sobel_y).sum())
    mag = np.sqrt(gx**2 + gy**2)
    mmin, mmax = float(mag.min()), float(mag.max())
    if mmax > mmin:
        mag = (mag - mmin) / (mmax - mmin)
    else:
        mag = np.zeros_like(mag)
    return mag


def draw_detections(pil_img: Image.Image, dets: List[Dict]) -> Image.Image:
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f'{d.get("class_name","")} {d.get("confidence",0):.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        if font:
            draw.text((x1 + 4, y1 + 4), label, fill=(0, 255, 0), font=font)
    return out


def draw_ocr_spans(pil_img: Image.Image, spans: List[Dict]) -> Image.Image:
    """Draw OCR bboxes in magenta with text."""
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for s in spans:
        bx1, by1, bx2, by2 = s.get("bbox", [0,0,0,0])
        bx1, by1, bx2, by2 = map(int, [bx1, by1, bx2, by2])
        txt = s.get("text", "")
        draw.rectangle((bx1, by1, bx2, by2), outline=(255, 0, 255), width=2)
        if txt and font:
            draw.text((bx1 + 2, by1 + 2), txt[:32], fill=(255, 0, 255), font=font)
    return out


def to_hex(rgb01):
    r,g,b = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to the input ad image")
    ap.add_argument("--size", nargs=2, type=int, default=(1024, 1024), help="Target size W H")
    ap.add_argument("--out", type=str, default="viz_out", help="Output directory for artifacts")
    args = ap.parse_args()

    in_path = Path(args.image)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    target_size = (args.size[0], args.size[1])

    # 1) Standardize
    std_pil, std_arr = load_and_standardize(in_path, target_size, keep_aspect=True)

    # 2) Run your visual semantics extractor
    image_id = in_path.stem
    result = vs.extract(std_arr, image_id)

    # 3) Save JSON
    json_path = out_dir / f"{image_id}_visual_semantics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved features to {json_path}")

    # 4) Detections overlay (if any)
    det_img = draw_detections(std_pil, result.get("detections", []))
    det_img_path = out_dir / f"{image_id}_detections.png"
    det_img.save(det_img_path)
    print(f"[OK] Saved detections overlay to {det_img_path}")

    # 5) Simple saliency overlay
    sal = simple_saliency(std_arr)
    plt.figure(figsize=(6, 6))
    plt.imshow(std_pil)
    plt.imshow(sal, alpha=0.5, cmap="jet")
    plt.axis("off")
    sal_path = out_dir / f"{image_id}_saliency.png"
    plt.tight_layout(); plt.savefig(sal_path, dpi=150); plt.close()
    print(f"[OK] Saved saliency overlay to {sal_path}")

    # 6) Dominant color palette
    colors, weights = kmeans_palette(std_arr, k=5, seed=42)
    pal_img = palette_strip(colors, weights)
    pal_path = out_dir / f"{image_id}_palette.png"
    pal_img.save(pal_path)
    hexes = [to_hex(c) for c in colors]
    print("[Palette hex -> weight]:")
    for h, w in zip(hexes, weights):
        print(f"  {h}  {float(w):.3f}")
    print(f"[OK] Saved palette strip to {pal_path}")

    # 7) OCR overlay + text dump (if available)
    ocr = result.get("ocr", {}) or {}
    ocr_status = ocr.get("status")
    ocr_engine = ocr.get("engine")
    spans = ocr.get("spans", []) or []
    full_text = (ocr.get("full_text") or "").strip()

    print("\n=== OCR ===")
    print(f"Status: {ocr_status} | Engine: {ocr_engine}")
    if full_text:
        snippet = (full_text[:160] + "...") if len(full_text) > 160 else full_text
        print(f"Text snippet: {snippet}")

        # save full text
        txt_path = out_dir / f"{image_id}_ocr.txt"
        with open(txt_path, "w") as f:
            f.write(full_text)
        print(f"[OK] Saved OCR text to {txt_path}")

    if spans:
        ocr_img = draw_ocr_spans(std_pil, spans)
        ocr_img_path = out_dir / f"{image_id}_ocr.png"
        ocr_img.save(ocr_img_path)
        print(f"[OK] Saved OCR overlay to {ocr_img_path}")

    # 8) Console summary for quick verification
    print("\n=== Summary ===")
    print(f"Status: {result.get('status')}")
    print(f"Detections: {len(result.get('detections', []))}")
    if result.get("scene_labels"):
        top_scene = result["scene_labels"][0]
        print(f"Top scene label: {top_scene['label']} ({top_scene['score']:.2f})")
    if result.get("product_category"):
        pc = result["product_category"]
        src = pc.get("from")
        lbl0 = pc.get("labels", [])[:1]
        print(f"Product category source: {src}, top: {lbl0}")

    # Optional: device info if torch present
    tm = getattr(vs, "_torch", None)
    if tm is not None:
        mps = getattr(getattr(tm.backends, "mps", None), "is_available", lambda: False)()
        print(f"torch version: {tm.__version__}, CUDA: {tm.cuda.is_available()}, MPS: {mps}")


if __name__ == "__main__":
    main()
