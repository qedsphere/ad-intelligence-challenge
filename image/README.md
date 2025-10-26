# Visual Semantics

Feature extraction for ad images using fast, pretrained components with graceful fallbacks.

**Outputs**

- **detections** (YOLO): `bbox`, `confidence`, `class_id`, `class_name`
- **global_embedding** (CLIP): dense vector for retrieval/semantics
- **scene_labels** (zero-shot CLIP): top scene descriptions with scores
- **spatial** relations: `left_of`, `above`, `overlaps` among boxes
- **product_category**: robust label via FAISS text index or zero-shot CLIP
- **ocr** (optional): full ad text + spans (`bbox`, `text`, `conf`) via PaddleOCR or Tesseract

Everything is **lazy-loaded** and optional. If a dependency is missing, the rest still works.

---

## Directory layout

```
image/
  visual_semantics.py      # the extractor
  scripts/
    demo_visual_semantics.py
```

---

## Installation

> Create and activate a virtual environment first.

```bash
pip install --upgrade pip
pip install   numpy Pillow opencv-python   torch open_clip_torch ultralytics   faiss-cpu   paddlepaddle paddleocr   pytesseract matplotlib
```

**Notes**

- **Device:** The code auto-selects CUDA → MPS (Apple Silicon) → CPU when available.
- **OCR:** Only one engine is needed. The module tries **PaddleOCR** first, then **Tesseract**.
- **FAISS:** Optional. If missing, product category falls back to zero-shot CLIP.

---

## (Optional) FAISS open-vocabulary index

Place these files where you run the code (or change the paths inside `visual_semantics.py`):

- `text_labels.index` – FAISS index of text embeddings
- `text_labels.json` – parallel list of label strings

When present, FAISS is used for robust open-vocabulary product categorization.

---

## Run the demo

```bash
python -m image.scripts.demo_visual_semantics   --image ads/images/i0006.png   --size 1024 1024   --out viz_out
```

Artifacts saved to `viz_out/`:

- `*_visual_semantics.json` – full feature JSON
- `*_detections.png` – YOLO boxes overlay (if any)
- `*_saliency.png` – simple gradient saliency overlay
- `*_palette.png` – dominant color palette strip

> The demo **saves files**; it does not open GUI windows.

---

## Programmatic usage

```python
import numpy as np
from PIL import Image
from image import visual_semantics as vs

pil = Image.open("ads/images/i0006.png").convert("RGB").resize((1024, 1024))
arr = (np.asarray(pil).astype(np.float32) / 255.0)  # HxWx3 in [0,1]

result = vs.extract(arr, image_id="i0006")
print(result["status"])             # "ok" when useful output was produced
print(result["product_category"])   # {"from": "...", "labels": [{"label": "...", "score": ...}, ...]}
```

---

## Output schema (abridged)

```json
{
  "image_id": "i0006",
  "status": "ok",
  "errors": [],
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.92,
      "class_id": 39,
      "class_name": "bottle",
      "embedding": [ ... ],
      "top_labels": [
        {"label": "bagel", "score": 0.31},
        {"label": "donut", "score": 0.22}
      ]
    }
  ],
  "global_embedding": [ ... ],
  "scene_labels": [
    {"label": "breakfast scene", "score": 0.18},
    {"label": "studio product shot", "score": 0.16}
  ],
  "spatial": { "pairs": [ { "subject_index": 0, "object_index": 1, "relations": ["left_of","above"] } ] },
  "product_category": {
    "from": "faiss_image | clip_image | faiss_crop | clip_crop",
    "labels": [
      {"label": "bagel", "score": 0.31},
      {"label": "pretzel", "score": 0.24},
      {"label": "donut", "score": 0.18}
    ]
  },
  "ocr": {
    "status": "ok | error | unavailable",
    "engine": "paddle | tesseract | null",
    "full_text": "PRETZEL BAGEL DOUBLE DIPPER NEW!",
    "spans": [
      {"bbox": [x1, y1, x2, y2], "text": "PRETZEL", "conf": 0.98, "source": "full_image"}
    ]
  }
}
```

---

## Configuration knobs (inside `visual_semantics.py`)

- YOLO weights: `_YOLO_WEIGHTS = "yolov8n.pt"`
- CLIP model: `_CLIP_MODEL = "ViT-B-32"`, `_CLIP_PRETRAINED = "openai"`
- Temperature for zero-shot softmax: `_TEMPERATURE = 0.07`
- Product gating: `_TOP1_MIN = 0.10`, `_MARGIN_MIN = 0.03`
- FAISS files: `_FAISS_INDEX_PATH`, `_FAISS_LABELS_PATH`
- Scene & product mini-banks: `_SCENE_LABELS`, `_FOOD_LABELS`, `_ELECTRONICS_LABELS`, `_META_LABELS`

---

## How the pipeline decides

1. **YOLO** (if available) → candidate boxes.
2. **OCR** on full image (and per YOLO crop) → `ocr.full_text` + spans.
3. **CLIP** encodes whole image and crops.
4. **Product category**
   - Use **FAISS** (if index present) on crops; fall back to image.
   - Else use **zero-shot CLIP** over a scene-aware label bank.
5. **Spatial** relations from box geometry.
6. **Status** is `"ok"` if any of {detections, global embedding, OCR text} exist.

---

## Troubleshooting

- **`No module named 'image'`**  
  Run from repo root with `-m`:  
  `python -m image.scripts.demo_visual_semantics ...`

- **No boxes in detections**  
  YOLO may lack that category (e.g., “bagel” isn’t in COCO).  
  Product category can still be correct via CLIP/FAISS. Consider adding a text-prompted detector as an optional future fallback.

- **Slow on first run**  
  Weights download the first time; subsequent runs are faster.

- **Apple Silicon**  
  If your Torch has MPS, it’ll be selected automatically. The demo prints the device summary.

---

## Licenses

This module depends on third-party libraries/models (Ultralytics YOLO, OpenCLIP, PaddleOCR, Tesseract, FAISS). Ensure your usage complies with their respective licenses.