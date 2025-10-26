
# 🧠 Ad Intelligence — Image Feature Extraction Runner

This module (`image/extraction.py`) runs the full **Ad Intelligence** batch extraction pipeline. It standardizes images, extracts visual and attention-based features, and produces human-readable artifacts for each image.

---

## 🚀 Overview

The extraction process has **two main stages**, executed in parallel:

1. **Stage A – Standardization:**  
   Loads each input image, resizes/pads it to 1024×1024 RGB float32 [0–1], and writes standardized versions.

2. **Stage B – Feature Extraction:**  
   For each standardized image, runs a feature extractor suite:
   - **Visual Semantics:** YOLO (detections), CLIP (embeddings + scene labels), OCR, and product category.
   - **Attention Map:** TPPGaze simulated gaze heatmaps, peaks, and entropy.
   - **LLM Description (optional):** lightweight textual summary (placeholder).

The system is **deadline- and timeout-aware**, allowing a global runtime budget and per-image timeouts.

---

## 🧩 CLI Usage

```bash
python -m image.extraction   --images_dir ads/images   --out_dir extracted_features   --size 1024 1024   --std_workers 6   --ext_workers auto   --deadline_seconds 300   --per_image_timeout 120   [--force]
```

### Parameters

| Flag | Description |
|------|--------------|
| `--images_dir` | Folder containing input ad images. |
| `--out_dir` | Where to save standardized images and extracted features. |
| `--size` | Target width and height (default 1024×1024). |
| `--std_workers` | Parallel workers for image standardization. |
| `--ext_workers` | Parallel workers for feature extraction (`auto` = detect). |
| `--deadline_seconds` | Global time budget for entire batch. |
| `--per_image_timeout` | Optional timeout per image (0 = disabled). |
| `--force` | Re-run even if outputs already exist. |

---

## ⚙️ How It Works

### **1. Stage A — Standardization**

Each image is:
- Opened and converted to RGB.  
- Resized and padded to maintain aspect ratio.  
- Saved as `<image_id>_standardized.png`.  
- Metadata (size, bytes, format, etc.) recorded in a `Meta` object.

Returns:  
A payload including the standardized PIL image, NumPy array, and metadata.

### **2. Stage B — Extraction**

The standardized image flows immediately into extraction:

#### a. Visual Semantics

```python
vs_out = VS.extract(std_arr, image_id)
```

Extracts:
- **YOLO detections** (`detections` list)
- **CLIP scene labels** (`scene_labels`)
- **Product category** (via CLIP or FAISS)
- **OCR text** (`ocr`)

Artifacts:
```
*_visual_semantics.json
*_detections.png
*_ocr.txt
```

#### b. Attention Map (TPPGaze)

```python
am_out = AM.extract(std_arr, image_id)
```

Generates:
- Heatmap (.npz)
- Overlay visualization (.png)
- Attention metrics (entropy, peaks, scanpaths)

Artifacts:
```
*_attention_heatmap.npz
*_attention_overlay.png
*_attention_map.json
```

#### c. LLM Description (optional)

If available, calls `LLM.extract(std_arr, image_id)` to produce:
```
*_llm_description.json
```

#### d. Summary Assembly

All components are combined into:
```
*_summary.json
```

Example structure:
```json
{
  "image_id": "ad_001",
  "metadata": {...},
  "files": {...},
  "quicklook": {
    "num_detections": 3,
    "scene_top": [{"label": "breakfast table", "score": 0.74}],
    "product_category": {...},
    "ocr_snippet": "Freshly baked bagels...",
    "attention_entropy": 2.19
  }
}
```

---

## 📦 Output Structure

Example layout under `extracted_features/`:

```
extracted_features/
└── ad_001/
    ├── ad_001_standardized.png
    ├── ad_001_detections.png
    ├── ad_001_ocr.txt
    ├── ad_001_visual_semantics.json
    ├── ad_001_attention_heatmap.npz
    ├── ad_001_attention_overlay.png
    ├── ad_001_attention_map.json
    ├── ad_001_llm_description.json
    └── ad_001_summary.json
```

And a project-level index file:
```
extracted_features/features_index.csv
```

### CSV Columns

| Column | Description |
|--------|--------------|
| image_id | Unique ID for image |
| original_path | Original file path |
| original_size | (W,H) of input |
| standardized_size | (H,W) of resized image |
| file_size_bytes | File size in bytes |
| format | Original image format |
| processing_time_ms | Total extraction time |
| num_detections | YOLO box count |
| product_category_from | Source of product label |
| product_category_top | Top predicted label |
| scene_top_label | CLIP top scene label |
| scene_top_score | CLIP confidence |
| ocr_chars | OCR text length |
| attention_entropy | Attention map entropy |

---

## 🧠 Data Flow Summary

| Feature | Function | Module | Artifacts |
|----------|-----------|---------|------------|
| YOLO + CLIP + OCR + Category | `VS.extract()` | `image/visual_semantics.py` | `_visual_semantics.json`, `_detections.png`, `_ocr.txt` |
| Attention / Gaze | `AM.extract()` | `image/attention_map.py` | `_attention_heatmap.npz`, `_attention_overlay.png`, `_attention_map.json` |
| Text Summary | `LLM.extract()` | `image/llm_description.py` | `_llm_description.json` |
| Aggregation | `extract_one()` | `image/extraction.py` | `_summary.json`, `features_index.csv` |

---

## ✅ Summary

- All **feature extraction** occurs inside `extract_one()`.
- Visual + attention outputs are **fully implemented and saved**.
- LLM description is optional but will automatically integrate once implemented.
- The pipeline is **parallel, deadline-aware**, and produces ready-to-visualize artifacts for the web UI.

