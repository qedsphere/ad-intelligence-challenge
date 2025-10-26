"""
Visual Semantics Feature Extraction (YOLO for boxes, CLIP zero-shot for labels)

Returns:
- detections: YOLO proposals (bbox, conf, class_id, class_name) for overlays/spatial
- global_embedding: CLIP embedding for the full image
- scene_labels: CLIP zero-shot labels for whole image
- spatial: simple relations (left_of/right_of/above/below/overlaps)
- product_category: robust category via CLIP zero-shot
  - from: "clip_crop" (per-crop) when confident, else "clip_image" (whole image)

All heavy deps are optional; function degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional / lazy dependencies
# -----------------------------------------------------------------------------
_YOLO_AVAILABLE = False
_CLIP_AVAILABLE = False

_yolo_model = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_torch = None

# Small & fast defaults
_YOLO_WEIGHTS = "yolov8n.pt"
_CLIP_MODEL = "ViT-B-32"
_CLIP_PRETRAINED = "openai"

# Scene labels
_SCENE_LABELS = [
    "food photography", "breakfast scene", "kitchen counter",
    "outdoor picnic", "restaurant table", "abstract background",
    "studio product shot", "lifestyle photo", "text-heavy ad"
]

# Product label banks (scene-aware selection uses these)
_FOOD_LABELS = [
    "bagel", "donut", "pretzel", "sandwich", "burger", "fries", "salad",
    "soup", "pizza", "cake", "coffee", "tea", "juice", "soda",
    "sauce cup", "dipping sauce", "plate", "tray", "napkin", "cutting board",
]
_ELECTRONICS_LABELS = ["phone", "laptop", "watch", "headphones", "game controller"]
_META_LABELS = ["brand logo", "price tag", "discount badge", "coupon", "app icon", "QR code"]

# Gating/softmax temperature for CLIP zero-shot
_TEMPERATURE = 0.07
_TOP1_MIN = 0.10
_MARGIN_MIN = 0.03

# -----------------------------------------------------------------------------
# Dependency probing
# -----------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except Exception as e:
    logger.debug("ultralytics (YOLO) not available: %s", e)

try:
    import open_clip  # type: ignore
    import torch  # type: ignore
    _torch = torch
    _CLIP_AVAILABLE = True
except Exception as e:
    logger.debug("open_clip/torch not available: %s", e)


def _device():
    if _torch is None:
        return None
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    try:
        if getattr(_torch.backends, "mps", None) is not None and _torch.backends.mps.is_available():
            return _torch.device("mps")
    except Exception:
        pass
    return _torch.device("cpu")


def _init_yolo() -> Optional[Any]:
    global _yolo_model
    if not _YOLO_AVAILABLE:
        return None
    if _yolo_model is None:
        try:
            _yolo_model = YOLO(_YOLO_WEIGHTS)
            logger.info("Loaded YOLO weights: %s", _YOLO_WEIGHTS)
        except Exception as e:
            logger.warning("Failed to load YOLO weights %s: %s", _YOLO_WEIGHTS, e)
            _yolo_model = None
    return _yolo_model


def _init_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if not _CLIP_AVAILABLE:
        return None, None, None, None, None
    if _clip_model is None:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL, pretrained=_CLIP_PRETRAINED
            )
            tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)
            dev = _device()
            if dev is not None:
                model = model.to(dev)
            model.eval()
            _clip_model, _clip_preprocess, _clip_tokenizer = model, preprocess, tokenizer
            logger.info("Loaded CLIP: %s (%s)", _CLIP_MODEL, _CLIP_PRETRAINED)
        except Exception as e:
            logger.warning("Failed to init CLIP %s/%s: %s", _CLIP_MODEL, _CLIP_PRETRAINED, e)
            _clip_model = None
            _clip_preprocess = None
            _clip_tokenizer = None
    return _clip_model, _clip_preprocess, _clip_tokenizer, _device(), _torch

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_pil_uint8(image: np.ndarray):
    from PIL import Image  # local import
    arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def _tensor_to_list(t) -> Optional[List[float]]:
    try:
        return t.detach().cpu().numpy().astype(np.float32).tolist()
    except Exception:
        try:
            return t.cpu().numpy().astype(np.float32).tolist()
        except Exception:
            return None

def _normalize_feats(x):
    return x / x.norm(dim=-1, keepdim=True)

def _select_label_bank(scene_labels: List[Dict[str, Any]]) -> List[str]:
    text = " ".join(lbl.get("label", "") for lbl in scene_labels) if scene_labels else ""
    food_hint = any(k in text for k in ["food", "breakfast", "restaurant", "kitchen"])
    if food_hint:
        return _FOOD_LABELS + _META_LABELS
    return _FOOD_LABELS + _ELECTRONICS_LABELS + _META_LABELS

def _zero_shot_image_labels(clip_model, preprocess, tokenizer, device, torch_mod, pil_img, labels: List[str]) -> List[Dict[str, Any]]:
    if any(x is None for x in (clip_model, preprocess, tokenizer)):
        return []
    img = preprocess(pil_img).unsqueeze(0)
    if device is not None:
        img = img.to(device)
    cm = torch_mod.no_grad() if (torch_mod is not None and hasattr(torch_mod, "no_grad")) else nullcontext()
    with cm:
        img_feat = _normalize_feats(clip_model.encode_image(img))
    prompts = [f"a product photo of a {lbl}" for lbl in labels]
    tokens = tokenizer(prompts)
    if device is not None:
        tokens = tokens.to(device)
    with cm:
        text_feats = _normalize_feats(clip_model.encode_text(tokens))
        sims = (img_feat @ text_feats.T).squeeze(0)
        probs = (sims / _TEMPERATURE).softmax(dim=0)
    probs_np = probs.detach().cpu().numpy()
    order = np.argsort(-probs_np)
    return [{"label": labels[i], "score": float(probs_np[i])} for i in order[:5]]

def _classify_crop(clip_model, preprocess, tokenizer, device, torch_mod, crop_img, text_feats, labels: List[str]) -> Tuple[List[Dict[str, Any]], float, float]:
    cm = torch_mod.no_grad() if (torch_mod is not None and hasattr(torch_mod, "no_grad")) else nullcontext()
    t = preprocess(crop_img).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    with cm:
        img_feat = _normalize_feats(clip_model.encode_image(t))
        sims = (img_feat @ text_feats.T).squeeze(0)
        probs = (sims / _TEMPERATURE).softmax(dim=0)
    probs_np = probs.detach().cpu().numpy().astype("float32")
    order = np.argsort(-probs_np)
    topk = [{"label": labels[i], "score": float(probs_np[i])} for i in order[:3]]
    p1 = float(probs_np[order[0]]) if probs_np.size else 0.0
    p2 = float(probs_np[order[1]]) if probs_np.size > 1 else 0.0
    return topk, p1, p2

def _basic_spatial_relations(dets: List[Dict[str, Any]]) -> Dict[str, Any]:
    rels = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a = dets[i]["bbox"]; b = dets[j]["bbox"]
            axc, ayc = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
            bxc, byc = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            xI = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            yI = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = xI * yI
            a_area = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
            b_area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            union = a_area + b_area - inter
            iou = inter / union if union > 0 else 0.0
            relation = []
            relation.append("left_of" if axc < bxc else "right_of")
            relation.append("above" if ayc < byc else "below")
            if iou > 0.05:
                relation.append("overlaps")
            rels.append({"subject_index": i, "object_index": j, "relations": relation})
    return {"pairs": rels}

# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------
def extract(image: np.ndarray, image_id: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "image_id": image_id,
        "status": "not_implemented",
        "errors": [],
        "detections": [],
        "global_embedding": None,
        "scene_labels": [],
        "spatial": {"pairs": []},
        "product_category": {"from": None, "labels": []},
    }

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        results["errors"].append({"input_error": "image must be HxWx3 numpy array in [0,1]"})
        return results

    # Convert to PIL
    try:
        pil_img = _to_pil_uint8(image)
    except Exception as e:
        results["errors"].append({"convert_error": str(e)})
        return results

    # ------------------ YOLO: boxes only ------------------
    try:
        yolo = _init_yolo()
        if yolo is not None:
            np_img = np.array(pil_img)
            yres = yolo.predict(np_img, conf=0.30, iou=0.50, verbose=False)
            names = getattr(yolo, "names", {}) or {}
            if len(yres) > 0 and hasattr(yres[0], "boxes"):
                boxes = yres[0].boxes
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                for bbox, conf, cid in zip(xyxy, confs, cls_ids):
                    x1, y1, x2, y2 = map(float, bbox[:4])
                    cid_int = int(cid)
                    cname = names.get(cid_int, str(cid_int))
                    results["detections"].append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": cid_int,
                        "class_name": cname
                    })
    except Exception as e:
        logger.debug("YOLO detection failed: %s", e)
        results["errors"].append({"yolo_error": str(e)})

    # ------------------ CLIP: embeddings, scene, product ------------------
    try:
        clip_model, preprocess, tokenizer, device, torch_mod = _init_clip()
        if clip_model is not None and preprocess is not None:
            # Global embedding
            img_tensor = preprocess(pil_img).unsqueeze(0)
            if device is not None:
                img_tensor = img_tensor.to(device)
            cm = torch_mod.no_grad() if (torch_mod is not None and hasattr(torch_mod, "no_grad")) else nullcontext()
            with cm:
                g_emb = clip_model.encode_image(img_tensor)
            results["global_embedding"] = _tensor_to_list(g_emb)

            # Scene labels
            try:
                results["scene_labels"] = _zero_shot_image_labels(
                    clip_model, preprocess, tokenizer, device, torch_mod, pil_img, _SCENE_LABELS
                )
            except Exception as e:
                results["errors"].append({"clip_scene_error": str(e)})

            # Prepare product text features with scene-aware label bank
            best_crop = None
            try:
                label_bank = _select_label_bank(results.get("scene_labels", []))
                prompts = [f"a product photo of a {lbl}" for lbl in label_bank]
                tokens = tokenizer(prompts)
                if device is not None:
                    tokens = tokens.to(device)
                with cm:
                    text_feats = _normalize_feats(clip_model.encode_text(tokens))
            except Exception as e:
                text_feats = None
                label_bank = []
                results["errors"].append({"clip_textfeat_error": str(e)})

            # Per-detection embeddings + per-crop zero-shot classification
            if results["detections"] and text_feats is not None and label_bank:
                for det in results["detections"]:
                    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(image.shape[1] - 1, x2)
                    y2 = min(image.shape[0] - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        det["embedding"] = None
                        det["top_labels"] = []
                        continue

                    crop = pil_img.crop((x1, y1, x2, y2))

                    # per-crop embedding
                    try:
                        t = preprocess(crop).unsqueeze(0)
                        if device is not None:
                            t = t.to(device)
                        with cm:
                            c_emb = clip_model.encode_image(t)
                        det["embedding"] = _tensor_to_list(c_emb)
                    except Exception as e:
                        det["embedding"] = None
                        results["errors"].append({"clip_crop_embed_error": str(e)})

                    # per-crop zero-shot labels
                    try:
                        topk, p1, p2 = _classify_crop(
                            clip_model, preprocess, tokenizer, device, torch_mod, crop, text_feats, label_bank
                        )
                        det["top_labels"] = topk
                        margin = p1 - p2
                        candidate = {"topk": topk, "margin": margin, "p1": p1}
                        if (best_crop is None) or (margin > best_crop["margin"]):
                            best_crop = candidate
                    except Exception as e:
                        det["top_labels"] = []
                        results["errors"].append({"clip_crop_class_error": str(e)})

            # Product category resolution (gated; fallback to whole image)
            if best_crop is not None and (best_crop["p1"] >= _TOP1_MIN and best_crop["margin"] >= _MARGIN_MIN):
                results["product_category"] = {"from": "clip_crop", "labels": best_crop["topk"]}
            else:
                try:
                    # if label_bank failed above, use a default comprehensive bank
                    lb = label_bank if label_bank else (_FOOD_LABELS + _ELECTRONICS_LABELS + _META_LABELS)
                    zs = _zero_shot_image_labels(
                        clip_model, preprocess, tokenizer, device, torch_mod, pil_img, lb
                    )
                    results["product_category"] = {"from": "clip_image", "labels": zs[:3]}
                except Exception as e:
                    results["errors"].append({"clip_product_error": str(e)})
    except Exception as e:
        logger.debug("CLIP pipeline failed: %s", e)
        results["errors"].append({"clip_error": str(e)})

    # ------------------ Spatial relations ------------------
    try:
        if results["detections"]:
            results["spatial"] = _basic_spatial_relations(results["detections"])
        else:
            results["spatial"] = {"pairs": []}
    except Exception as e:
        results["errors"].append({"spatial_error": str(e)})

    # ------------------ Final status ------------------
    produced_any = bool(results["detections"]) or (results.get("global_embedding") is not None)
    if produced_any:
        results["status"] = "ok"
    else:
        results["status"] = "partial" if results["errors"] else "not_implemented"

    return results
