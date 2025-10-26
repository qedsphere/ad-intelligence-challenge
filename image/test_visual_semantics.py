#Run with pytest -v image/test_visual_semantics.py
import json
import numpy as np
import pytest

from image import visual_semantics as vs


# -----------------------
# Minimal fakes & helpers
# -----------------------
class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        # numpy arrays to mimic ultralytics tensors
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=int)

class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

class FakeYOLOModel:
    def __init__(self, results, names=None):
        self._results = results  # list[FakeResult]
        self.names = names or {}

    def predict(self, *args, **kwargs):
        return self._results

class FakeTensor:
    """Tensor-ish object compatible with _tensor_to_list and demo CLIP code."""
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=float)

    # Methods used by _tensor_to_list
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self._arr

    # Extra stubs that may be seen in code paths
    def unsqueeze(self, _): return self
    def to(self, _): return self

class FakeCLIPModel:
    def __init__(self, emb_dim=4):
        self.emb_dim = emb_dim

    def encode_image(self, _img_tensor):
        # return a fixed-length embedding
        return FakeTensor([0.1] * self.emb_dim)

    def encode_text(self, _tokens):
        # not actually used in our monkeypatched classification path
        # but called during text_feats preparation; just return a dummy tensor
        return FakeTensor(np.ones((8, self.emb_dim)))  # shape arbitrary

def fake_preprocess(_pil_img):
    # Return a FakeTensor that supports unsqueeze() and to()
    return FakeTensor(np.ones((3, 224, 224)))

def make_dummy_image(h=64, w=64):
    # float [0,1], HxWx3
    return np.zeros((h, w, 3), dtype=float)


# =======================
#          TESTS
# =======================

def test_no_dependencies(monkeypatch):
    # Force both unavailable
    monkeypatch.setattr(vs, "_YOLO_AVAILABLE", False)
    monkeypatch.setattr(vs, "_CLIP_AVAILABLE", False)

    out = vs.extract(make_dummy_image(), "img_00")

    assert out["image_id"] == "img_00"
    assert out["detections"] == []
    assert out["global_embedding"] is None
    assert out["product_category"]["from"] is None
    assert out["status"] == "not_implemented"
    # JSON serializable
    json.dumps(out)


def test_yolo_only_clip_missing(monkeypatch):
    # YOLO on, CLIP off → boxes present, no embeddings, product_category remains None
    monkeypatch.setattr(vs, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(vs, "_CLIP_AVAILABLE", False)

    boxes = FakeBoxes(
        xyxy=[[5, 5, 25, 25], [30, 10, 50, 30]],
        conf=[0.92, 0.88],
        cls=[1, 2],
    )
    fake_res = FakeResult(boxes)
    fake_yolo = FakeYOLOModel([fake_res], names={1: "product", 2: "logo"})
    monkeypatch.setattr(vs, "_init_yolo", lambda: fake_yolo)

    out = vs.extract(make_dummy_image(), "img_01")

    assert out["status"] == "ok"
    assert len(out["detections"]) == 2
    assert out["global_embedding"] is None
    assert out["product_category"]["from"] is None
    assert "spatial" in out and len(out["spatial"]["pairs"]) == 1
    json.dumps(out)


def test_yolo_and_clip_clip_crop_category(monkeypatch):
    monkeypatch.setattr(vs, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(vs, "_CLIP_AVAILABLE", True)

    boxes = FakeBoxes(xyxy=[[5, 5, 40, 40]], conf=[0.9], cls=[1])
    fake_res = FakeResult(boxes)
    fake_yolo = FakeYOLOModel([fake_res], names={1: "product"})
    monkeypatch.setattr(vs, "_init_yolo", lambda: fake_yolo)

    fake_clip = FakeCLIPModel(emb_dim=3)
    monkeypatch.setattr(vs, "_init_clip",
        lambda: (fake_clip, fake_preprocess, lambda prompts: "TOKENS", None, None)
    )

    # Ensure text feature normalization doesn't throw (so text_feats != None)
    monkeypatch.setattr(vs, "_normalize_feats", lambda x: x)

    # Bypass real scene zero-shot
    monkeypatch.setattr(vs, "_zero_shot_image_labels",
        lambda *args, **kwargs: [{"label": "studio product shot", "score": 0.9}]
    )

    # Force per-crop classification outcome
    def fake_classify_crop(*args, **kwargs):
        topk = [
            {"label": "bagel", "score": 0.70},
            {"label": "donut", "score": 0.20},
            {"label": "coffee", "score": 0.10},
        ]
        return topk, 0.70, 0.20
    monkeypatch.setattr(vs, "_classify_crop", fake_classify_crop)

    out = vs.extract(make_dummy_image(64, 64), "img_02")

    assert out["status"] == "ok"
    assert isinstance(out["global_embedding"], list) and len(out["global_embedding"]) == 3
    assert len(out["detections"]) == 1
    det = out["detections"][0]
    assert "embedding" in det and isinstance(det["embedding"], list)
    assert "top_labels" in det and det["top_labels"][0]["label"] == "bagel"
    assert out["scene_labels"] and out["scene_labels"][0]["label"] == "studio product shot"
    assert out["product_category"]["from"] == "clip_crop"
    assert out["product_category"]["labels"][0]["label"] == "bagel"
    json.dumps(out)



def test_clip_only_clip_image_category(monkeypatch):
    # No YOLO, CLIP on → whole-image zero-shot for product category
    monkeypatch.setattr(vs, "_YOLO_AVAILABLE", False)
    monkeypatch.setattr(vs, "_CLIP_AVAILABLE", True)

    fake_clip = FakeCLIPModel(emb_dim=2)
    monkeypatch.setattr(vs, "_init_clip",
        lambda: (fake_clip, fake_preprocess, lambda prompts: "TOKENS", None, None)
    )

    # Scene labels not important here; product via clip_image:
    def fake_zero_shot_image_labels(_clip_model, _preprocess, _tokenizer, _device, _torch_mod, _pil_img, labels):
        # Called twice in code: once for scene (_SCENE_LABELS) and maybe for product labels
        if "breakfast scene" in labels:
            return [{"label": "breakfast scene", "score": 0.6}]
        # product labels path
        return [{"label": "bagel", "score": 0.7}, {"label": "coffee", "score": 0.2}, {"label": "donut", "score": 0.1}]
    monkeypatch.setattr(vs, "_zero_shot_image_labels", fake_zero_shot_image_labels)

    out = vs.extract(make_dummy_image(), "img_03")

    assert out["status"] in ("ok", "partial")  # ok if embedding produced
    assert out["detections"] == []
    assert out["global_embedding"] is not None
    assert out["product_category"]["from"] == "clip_image"
    assert out["product_category"]["labels"][0]["label"] == "bagel"
    json.dumps(out)
