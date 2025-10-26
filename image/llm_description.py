# image/llm_description.py
"""
LLM description via Gemini (proxied through Lava)
- Input: numpy image array in [0,1], shape (H, W, 3)
- Output: very short demographic + emotional blurb (safe, neutral)
Env:
  - LAVA_SECRET_KEY (required)
  - LAVA_CONNECTION_SECRET (recommended; should point to your Google/Gemini connection in Lava)
  - LAVA_PRODUCT_SECRET (optional)
  - LAVA_BASE_URL (optional; defaults to https://api.lavapayments.com/v1)
  - GEMINI_MODEL (optional; defaults to gemini-1.5-flash-latest)
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import requests
from PIL import Image


# ----------------------------
# Lava config
# ----------------------------

@dataclass
class LavaConfig:
    secret_key: str = os.getenv("LAVA_SECRET_KEY", "")
    connection_secret: Optional[str] = os.getenv("LAVA_CONNECTION_SECRET")
    product_secret: Optional[str] = os.getenv("LAVA_PRODUCT_SECRET")
    base_url: str = os.getenv("LAVA_BASE_URL", "https://api.lavapayments.com/v1")

    def forward_auth_token(self) -> str:
        if not self.secret_key:
            raise EnvironmentError("LAVA_SECRET_KEY is required.")
        payload = {
            "secret_key": self.secret_key,
            "connection_secret": self.connection_secret,
            "product_secret": self.product_secret,
        }
        # Strip Nones to keep token compact
        payload = {k: v for k, v in payload.items() if v}
        token = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
        return f"Bearer {token}"


# ----------------------------
# Utilities
# ----------------------------

def _np_to_jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    """Convert float [0,1] HxWx3 to JPEG bytes."""
    if image.dtype != np.uint8:
        arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = image
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _prompt(image_id: str) -> str:
    # Keep it short, safe, and non-sensitive
    return (
        "In ≤ 18 words, write a neutral, non-sensitive blurb about the likely audience demographic "
        "and emotional tone of this advertisement. Avoid protected attributes and proper names. "
        f"Tag: [{image_id}]"
    )


# ----------------------------
# Gemini via Lava
# ----------------------------

def _call_gemini_via_lava(image_jpeg: bytes, model: str, lava: LavaConfig) -> str:
    b64_img = base64.b64encode(image_jpeg).decode("utf-8")

    # Google Generative Language API: generateContent (vision with inline bytes)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_img,
                        }
                    },
                    {"text": "Respond with only the blurb, no extra prose."},
                    {"text": _prompt(image_id="")},
                ]
            }
        ]
    }

    forward_url = f"{lava.base_url}/forward"
    params = {
        "u": f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    }
    headers = {
        "Authorization": lava.forward_auth_token(),
        "Content-Type": "application/json",
    }

    resp = requests.post(forward_url, params=params, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    provider_data = data.get("data", data)

    # Typical Gemini response parsing
    try:
        txt = provider_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return txt
    except Exception as e:
        raise RuntimeError(f"Gemini via Lava: could not parse response: {e}")


# ----------------------------
# Public API
# ----------------------------

def extract(image: np.ndarray, image_id: str) -> Dict[str, Any]:
    """
    Returns:
        {
          "status": "ok",
          "image_id": <id>,
          "provider": "gemini",
          "model": <model_name>,
          "blurb": <short string>
        }
        or {"status": "error", "image_id": <id>, "error": "..."}
    """
    lava = LavaConfig()
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

    try:
        jpeg = _np_to_jpeg_bytes(image)
        blurb = _call_gemini_via_lava(jpeg, model, lava)
        # Ensure ultra-short, single-line
        blurb = " ".join(blurb.split())
        if "[" not in blurb and image_id:
            blurb = f"{blurb} [{image_id}]"
        return {
            "status": "ok",
            "image_id": image_id,
            "provider": "gemini",
            "model": model,
            "blurb": blurb[:240],
        }
    except Exception as e:
        return {
            "status": "error",
            "image_id": image_id,
            "provider": "gemini",
            "model": model,
            "error": str(e),
        }
