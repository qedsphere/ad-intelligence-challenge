"""
Utility functions for image processing and feature extraction
"""

import numpy as np
from typing import Dict, Any


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image array to [0, 1] range"""
    if image.max() > 1.0:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def get_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """Get basic statistics about an image"""
    return {
        'shape': image.shape,
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': float(np.min(image)),
        'max': float(np.max(image))
    }

