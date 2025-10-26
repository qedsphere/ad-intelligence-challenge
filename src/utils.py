"""
Utility functions for the audio feature extraction pipeline
"""

import os
import json
import logging


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    """Save data to JSON file with pretty formatting."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info("Saved: %s", filepath)



