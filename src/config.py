"""
Central configuration for the audio feature extraction pipeline.

All modules should import constants from here to avoid duplication.
"""

# Data locations
VIDEO_DIR = "ads/videos/"
OUTPUT_DIR = "output"
OUTPUT_AUDIO_DIR = f"{OUTPUT_DIR}/audio"
OUTPUT_VOCALS_DIR = f"{OUTPUT_DIR}/vocals"
OUTPUT_JSON = f"{OUTPUT_DIR}/features.json"

# Sample rates
SPEECH_SR = 16000
MUSIC_SR = 22050

# Separation / Demucs
DEMUCS_MODEL = "mdx_q"
DEMUCS_OVERLAP = 0.25

# STT defaults
DEFAULT_STT_BACKEND = "faster"  # options: "faster", "openai"
DEFAULT_STT_MODEL = "small"

# Concurrency
DEFAULT_CPU_WORKERS = 4

# VAD / hallucination filtering
NO_SPEECH_THRESHOLD = 0.6


