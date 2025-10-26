#!/usr/bin/env python3
"""
Unified CLI entrypoint. Delegates to src.pipeline.run_pipeline.
"""

import argparse

from src import config as cfg
from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Audio Feature Extraction Pipeline")
    p.add_argument("--clean", action="store_true", help="Delete output directory before run")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate WAVs/stems")
    p.add_argument(
        "--workers",
        type=int,
        default=cfg.DEFAULT_CPU_WORKERS,
        help="CPU worker threads (default: %(default)s)",
    )
    p.add_argument(
        "--model-size",
        default=cfg.DEFAULT_STT_MODEL,
        help="faster-whisper model size (tiny/base/small/...)",
    )
    p.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for faster-whisper (int8,int8_float16,float16,float32)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process (for quick tests)",
    )
    p.add_argument(
        "--include-segments",
        action="store_true",
        help="Include per-segment timestamps in output (speech)",
    )
    p.add_argument("--progress", dest="progress", action="store_true", help="Show progress bars (default)")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bars")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.set_defaults(progress=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)

