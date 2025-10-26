# Ad Intelligence - Audio

Audio feature extraction pipeline for ad creatives.

## Quickstart

Requirements:
- Python 3.13
- FFmpeg installed (macOS: `brew install ffmpeg`)

Create venv and install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline (clean + progress):
```bash
./run_pipeline.sh
```

Outputs:
- `output/features.json` (final consolidated results)

Optional flags (if running main.py directly):
- `--workers 6`                       number of CPU workers (default 4)
- `--model-size small`                faster-whisper model size (tiny/base/small/...)
- `--compute-type int8`               faster-whisper compute (int8,int8_float16,float16,float32)
- `--keep-intermediate`               keep WAVs and stems in `output/`
- `--no-progress`                     disable progress bars
- `--log-level DEBUG`                 set logging level
- `--limit 2`                         process only first N videos (smoke test)

## Features Extracted
- Speech-to-text transcription
- Speaker analysis: gender, prosody, emotion (heuristic)
- Text analysis: keywords, entities, CTA, intent (heuristic)
- Music analysis: tempo, energy, mood (heuristic)

## Performance
Optimized with a streaming scheduler and faster-whisper backend.
On M-series Mac, 23 videos process in ~2.2 minutes.


