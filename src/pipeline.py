"""
Pipeline orchestration for the audio feature extraction.
Main entry calls run_pipeline(args) from this module.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from . import config as cfg
from .audio_extractor import extract_audio_batch
from .source_separator import separate_vocals
from .speech_analyzer import analyze_speech_for_video
from .text_analyzer import analyze_text_for_video
from .music_analyzer import analyze_music_for_video
from .utils import save_json


def build_transcriber(model_name: str, compute_type: str | None = None):
    from .transcriber_fast import (
        transcribe_audio_for_video,
        detect_speech_quick,
        configure,
    )

    configure(model_name=model_name, device="cpu", compute_type=compute_type or "int8")
    return transcribe_audio_for_video, detect_speech_quick


def run_pipeline(args) -> None:
    from tqdm import tqdm  # local import to allow disabling if needed

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    transcribe_audio_for_video, detect_speech_quick = build_transcriber(
        args.model_size, args.compute_type
    )

    start_time = time.time()

    # Clean outputs if requested
    if args.clean and os.path.isdir(cfg.OUTPUT_DIR):
        shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)

    # Discover videos
    video_files = sorted(glob.glob(os.path.join(cfg.VIDEO_DIR, "*.mp4")))
    if args.limit:
        video_files = video_files[: args.limit]
    if not video_files:
        logging.error("No videos found in %s", cfg.VIDEO_DIR)
        return

    logging.info("%s", "=" * 60)
    logging.info("Audio Feature Extraction Pipeline")
    logging.info("%s", "=" * 60)
    logging.info("Found %d videos", len(video_files))

    # Phase 1: Audio Extraction
    audio_results = extract_audio_batch(video_files)
    audio_files = [r['audio_path'] for r in audio_results]
    video_ids = [r['video_id'] for r in audio_results]
    id_to_meta = {r['video_id']: r for r in audio_results}

    # Phase 2: Speech Detection
    speech_detected = {}
    iterator = (
        tqdm(zip(audio_files, video_ids), total=len(video_ids), desc="Detecting speech", unit="video")
        if args.progress else zip(audio_files, video_ids)
    )
    for audio_path, vid in iterator:
        has_speech = detect_speech_quick(audio_path)
        speech_detected[vid] = has_speech
    logging.info("  → Speech detected: %d/%d", sum(speech_detected.values()), len(video_ids))

    # Phase 3: Streaming Processing (GPU + CPU overlap)
    results: dict[str, dict] = {}
    results_lock = Lock()

    progress_bar = None
    if args.progress:
        progress_bar = tqdm(total=len(video_ids), desc="Processing videos", unit="video")

    def process_video_cpu(video_id: str, original_audio_path: str, vocals_path: str | None):
        try:
            analysis_audio = vocals_path if vocals_path and os.path.exists(vocals_path) else original_audio_path
            tx = transcribe_audio_for_video(analysis_audio, video_id)
            speech_analysis = analyze_speech_for_video(
                analysis_audio, video_id, tx.get('has_speech', False), tx.get('word_count', 0)
            )
            text_analysis = analyze_text_for_video(video_id, tx.get('transcription', ''))
            music_analysis = analyze_music_for_video(video_id, original_audio_path, vocals_path, tx)
        except Exception as e:
            tx = {'has_speech': False, 'transcription': '', 'language': 'unknown', 'word_count': 0, 'error': str(e)}
            speech_analysis = {'gender': 'unknown', 'prosody': {}, 'emotion': {'label': 'unknown', 'confidence': 0}}
            text_analysis = {'keywords': [], 'entities': [], 'cta': {'detected': False, 'phrases': []}, 'intent': {'primary': 'unknown', 'confidence': 0}}
            music_analysis = {'has_music': False, 'tempo_bpm': 0.0, 'energy': 'unknown', 'mood': {'mood': 'unknown', 'confidence': 0.0}}

        with results_lock:
            results[video_id] = {
                'transcription': tx,
                'speech_analysis': speech_analysis,
                'text_analysis': text_analysis,
                'music_analysis': music_analysis,
            }
            if progress_bar:
                progress_bar.update(1)

    speech_videos = [(a, v) for a, v in zip(audio_files, video_ids) if speech_detected[v]]
    no_speech_videos = [(a, v) for a, v in zip(audio_files, video_ids) if not speech_detected[v]]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        # GPU separation sequentially; submit CPU work after each
        for audio_path, vid in speech_videos:
            vocals_path = None
            try:
                vocals_path = separate_vocals(audio_path)
            except Exception as e:
                logging.warning("Separation failed for %s: %s", vid, e)
                vocals_path = None
            futures.append(pool.submit(process_video_cpu, vid, audio_path, vocals_path))

        # Non-speech: CPU-only
        for audio_path, vid in no_speech_videos:
            futures.append(pool.submit(process_video_cpu, vid, audio_path, None))

        for f in as_completed(futures):
            _ = f.result()

    if progress_bar:
        progress_bar.close()

    # Phase 4: Combine and save
    features = {}
    def round_float(x):
        try:
            return round(float(x), 3)
        except Exception:
            return x

    for vid in video_ids:
        r = results.get(vid, {})
        tx = r.get('transcription', {})
        sp = r.get('speech_analysis', {})
        txa = r.get('text_analysis', {})
        mu = r.get('music_analysis', {})

        # Refine prosody
        prosody = sp.get('prosody') or {}
        refined_prosody = {}
        if prosody:
            if 'pitch_mean_hz' in prosody:
                refined_prosody['f0_mean_hz'] = round_float(prosody['pitch_mean_hz'])
            if 'pitch_std_hz' in prosody:
                refined_prosody['f0_std_hz'] = round_float(prosody['pitch_std_hz'])
            if 'energy_mean' in prosody:
                refined_prosody['rms'] = round_float(prosody['energy_mean'])
            f0 = refined_prosody.get('f0_mean_hz')
            f0std = refined_prosody.get('f0_std_hz')
            if isinstance(f0, (int, float)):
                refined_prosody['pitch_register'] = 'low' if f0 < 135 else ('mid' if f0 <= 185 else 'high')
                if 85 <= f0 <= 180:
                    refined_prosody['pitch_vs_adult_reference'] = 'within typical male range'
                elif 165 <= f0 <= 255:
                    refined_prosody['pitch_vs_adult_reference'] = 'within typical female range'
                elif f0 < 85:
                    refined_prosody['pitch_vs_adult_reference'] = 'below adult typical range'
                elif f0 > 255:
                    refined_prosody['pitch_vs_adult_reference'] = 'above adult typical range'
                else:
                    refined_prosody['pitch_vs_adult_reference'] = 'between male and female typical ranges'
            if isinstance(f0std, (int, float)):
                refined_prosody['pitch_variability'] = 'monotone' if f0std < 20 else ('moderate' if f0std <= 50 else 'expressive')

        # Gender gating
        gender = sp.get('gender')
        gender_conf = sp.get('gender_confidence')
        refined_gender = gender if (gender and isinstance(gender_conf, (int, float)) and gender_conf >= 0.6) else None

        # Emotion gating
        emotion = sp.get('emotion') or {}
        refined_emotion = emotion if (emotion and isinstance(emotion.get('confidence'), (int, float)) and emotion['confidence'] >= 0.6) else None

        combined_speech = {**tx}
        if refined_gender:
            combined_speech['gender'] = refined_gender
        if refined_prosody:
            combined_speech['prosody'] = refined_prosody
        if refined_emotion:
            combined_speech['emotion'] = refined_emotion

        item = {
            'metadata': {
                'video_id': vid,
                'duration_seconds': id_to_meta[vid]['duration_seconds'],
                'processed_at': datetime.now().isoformat()
            },
            'text': txa,
        }

        if tx.get('has_speech'):
            item['speech'] = combined_speech

        if mu.get('has_music'):
            music_out = {
                'has_music': True,
                'tempo_bpm': round_float(mu.get('tempo_bpm', 0.0))
            }
            if 'rms' in mu:
                music_out['rms'] = round_float(mu['rms'])
            if 'tempo_strength' in mu:
                music_out['tempo_strength'] = round_float(mu['tempo_strength'])
            mood = (mu.get('mood') or {})
            if isinstance(mood.get('confidence'), (int, float)) and mood['confidence'] >= 0.6:
                music_out['mood'] = mood
            item['music'] = music_out

        features[vid] = item

    save_json(features, cfg.OUTPUT_JSON)

    # Optionally delete intermediates
    if not args.keep_intermediate:
        try:
            shutil.rmtree(cfg.OUTPUT_VOCALS_DIR, ignore_errors=True)
            shutil.rmtree(cfg.OUTPUT_AUDIO_DIR, ignore_errors=True)
        except Exception:
            pass

    elapsed = time.time() - start_time
    logging.info("\n%s", "=" * 60)
    logging.info("Pipeline complete in %.1fs", elapsed)
    logging.info("Results saved to: %s", cfg.OUTPUT_JSON)
    logging.info("%s", "=" * 60)


