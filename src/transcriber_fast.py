"""
Speech-to-text transcription using faster-whisper
Optimized for speed with CTranslate2 backend
"""

import os
from faster_whisper import WhisperModel


# Global model cache
_whisper_model = None
_configured_model_name = "small"
_configured_device = "cpu"
_configured_compute_type = "int8"


def configure(model_name: str = None, device: str = None, compute_type: str = None):
    """Configure global model settings and reset cache so next load uses them."""
    global _configured_model_name, _configured_device, _configured_compute_type, _whisper_model
    if model_name:
        _configured_model_name = model_name
    if device:
        _configured_device = device
    if compute_type:
        _configured_compute_type = compute_type
    # Reset cached model so next call reloads with new settings
    _whisper_model = None


def load_whisper_model(model_name: str = None, device: str = None, compute_type: str = None):
    """
    Load faster-whisper model (cached globally)
    
    Args:
        model_name: Model size (tiny, base, small, medium, large-v2, large-v3)
        device: Device to use ("cpu", "cuda", or "auto")
        compute_type: Quantization type ("int8", "int8_float16", "float16", "float32")
                     int8 is fastest on CPU, float16 is best for GPU
        
    Returns:
        Loaded WhisperModel
    """
    global _whisper_model
    
    # Use configured values if none provided
    model_name = model_name or _configured_model_name
    device = device or _configured_device
    compute_type = compute_type or _configured_compute_type

    if _whisper_model is None:
        print(f"Loading faster-whisper '{model_name}' model (first time only)...")
        print(f"  Device: {device}, Compute type: {compute_type}")
        
        # Load model with CTranslate2 backend
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=1  # Single worker for sequential processing
        )
        print("Model loaded!")
    
    return _whisper_model


def detect_speech_quick(audio_path):
    """
    Quick speech detection using VAD
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        bool: True if speech likely detected, False if music/silence only
    """
    model = load_whisper_model()
    
    # Transcribe first 30 seconds only for detection
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,  # Faster beam search
        vad_filter=True,  # Use VAD to filter
        vad_parameters=dict(min_silence_duration_ms=500),
        language=None,  # Auto-detect
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=False
    )
    
    # Check if any segments were detected
    segment_count = 0
    for _ in segments:
        segment_count += 1
        if segment_count > 0:
            break
    
    # If no segments or very low speech probability, no speech
    has_speech = segment_count > 0 and info.language_probability > 0.3
    
    return has_speech


def transcribe_audio(audio_path, model_name: str = None):
    """
    Transcribe audio file to text with hallucination detection
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        
    Returns:
        dict: Transcription results
    """
    model = load_whisper_model(model_name)
    
    # Transcribe with optimized settings
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,  # Balance between speed and accuracy
        vad_filter=True,  # Filter out non-speech
        vad_parameters=dict(
            min_silence_duration_ms=500,
            threshold=0.5
        ),
        language=None,  # Auto-detect
        condition_on_previous_text=False,  # Reduce hallucinations
        initial_prompt=None,
        word_timestamps=False,  # Faster without word timestamps
        temperature=0.0,  # Deterministic output
        compression_ratio_threshold=2.4,  # Filter low-quality segments
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6  # Higher threshold for no speech
    )
    
    # Collect segments
    all_segments = []
    full_text = []
    
    for segment in segments:
        all_segments.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'no_speech_prob': segment.no_speech_prob
        })
        full_text.append(segment.text)
    
    # Calculate average no_speech probability
    avg_no_speech = 0.0
    if all_segments:
        avg_no_speech = sum(s['no_speech_prob'] for s in all_segments) / len(all_segments)
    
    return {
        'text': ' '.join(full_text).strip(),
        'language': info.language,
        'language_probability': info.language_probability,
        'segments': all_segments,
        'avg_no_speech_prob': avg_no_speech
    }


def transcribe_audio_for_video(audio_path, video_id):
    """
    Transcribe audio and format results for a video with hallucination filtering
    
    Args:
        audio_path: Path to audio file
        video_id: Video identifier
        
    Returns:
        dict: Formatted transcription results
    """
    print(f"[TRANSCRIBE] {video_id}...")
    
    result = transcribe_audio(audio_path)
    
    # Check if this is likely hallucination (music transcribed as words)
    avg_no_speech = result.get('avg_no_speech_prob', 0.0)
    lang_prob = result.get('language_probability', 0.0)
    is_hallucination = avg_no_speech > 0.5 or lang_prob < 0.3
    
    if is_hallucination:
        print(f"[MUSIC] {video_id} - No speech detected (music/background only)")
        return {
            'has_speech': False,
            'transcription': '',
            'language': result['language'],
            'word_count': 0
        }
    
    # Count words
    word_count = len(result['text'].split()) if result['text'] else 0
    has_speech = word_count > 0
    
    output = {
        'has_speech': has_speech,
        'transcription': result['text'],
        'language': result['language'],
        'word_count': word_count,
    }
    
    print(f"[DONE] {video_id} - {word_count} words ({result['language']})")
    
    return output


def transcribe_batch(audio_files, video_ids):
    """
    Transcribe multiple audio files
    
    Args:
        audio_files: List of audio file paths
        video_ids: List of video IDs corresponding to audio files
        
    Returns:
        dict: Transcription results keyed by video ID
    """
    print(f"\nTranscribing {len(audio_files)} audio files...")
    print("=" * 60)
    
    results = {}
    
    for audio_path, video_id in zip(audio_files, video_ids):
        if not os.path.exists(audio_path):
            print(f"[SKIP] {video_id} - audio file not found")
            continue
        
        try:
            result = transcribe_audio_for_video(audio_path, video_id)
            results[video_id] = result
        except Exception as e:
            print(f"[ERROR] {video_id} - {e}")
            results[video_id] = {
                'has_speech': False,
                'transcription': '',
                'language': 'unknown',
                'word_count': 0,
                'error': str(e)
            }
    
    print("=" * 60)
    print(f"Transcription complete: {len(results)} videos")
    
    return results


