"""
Speech feature analysis using librosa and pyAudioAnalysis
Extracts gender, prosody, and emotion from speech
"""

import os
import numpy as np
import librosa
import soundfile as sf


def extract_prosody_features(audio_path):
    """
    Extract prosody features from audio
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        dict: Prosody features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Skip if audio is too short or silent
    if len(y) < sr * 0.1:  # Less than 0.1 seconds
        return {
            'pitch_mean_hz': 0,
            'pitch_std_hz': 0,
            'energy_mean': 0,
            'energy_std': 0,
            'speaking_rate_estimate': 0,
            'zero_crossing_rate': 0
        }
    
    # Extract pitch using librosa with speech-specific parameters
    pitches, magnitudes = librosa.piptrack(
        y=y, 
        sr=sr,
        fmin=75,      # Minimum F0 for male voices
        fmax=300,     # Maximum F0 for female voices
        threshold=0.1 # Higher threshold to avoid noise
    )
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        # Only consider realistic speech pitch range
        if 75 < pitch < 300:
            pitch_values.append(pitch)
    
    # Calculate pitch statistics (use median to be robust against outliers)
    if len(pitch_values) > 10:  # Need minimum samples
        pitch_mean = float(np.median(pitch_values))
        pitch_std = float(np.std(pitch_values))
    else:
        pitch_mean = 0
        pitch_std = 0
    
    # Extract energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    
    # Zero crossing rate (speaking rate indicator)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    
    # Estimate speaking rate from onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    return {
        'pitch_mean_hz': pitch_mean,
        'pitch_std_hz': pitch_std,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'speaking_rate_estimate': float(tempo),
        'zero_crossing_rate': zcr_mean
    }


def classify_gender_from_pitch(pitch_mean_hz, pitch_std_hz, energy):
    """
    Gender classification based on pitch and spectral features
    Uses more robust thresholds based on speech research
    
    Typical ranges:
    - Male: 85-180 Hz (mean ~120 Hz)
    - Female: 165-255 Hz (mean ~210 Hz)
    
    Args:
        pitch_mean_hz: Mean pitch in Hz
        pitch_std_hz: Standard deviation of pitch
        energy: Average energy
        
    Returns:
        str: Gender classification (male/female/unknown)
    """
    if pitch_mean_hz == 0 or pitch_mean_hz < 50 or pitch_mean_hz > 500:
        # Out of typical human voice range
        return "unknown"
    
    # More refined thresholds
    if pitch_mean_hz < 150:
        # Likely male
        return "male"
    elif pitch_mean_hz > 180:
        # Likely female
        return "female"
    else:
        # Ambiguous range (150-180 Hz) - use additional features
        # Higher pitch variability often indicates female voices
        if pitch_std_hz > 40:
            return "female"
        elif pitch_std_hz < 30:
            return "male"
        else:
            return "unknown"


def estimate_emotion(energy_mean, pitch_std, speaking_rate):
    """
    Estimate emotion from prosody features
    
    Args:
        energy_mean: Mean energy
        pitch_std: Pitch variation
        speaking_rate: Speaking rate estimate
        
    Returns:
        dict: Emotion estimate
    """
    # Normalize features (rough estimates)
    arousal = min(1.0, (energy_mean * 10 + speaking_rate / 200) / 2)
    valence = min(1.0, max(0.0, (pitch_std / 100)))
    
    # Map to emotion labels
    if arousal > 0.6 and valence > 0.6:
        emotion = "excited"
    elif arousal > 0.6 and valence < 0.4:
        emotion = "angry"
    elif arousal < 0.4 and valence > 0.6:
        emotion = "calm"
    elif arousal < 0.4 and valence < 0.4:
        emotion = "sad"
    else:
        emotion = "neutral"
    
    confidence = 0.5 + min(0.4, abs(arousal - 0.5) + abs(valence - 0.5))
    
    return {
        'label': emotion,
        'confidence': float(confidence),
        'arousal': float(arousal),
        'valence': float(valence)
    }


def analyze_speech_for_video(audio_path, video_id, has_speech=True, word_count=0):
    """
    Analyze speech features for a single video
    
    Args:
        audio_path: Path to audio file
        video_id: Video identifier
        has_speech: Whether speech was detected
        word_count: Number of words transcribed
        
    Returns:
        dict: Speech analysis results
    """
    print(f"[ANALYZE] {video_id}...")
    
    try:
        # Only analyze if there's actual speech
        if not has_speech or word_count < 3:
            print(f"[SKIP] {video_id} - No speech detected")
            return {
                'gender': 'unknown',
                'prosody': {},
                'emotion': {'label': 'unknown', 'confidence': 0}
            }
        
        # Extract prosody features
        prosody = extract_prosody_features(audio_path)
        
        # Classify gender from pitch and spectral features
        gender = classify_gender_from_pitch(
            prosody['pitch_mean_hz'],
            prosody['pitch_std_hz'],
            prosody['energy_mean']
        )
        
        # Estimate emotion
        emotion = estimate_emotion(
            prosody['energy_mean'],
            prosody['pitch_std_hz'],
            prosody['speaking_rate_estimate']
        )
        
        print(f"[DONE] {video_id} - {gender}, {emotion['label']}")
        
        return {
            'gender': gender,
            'prosody': {
                'pitch_mean_hz': prosody['pitch_mean_hz'],
                'pitch_std_hz': prosody['pitch_std_hz'],
                'energy_level': int(prosody['energy_mean'] * 100),
                'speaking_rate_estimate': prosody['speaking_rate_estimate']
            },
            'emotion': emotion
        }
    
    except Exception as e:
        print(f"[ERROR] {video_id} - {e}")
        return {
            'gender': 'unknown',
            'prosody': {},
            'emotion': {'label': 'unknown', 'confidence': 0}
        }


def analyze_speech_batch(audio_files, video_ids, transcription_results):
    """
    Analyze speech features for multiple videos
    
    Args:
        audio_files: List of audio file paths
        video_ids: List of video IDs
        transcription_results: Dict of transcription results by video ID
        
    Returns:
        dict: Speech analysis results keyed by video ID
    """
    print(f"\nAnalyzing speech features for {len(audio_files)} videos...")
    print("=" * 60)
    
    results = {}
    
    for audio_path, video_id in zip(audio_files, video_ids):
        if not os.path.exists(audio_path):
            print(f"[SKIP] {video_id} - audio file not found")
            continue
        
        # Get transcription info
        transcription = transcription_results.get(video_id, {})
        has_speech = transcription.get('has_speech', False)
        word_count = transcription.get('word_count', 0)
        
        result = analyze_speech_for_video(audio_path, video_id, has_speech, word_count)
        results[video_id] = result
    
    print("=" * 60)
    print(f"Speech analysis complete: {len(results)} videos")
    
    return results

