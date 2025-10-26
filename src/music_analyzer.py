"""
Music analysis for extracting background music features
Uses librosa to detect tempo, energy, and mood
"""

import os
import numpy as np
import librosa


def detect_music_presence(audio_path, vocals_path=None):
    """
    Detect if background music is present by comparing full audio vs vocals
    
    Args:
        audio_path: Path to full audio
        vocals_path: Path to separated vocals (optional)
        
    Returns:
        bool: True if music detected
    """
    try:
        # Load full audio
        y_full, sr = librosa.load(audio_path, sr=16000, duration=30)
        
        # If vocals separated, compare energy
        if vocals_path and os.path.exists(vocals_path):
            y_vocals, _ = librosa.load(vocals_path, sr=16000, duration=30)
            
            # Calculate RMS energy
            energy_full = np.sqrt(np.mean(y_full**2))
            energy_vocals = np.sqrt(np.mean(y_vocals**2))
            
            # If full audio has significantly more energy, likely has music
            has_music = energy_full > (energy_vocals * 1.3)
        else:
            # Just check if audio has energy (not silence)
            energy = np.sqrt(np.mean(y_full**2))
            has_music = energy > 0.01
        
        return has_music
    except:
        return False


def estimate_tempo(audio_path):
    """
    Estimate tempo (BPM) of audio
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        float: Tempo in BPM
    """
    try:
        # Load audio (use first 30 seconds for speed)
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Estimate tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        return float(tempo)
    except:
        return 0.0


def estimate_energy(audio_path):
    """
    Estimate overall energy level of audio
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        str: Energy level (low, medium, high)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        
        # Classify energy level
        if energy_mean < 0.05:
            return "low"
        elif energy_mean < 0.15:
            return "medium"
        else:
            return "high"
    except:
        return "unknown"


def estimate_mood(audio_path):
    """
    Estimate mood/valence of music using spectral features
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        dict: Mood classification
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Extract features
        # Spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Spectral rolloff (energy distribution)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Zero crossing rate (percussiveness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Simple mood classification based on features
        # High tempo + high brightness = upbeat/energetic
        # Low tempo + low brightness = calm/melancholic
        
        brightness_score = min(spectral_centroid / 3000, 1.0)  # Normalize
        tempo_score = min(tempo / 150, 1.0)  # Normalize
        energy_score = min(zcr / 0.15, 1.0)  # Normalize
        
        # Combined valence (happiness) score
        valence = (brightness_score + tempo_score) / 2
        
        # Combined arousal (energy) score
        arousal = (tempo_score + energy_score) / 2
        
        # Classify mood
        if arousal > 0.6 and valence > 0.6:
            mood = "upbeat"
            confidence = 0.7
        elif arousal > 0.6 and valence < 0.4:
            mood = "intense"
            confidence = 0.6
        elif arousal < 0.4 and valence > 0.6:
            mood = "calm"
            confidence = 0.7
        elif arousal < 0.4 and valence < 0.4:
            mood = "melancholic"
            confidence = 0.6
        else:
            mood = "neutral"
            confidence = 0.5
        
        return {
            "mood": mood,
            "confidence": round(confidence, 2),
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3)
        }
    except Exception as e:
        return {
            "mood": "unknown",
            "confidence": 0.0,
            "valence": 0.5,
            "arousal": 0.5
        }


def analyze_music(audio_path, vocals_path=None, has_speech=False):
    """
    Comprehensive music analysis
    
    Args:
        audio_path: Path to full audio file
        vocals_path: Path to separated vocals (optional)
        has_speech: Whether speech is present
        
    Returns:
        dict: Music analysis results
    """
    # Detect music presence
    has_music = detect_music_presence(audio_path, vocals_path)
    
    if not has_music:
        return {
            "has_music": False,
            "tempo_bpm": 0.0,
            "energy": "unknown",
            "mood": {
                "mood": "unknown",
                "confidence": 0.0
            }
        }
    
    # Analyze music features
    tempo = estimate_tempo(audio_path)
    energy = estimate_energy(audio_path)
    mood = estimate_mood(audio_path)
    
    return {
        "has_music": True,
        "tempo_bpm": round(tempo, 1),
        "energy": energy,
        "mood": mood
    }


def analyze_music_for_video(video_id, audio_path, vocals_path=None, has_speech=False):
    """
    Analyze music for a single video
    
    Args:
        video_id: Video identifier
        audio_path: Path to audio file
        vocals_path: Path to vocals file (if separated)
        has_speech: Whether speech detected
        
    Returns:
        dict: Music analysis results
    """
    print(f"[ANALYZE] {video_id}...")
    
    if not os.path.exists(audio_path):
        print(f"[SKIP] {video_id} - Audio file not found")
        return {
            "has_music": False,
            "tempo_bpm": 0.0,
            "energy": "unknown",
            "mood": {"mood": "unknown", "confidence": 0.0}
        }
    
    result = analyze_music(audio_path, vocals_path, has_speech)
    
    # Log results
    if result["has_music"]:
        print(f"[DONE] {video_id} - Music: Yes | Tempo: {result['tempo_bpm']} BPM | Mood: {result['mood']['mood']}")
    else:
        print(f"[DONE] {video_id} - Music: No")
    
    return result


def analyze_music_batch(video_ids, audio_files, vocals_paths, transcription_results):
    """
    Analyze music for multiple videos
    
    Args:
        video_ids: List of video IDs
        audio_files: List of original audio file paths
        vocals_paths: Dict of vocals paths by video ID
        transcription_results: Dict of transcription results
        
    Returns:
        dict: Music analysis results keyed by video ID
    """
    print(f"\nAnalyzing music from {len(video_ids)} videos...")
    print("=" * 60)
    
    results = {}
    
    for audio_path, video_id in zip(audio_files, video_ids):
        # Get vocals path if available
        vocals_path = vocals_paths.get(video_id)
        
        # Check if speech detected
        transcription = transcription_results.get(video_id, {})
        has_speech = transcription.get('has_speech', False)
        
        result = analyze_music_for_video(video_id, audio_path, vocals_path, has_speech)
        results[video_id] = result
    
    print("=" * 60)
    print(f"Music analysis complete: {len(results)} videos")
    
    # Summary stats
    music_count = sum(1 for r in results.values() if r.get('has_music', False))
    print(f"  Music detected: {music_count}/{len(results)}")
    
    return results


