"""
Audio extraction from video files
Extracts audio tracks and converts to WAV format (16kHz, mono)
"""

import os
import subprocess
from pathlib import Path
import soundfile as sf


def extract_audio_ffmpeg(video_path, output_path, sample_rate=16000):
    """
    Extract audio from video using FFmpeg
    
    Args:
        video_path: Path to input video file
        output_path: Path to output WAV file
        sample_rate: Target sample rate (default: 16000 Hz)
    
    Returns:
        str: Path to extracted audio file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FFmpeg command to extract audio
    # -i: input file
    # -vn: no video
    # -acodec pcm_s16le: 16-bit PCM
    # -ar: sample rate
    # -ac 1: mono (1 channel)
    # -y: overwrite output file
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite
        output_path
    ]
    
    # Run FFmpeg (suppress output)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed for {video_path}: {e}")


def get_audio_duration(audio_path):
    """
    Get duration of audio file in seconds
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        float: Duration in seconds
    """
    # Use soundfile to get duration
    info = sf.info(audio_path)
    return info.duration


def extract_audio_for_video(video_path, output_dir="output/audio"):
    """
    Extract audio from a single video file
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio
        
    Returns:
        dict: Information about extracted audio
    """
    # Get video ID from filename
    video_id = Path(video_path).stem
    
    # Output path
    output_path = os.path.join(output_dir, f"{video_id}.wav")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"[SKIP] Audio already exists: {video_id}")
        duration = get_audio_duration(output_path)
        return {
            'video_id': video_id,
            'audio_path': output_path,
            'duration_seconds': duration,
            'extracted': False
        }
    
    # Extract audio
    print(f"[EXTRACT] {video_id}...")
    extract_audio_ffmpeg(video_path, output_path)
    
    # Get duration
    duration = get_audio_duration(output_path)
    
    print(f"[DONE] {video_id} ({duration:.1f}s)")
    
    return {
        'video_id': video_id,
        'audio_path': output_path,
        'duration_seconds': duration,
        'extracted': True
    }


def extract_audio_batch(video_files, output_dir="output/audio"):
    """
    Extract audio from multiple video files
    
    Args:
        video_files: List of video file paths
        output_dir: Directory to save extracted audio
        
    Returns:
        list: List of extraction results
    """
    print(f"\nExtracting audio from {len(video_files)} videos...")
    print("=" * 60)
    
    results = []
    for video_path in video_files:
        result = extract_audio_for_video(video_path, output_dir)
        results.append(result)
    
    # Summary
    newly_extracted = sum(1 for r in results if r['extracted'])
    already_existed = len(results) - newly_extracted
    
    print("=" * 60)
    print(f"Extraction complete:")
    print(f"  Newly extracted: {newly_extracted}")
    print(f"  Already existed: {already_existed}")
    print(f"  Total: {len(results)}")
    
    return results

