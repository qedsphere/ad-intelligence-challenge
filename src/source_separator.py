"""
Source separation using Demucs
Separates vocals from background music/noise
"""

import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import numpy as np

# Global model cache
_demucs_model = None
_device = None


def load_demucs_model(model_name="mdx_q"):
    """
    Load Demucs model with MPS support
    
    Args:
        model_name: Model to use ('mdx_q' is fastest with good quality)
    
    Returns:
        tuple: (model, device)
    """
    global _demucs_model, _device
    
    if _demucs_model is None:
        print(f"Loading Demucs '{model_name}' model (first time only)...")
        
        # Detect device
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
            print("Using MPS (Apple GPU)")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            _device = torch.device("cpu")
            print("Using CPU")
        
        # Load model
        _demucs_model = get_model(model_name)
        _demucs_model.to(_device)
        _demucs_model.eval()
        
        print("Model loaded!")
    
    return _demucs_model, _device


def separate_vocals(audio_path, output_dir="output/vocals"):
    """
    Separate vocals from audio using Demucs
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated vocals
        
    Returns:
        str: Path to separated vocals file
    """
    # Generate output path
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_dir, f"{filename}.wav")
    
    # Skip if already separated
    if os.path.exists(vocals_path):
        return vocals_path
    
    # Load model
    model, device = load_demucs_model()
    
    # Load audio using soundfile (avoid TorchCodec issues)
    audio_np, sr = sf.read(audio_path)
    
    # Convert to tensor and transpose if needed
    if len(audio_np.shape) == 1:
        # Mono - convert to stereo
        audio = torch.from_numpy(audio_np).float()
        audio = torch.stack([audio, audio])  # [2, samples]
    else:
        # Stereo or multi-channel - transpose to [channels, samples]
        audio = torch.from_numpy(audio_np.T).float()
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
    
    # Resample to model's expected sample rate (44100 Hz)
    if sr != model.samplerate:
        resampler = torchaudio.transforms.Resample(sr, model.samplerate)
        audio = resampler(audio)
    
    # Move to device and add batch dimension
    audio = audio.to(device)
    audio = audio.unsqueeze(0)  # [batch, channels, samples]
    
    # Apply separation
    with torch.no_grad():
        sources = apply_model(model, audio, device=device)
    
    # Extract vocals (index depends on model)
    # For mdx: sources = [batch, stems, channels, samples]
    # stems order: drums, bass, other, vocals
    vocals = sources[0, -1]  # Last stem is vocals
    
    # Convert to mono and move to CPU
    vocals_mono = vocals.mean(dim=0, keepdim=True).cpu()
    
    # Resample to 16kHz for speech processing
    resampler = torchaudio.transforms.Resample(model.samplerate, 16000)
    vocals_mono = resampler(vocals_mono)
    
    # Save using soundfile (avoid TorchCodec issues)
    vocals_np = vocals_mono.squeeze().numpy()
    sf.write(vocals_path, vocals_np, 16000)
    
    return vocals_path


def separate_vocals_batch(audio_files, video_ids):
    """
    Separate vocals from a batch of audio files
    
    Args:
        audio_files: List of audio file paths
        video_ids: List of corresponding video IDs
        
    Returns:
        dict: Dictionary mapping video_id to vocals path
    """
    print(f"\nSeparating vocals from {len(audio_files)} audio files...")
    print("=" * 60)
    
    results = {}
    newly_separated = 0
    
    for audio_path, video_id in zip(audio_files, video_ids):
        print(f"[SEPARATE] {video_id}...")
        
        try:
            vocals_path = separate_vocals(audio_path)
            
            # Check if it was newly separated
            if not os.path.exists(vocals_path.replace('.wav', '.done')):
                newly_separated += 1
            
            results[video_id] = vocals_path
            print(f"[DONE] {video_id}")
            
        except Exception as e:
            print(f"[ERROR] {video_id} - {e}")
            results[video_id] = audio_path  # Fallback to original
    
    print("=" * 60)
    print(f"Separation complete: {len(results)} vocals extracted")
    if newly_separated > 0:
        print(f"  Newly separated: {newly_separated}")
    
    return results

