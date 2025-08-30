#!/usr/bin/env python3
"""Test diarization on Blackwell GPU with PyAnnote"""

import torch
import numpy as np
import soundfile as sf
import tempfile
import os
from pyannote.audio import Pipeline

print("=" * 50)
print("🎯 Testing PyAnnote Diarization on Blackwell GPU")
print("=" * 50)

# Check GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    
# Load diarization pipeline
print("\n📊 Loading PyAnnote pipeline...")
token = os.environ.get("HF_TOKEN", "your_hf_token_here")  # Set HF_TOKEN env var
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)
pipeline.to(torch.device("cuda"))
print("✅ Pipeline loaded on GPU")

# Create test audio with two distinct sections
print("\n🎵 Creating test audio...")
sample_rate = 16000
duration_per_speaker = 5

# Speaker 1: Low frequency tone
speaker1_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_per_speaker, sample_rate * duration_per_speaker)) * 0.3

# Speaker 2: High frequency tone  
speaker2_audio = np.sin(2 * np.pi * 880 * np.linspace(0, duration_per_speaker, sample_rate * duration_per_speaker)) * 0.3

# Combine
audio = np.concatenate([speaker1_audio, speaker2_audio])

# Save to file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio, sample_rate)
    audio_path = f.name
    print(f"Audio saved: {audio_path}")

# Run diarization
print("\n🔍 Running diarization...")
try:
    diarization = pipeline(audio_path)
    
    # Process results
    speakers = set()
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    print(f"\n✅ Diarization successful!")
    print(f"   Found {len(speakers)} speakers: {speakers}")
    print(f"   Total segments: {len(segments)}")
    
    # Show first few segments
    print("\n📋 Sample segments:")
    for seg in segments[:5]:
        print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['speaker']}")
    
    # Check GPU memory usage
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n🎉 DIARIZATION WORKS ON BLACKWELL GPU!")
    
except Exception as e:
    print(f"\n❌ Diarization failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    import os
    os.unlink(audio_path)

print("=" * 50)