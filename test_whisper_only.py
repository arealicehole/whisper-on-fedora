#!/usr/bin/env python3
"""Test Whisper transcription on GPU without diarization"""

import sys
import time
import tempfile
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel


def create_test_audio():
    """Create a simple test audio file"""
    sample_rate = 16000
    duration = 3  # seconds
    
    # Create a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # Add some noise
    audio += np.random.normal(0, 0.01, len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5
    
    return audio.astype(np.float32), sample_rate


def main():
    print("=" * 50)
    print("🚀 Whisper GPU Transcription Test")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 1
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    
    # Create test audio
    audio, sample_rate = create_test_audio()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        temp_path = tmp_file.name
    
    print(f"\n📊 Testing Whisper on GPU")
    print("-" * 30)
    
    try:
        # Initialize model
        print("Loading Whisper model...")
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        
        # Check GPU memory
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory before: {mem_before:.1f} MB")
        
        # Transcribe
        print("Transcribing...")
        start_time = time.time()
        segments, info = model.transcribe(temp_path, beam_size=5)
        
        # Process results
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        elapsed = time.time() - start_time
        mem_after = torch.cuda.memory_allocated() / 1024**2
        
        print(f"Transcription time: {elapsed:.2f}s")
        print(f"GPU memory after: {mem_after:.1f} MB")
        print(f"Transcription: '{transcription.strip()}'")
        
        print("\n✅ Whisper transcription on GPU: PASS")
        print("✅ GPU acceleration is working!")
        
        import os
        os.unlink(temp_path)
        
        return 0
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())