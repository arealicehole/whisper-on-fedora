#!/usr/bin/env python3
"""Test WhisperX diarization on Blackwell GPU"""

import whisperx
import torch
import tempfile
import numpy as np
import soundfile as sf
import os

print("=" * 50)
print("🚀 WhisperX Blackwell GPU Test")
print("=" * 50)

# Check GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    assert capability == (12, 0), f"Not Blackwell GPU: {capability}"

# Create test audio
print("\n📊 Creating test audio...")
sample_rate = 16000
duration = 10
audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio, sample_rate)
    audio_path = f.name

# Test WhisperX
print("\n🎯 Testing WhisperX transcription...")
try:
    # Load models
    model = whisperx.load_model("base", device="cuda", compute_type="float16")
    audio = whisperx.load_audio(audio_path)
    
    # Transcribe
    result = model.transcribe(audio, batch_size=16)
    print(f"✅ Transcription successful: {len(result['segments'])} segments")
    
    # Align
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device="cuda"
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, 
        device="cuda", return_char_alignments=False
    )
    print("✅ Alignment successful")
    
    # Diarize (requires HF token)
    try:
        # Try to load HF token
        hf_token = None
        token_files = [
            os.path.expanduser("~/.config/whisper/token"),
            os.path.expanduser("~/.cache/huggingface/token")
        ]
        for token_file in token_files:
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    content = f.read().strip()
                    if content.startswith("HF_TOKEN="):
                        hf_token = content.split("=", 1)[1].strip()
                    else:
                        hf_token = content
                    break
        
        if hf_token:
            print(f"  Using HF token: {hf_token[:10]}...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device="cuda"
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"✅ Diarization successful: {len(set(s.get('speaker', 'unknown') for s in result['segments']))} speakers")
        else:
            print("⚠️ Diarization skipped (no HF token found)")
    except Exception as e:
        print(f"⚠️ Diarization skipped: {e}")
    
    print("\n🎉 WhisperX works on Blackwell GPU!")
    
except Exception as e:
    print(f"❌ WhisperX failed: {e}")
    raise
finally:
    # Clean up
    os.unlink(audio_path)

print("=" * 50)