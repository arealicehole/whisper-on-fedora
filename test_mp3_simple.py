#!/usr/bin/env python3
"""Simple test of MP3 transcription only"""

import torch
from faster_whisper import WhisperModel
import time

print("=" * 60)
print("🎯 Testing MP3 Transcription on Blackwell GPU")
print("=" * 60)

# Check GPU
print(f"\n📊 System Info:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")

# Load Whisper model
print("\n📝 Loading Whisper model...")
model = WhisperModel("tiny", device="cuda", compute_type="float16")
print("✅ Whisper model loaded")

# Test on MP3 file
mp3_file = "250807_0615.mp3"
print(f"\n📁 Testing: {mp3_file}")

try:
    # Transcribe
    print("\nTranscribing...")
    start = time.time()
    segments, info = model.transcribe(mp3_file)
    segments_list = list(segments)
    transcribe_time = time.time() - start
    
    print(f"✅ Transcription complete in {transcribe_time:.2f}s")
    print(f"   Language: {info.language}")
    print(f"   Duration: {info.duration:.1f}s")
    print(f"   Segments: {len(segments_list)}")
    
    # Show transcription
    print(f"\n📝 Full transcription:")
    full_text = " ".join([s.text for s in segments_list])
    print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
    
    # Show first few segments with timestamps
    print(f"\n⏱️ First 5 segments with timestamps:")
    for i, segment in enumerate(segments_list[:5], 1):
        print(f"   {i}. [{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
    
    # GPU Memory
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n✅ SUCCESS: MP3 transcription works on Blackwell GPU!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)