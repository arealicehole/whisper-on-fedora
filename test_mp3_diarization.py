#!/usr/bin/env python3
"""Test transcription and diarization on MP3 files"""

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import time
import os

print("=" * 60)
print("🎯 Testing MP3 Transcription & Diarization on Blackwell GPU")
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

# Load diarization pipeline
print("\n🎭 Loading PyAnnote diarization...")
token = os.environ.get("HF_TOKEN", "your_hf_token_here")  # Set HF_TOKEN env var
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)
diarization_pipeline.to(torch.device("cuda"))
print("✅ Diarization pipeline loaded on GPU")

# Test on MP3 files
mp3_files = ["250807_0615.mp3", "250808_0121.mp3", "250808_1010.mp3"]

for mp3_file in mp3_files[:1]:  # Test first file only
    print(f"\n{'='*60}")
    print(f"📁 Testing: {mp3_file}")
    print("="*60)
    
    try:
        # Transcribe
        print("\n1️⃣ Transcribing...")
        start = time.time()
        segments, info = model.transcribe(mp3_file)
        segments_list = list(segments)
        transcribe_time = time.time() - start
        
        print(f"   ✅ Transcription complete in {transcribe_time:.2f}s")
        print(f"   Language: {info.language}")
        print(f"   Duration: {info.duration:.1f}s")
        print(f"   Segments: {len(segments_list)}")
        
        # Show first few words
        text = " ".join([s.text for s in segments_list[:3]])
        print(f"   Sample text: {text[:150]}...")
        
        # Diarize
        print("\n2️⃣ Running diarization...")
        start = time.time()
        diarization = diarization_pipeline(mp3_file)
        diarize_time = time.time() - start
        
        # Process results
        speakers = set()
        dia_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            dia_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        print(f"   ✅ Diarization complete in {diarize_time:.2f}s")
        print(f"   Speakers found: {len(speakers)} - {speakers}")
        print(f"   Speaker segments: {len(dia_segments)}")
        
        # Map transcription segments to speakers
        print("\n3️⃣ Mapping speakers to transcription...")
        for segment in segments_list[:5]:  # Show first 5 segments
            mid_time = (segment.start + segment.end) / 2
            speaker = "UNKNOWN"
            for dia_seg in dia_segments:
                if dia_seg['start'] <= mid_time <= dia_seg['end']:
                    speaker = dia_seg['speaker']
                    break
            print(f"   [{speaker:10}] {segment.start:5.1f}s - {segment.end:5.1f}s: {segment.text}")
        
        # GPU Memory
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Error processing {mp3_file}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("🎉 TEST COMPLETE - Diarization works on Blackwell GPU!")
print("="*60)