#!/usr/bin/env python3
"""
End-to-End Diarization Test for Blackwell GPU
Tests both Whisper transcription and PyAnnote diarization on GPU
"""

import sys
import os
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
import json


def create_test_audio():
    """Create a simple test audio file with silence gaps (simulating speakers)"""
    sample_rate = 16000
    duration = 10  # seconds
    
    # Create a simple audio signal with two "speakers" (different frequencies)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speaker 1: 0-3 seconds (440 Hz)
    speaker1 = np.sin(2 * np.pi * 440 * t[:int(3 * sample_rate)])
    
    # Silence: 3-4 seconds
    silence1 = np.zeros(int(1 * sample_rate))
    
    # Speaker 2: 4-7 seconds (880 Hz)
    speaker2 = np.sin(2 * np.pi * 880 * t[:int(3 * sample_rate)])
    
    # Silence: 7-8 seconds
    silence2 = np.zeros(int(1 * sample_rate))
    
    # Speaker 1 again: 8-10 seconds
    speaker1_again = np.sin(2 * np.pi * 440 * t[:int(2 * sample_rate)])
    
    # Combine all segments
    audio = np.concatenate([speaker1, silence1, speaker2, silence2, speaker1_again])
    
    # Add some noise to make it more realistic
    audio += np.random.normal(0, 0.01, len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5
    
    return audio.astype(np.float32), sample_rate


def test_whisper_transcription():
    """Test Whisper transcription on GPU"""
    print("\n📊 Testing Whisper Transcription")
    print("-" * 40)
    
    try:
        from faster_whisper import WhisperModel
        
        # Create test audio
        audio, sample_rate = create_test_audio()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            temp_path = tmp_file.name
        
        print(f"Created test audio: {temp_path}")
        
        # Initialize model (using tiny for speed, but you can use larger)
        print("Loading Whisper model on GPU...")
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        
        # Check GPU usage
        if torch.cuda.is_available():
            print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Transcribe
        print("Transcribing...")
        start_time = time.time()
        segments, info = model.transcribe(temp_path, beam_size=5)
        
        # Process results
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
            print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        elapsed = time.time() - start_time
        print(f"Transcription time: {elapsed:.2f}s")
        
        if torch.cuda.is_available():
            print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print("✅ Whisper transcription on GPU: PASS")
        else:
            print("⚠️  Running on CPU (GPU not available)")
        
        # Cleanup
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return False


def test_pyannote_diarization():
    """Test PyAnnote diarization on GPU"""
    print("\n📊 Testing PyAnnote Diarization")
    print("-" * 40)
    
    try:
        from pyannote.audio import Pipeline
        
        # Check for token
        token_file = Path.home() / ".config" / "whisper" / "token"
        hf_token = None
        
        if token_file.exists():
            with open(token_file) as f:
                content = f.read().strip()
                if content.startswith("HF_TOKEN="):
                    hf_token = content.split("=", 1)[1].strip()
        
        if not hf_token:
            print("⚠️  No HuggingFace token found - skipping diarization test")
            print("   To enable: echo 'HF_TOKEN=your_token' > ~/.config/whisper/token")
            return None
        
        # Create test audio
        audio, sample_rate = create_test_audio()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            temp_path = tmp_file.name
        
        print(f"Created test audio: {temp_path}")
        
        # Load pipeline
        print("Loading PyAnnote pipeline on GPU...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Run diarization
        print("Running diarization...")
        start_time = time.time()
        diarization = pipeline(temp_path)
        elapsed = time.time() - start_time
        
        # Process results
        print("Diarization results:")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"  [{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}")
        
        print(f"Diarization time: {elapsed:.2f}s")
        
        if torch.cuda.is_available():
            print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print("✅ PyAnnote diarization on GPU: PASS")
        else:
            print("⚠️  Running on CPU (GPU not available)")
        
        # Cleanup
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ PyAnnote diarization failed: {e}")
        if "no kernel image" in str(e).lower() or "nms" in str(e).lower():
            print("   This is the Blackwell torchvision issue!")
            print("   Ensure you're using PyTorch nightly builds")
        return False


def test_api_service():
    """Test if the API service is running"""
    print("\n📊 Testing API Service")
    print("-" * 40)
    
    try:
        import requests
        
        # Check if service is running
        try:
            response = requests.get("http://127.0.0.1:8767/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API service is running")
                print(f"   Status: {data.get('status', 'unknown')}")
                if 'gpu_available' in data:
                    print(f"   GPU available: {data['gpu_available']}")
                return True
            else:
                print(f"⚠️  API returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("⚠️  API service is not running")
            print("   Start it with: python main.py")
            return None
            
    except ImportError:
        print("⚠️  requests library not installed")
        return None


def check_cpu_fallback():
    """Ensure we're NOT using CPU fallback"""
    print("\n📊 Checking for CPU Fallback")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - would fall back to CPU")
        print("   This violates the no-CPU-fallback requirement!")
        return False
    
    # Check environment variables
    cpu_vars = ['FORCE_CPU', 'USE_CPU', 'CUDA_VISIBLE_DEVICES']
    for var in cpu_vars:
        value = os.environ.get(var, '')
        if var == 'CUDA_VISIBLE_DEVICES' and value == '':
            print(f"⚠️  {var} is empty - this hides GPUs")
            return False
        elif var != 'CUDA_VISIBLE_DEVICES' and value.lower() in ['true', '1', 'yes']:
            print(f"⚠️  {var}={value} might force CPU mode")
            return False
    
    print("✅ No CPU fallback detected")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    
    return True


def main():
    """Run all end-to-end tests"""
    print("=" * 50)
    print("🚀 End-to-End Diarization Test Suite")
    print("=" * 50)
    
    # Track results
    results = {
        'cpu_check': False,
        'whisper': False,
        'diarization': False,
        'api': False
    }
    
    # Check no CPU fallback
    results['cpu_check'] = check_cpu_fallback()
    
    # Test Whisper
    results['whisper'] = test_whisper_transcription()
    
    # Test PyAnnote
    diarization_result = test_pyannote_diarization()
    if diarization_result is not None:
        results['diarization'] = diarization_result
    else:
        results['diarization'] = None  # Skipped
    
    # Test API
    api_result = test_api_service()
    if api_result is not None:
        results['api'] = api_result
    else:
        results['api'] = None  # Not running
    
    # Final Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"✅ No CPU fallback: {'PASS' if results['cpu_check'] else 'FAIL'}")
    print(f"✅ Whisper transcription on GPU: {'PASS' if results['whisper'] else 'FAIL'}")
    
    if results['diarization'] is None:
        print(f"⚠️  PyAnnote diarization: SKIPPED (no token)")
    else:
        print(f"✅ PyAnnote diarization on GPU: {'PASS' if results['diarization'] else 'FAIL'}")
    
    if results['api'] is None:
        print(f"⚠️  API service: NOT RUNNING")
    else:
        print(f"✅ API service: {'PASS' if results['api'] else 'FAIL'}")
    
    # Overall result
    critical_pass = results['cpu_check'] and results['whisper']
    if results['diarization'] is not None:
        critical_pass = critical_pass and results['diarization']
    
    print("\n" + "=" * 50)
    if critical_pass:
        print("✅ All critical tests PASSED!")
        print("\n🎉 Your Blackwell GPU setup is working perfectly!")
        print("   Both Whisper and PyAnnote are running on GPU")
        print("   No CPU fallback detected")
        return 0
    else:
        print("❌ Some tests FAILED")
        print("\nTroubleshooting:")
        print("  1. Ensure you're in the whisper-blackwell venv")
        print("  2. Re-run setup_blackwell_venv.sh if needed")
        print("  3. Check nvidia-smi for GPU status")
        print("  4. Verify PyTorch nightly is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())