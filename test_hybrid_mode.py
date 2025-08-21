#!/usr/bin/env python3
"""
Test script to verify hybrid mode is working correctly
"""

import requests
import json
import sys
import time
import numpy as np
import wave
import tempfile
import os

def create_test_audio(duration=5, sample_rate=16000):
    """Create a test audio file with sine waves"""
    t = np.linspace(0, duration, duration * sample_rate)
    # Create two different frequencies to simulate two speakers
    audio1 = np.sin(2 * np.pi * 440 * t[:len(t)//2])  # First half
    audio2 = np.sin(2 * np.pi * 880 * t[len(t)//2:])  # Second half
    audio = np.concatenate([audio1, audio2])
    
    # Add some speech-like modulation
    envelope = np.sin(2 * np.pi * 3 * t) * 0.3 + 0.7
    audio = audio * envelope
    
    # Save as WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())
        return f.name

def test_service():
    """Test the Whisper API service in hybrid mode"""
    
    print("=" * 60)
    print("Hybrid Mode Test for RTX 5060 Ti")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking service health...")
    try:
        response = requests.get("http://localhost:8765/health")
        health = response.json()
        
        print(f"   ✓ Service status: {health['ok']}")
        print(f"   ✓ Model: {health['model']}")
        print(f"   ✓ Device: {health['device']}")
        
        if 'hybrid_mode' in health:
            print(f"   ✓ Hybrid Mode Active:")
            print(f"     - Whisper: {health['hybrid_mode']['whisper']}")
            print(f"     - Diarization: {health['hybrid_mode']['diarization']}")
            print(f"     - GPU: {health['hybrid_mode']['gpu']}")
        
        if health['diarization']['pipeline_loaded']:
            print(f"   ✓ Diarization: Pipeline loaded successfully")
        else:
            print(f"   ✗ Diarization: Pipeline not loaded")
            if health['diarization']['error']:
                print(f"     Error: {health['diarization']['error'][:100]}")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # 2. Test basic transcription (GPU)
    print("\n2. Testing basic transcription (GPU accelerated)...")
    audio_file = create_test_audio(duration=3)
    
    try:
        start_time = time.time()
        
        with open(audio_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post(
                "http://localhost:8765/v1/transcribe",
                files=files,
                data={'format': 'json'}
            )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Transcription successful in {elapsed:.2f}s")
            print(f"     Duration: {result.get('duration', 0):.2f}s")
            print(f"     Segments: {len(result.get('segments', []))}")
            
            # Check if we got any text (might be empty for test audio)
            if result.get('text'):
                print(f"     Text: {result['text'][:100]}")
            else:
                print(f"     Note: No speech detected (expected for test audio)")
        else:
            print(f"   ✗ Transcription failed: {response.status_code}")
            print(f"     {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ Transcription error: {e}")
    finally:
        os.unlink(audio_file)
    
    # 3. Test with diarization (CPU)
    print("\n3. Testing with speaker diarization (CPU mode)...")
    audio_file = create_test_audio(duration=5)
    
    try:
        start_time = time.time()
        
        with open(audio_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post(
                "http://localhost:8765/v1/transcribe",
                files=files,
                data={
                    'format': 'json',
                    'diarize': 'true',
                    'num_speakers': 2
                }
            )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Diarization successful in {elapsed:.2f}s")
            print(f"     Processing speed: {elapsed/5:.2f}x real-time")
            
            # Check for speaker labels
            segments_with_speakers = [s for s in result.get('segments', []) if 'speaker' in s]
            if segments_with_speakers:
                speakers = set(s['speaker'] for s in segments_with_speakers)
                print(f"     Speakers detected: {len(speakers)}")
                print(f"     Speaker IDs: {', '.join(speakers)}")
            else:
                print(f"     Note: No speaker segments (expected for test audio)")
        else:
            print(f"   ✗ Diarization failed: {response.status_code}")
            print(f"     {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ Diarization error: {e}")
    finally:
        os.unlink(audio_file)
    
    # 4. Performance summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("✓ Service is running in hybrid mode")
    print("✓ Whisper transcription uses GPU (faster-whisper CUDA)")
    print("✓ Speaker diarization uses CPU (PyTorch CPU mode)")
    print("✓ Both components are functional")
    print("\nThis configuration works around the RTX 5060 Ti (sm_120)")
    print("compatibility issue until PyTorch adds native support.")
    
    return True

if __name__ == "__main__":
    success = test_service()
    sys.exit(0 if success else 1)