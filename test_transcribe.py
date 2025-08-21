#!/usr/bin/env python3
"""Test script to debug transcription issues"""

import os
import sys
from faster_whisper import WhisperModel

# Set up library paths
os.environ['LD_LIBRARY_PATH'] = '/home/ice/.venvs/whisper312/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/ice/.venvs/whisper312/lib/python3.12/site-packages/nvidia/cublas/lib'

def test_transcription(audio_file):
    print(f"Testing transcription of: {audio_file}")
    
    # Try different models and settings
    for model_size in ['tiny', 'small']:
        print(f"\n=== Testing {model_size} model ===")
        
        try:
            model = WhisperModel(model_size, device='cuda', compute_type='float16')
            print(f"Model loaded successfully")
            
            # Test WITHOUT VAD filter first
            print("\nWithout VAD filter:")
            segments, info = model.transcribe(
                audio_file,
                language='en',
                beam_size=5,
                vad_filter=False  # No VAD
            )
            
            segments_list = list(segments)
            print(f"  Segments found: {len(segments_list)}")
            text = " ".join([s.text.strip() for s in segments_list])
            print(f"  Text: {text[:200] if text else '(empty)'}")
            
            # Test WITH VAD filter
            print("\nWith VAD filter (aggressive):")
            segments, info = model.transcribe(
                audio_file,
                language='en',
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            segments_list = list(segments)
            print(f"  Segments found: {len(segments_list)}")
            text = " ".join([s.text.strip() for s in segments_list])
            print(f"  Text: {text[:200] if text else '(empty)'}")
            
            # Test WITH VAD filter (less aggressive)
            print("\nWith VAD filter (less aggressive):")
            segments, info = model.transcribe(
                audio_file,
                language='en',
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=1000,
                    threshold=0.3,  # Lower threshold
                    min_speech_duration_ms=100,  # Shorter minimum
                    speech_pad_ms=400
                )
            )
            
            segments_list = list(segments)
            print(f"  Segments found: {len(segments_list)}")
            text = " ".join([s.text.strip() for s in segments_list])
            print(f"  Text: {text[:200] if text else '(empty)'}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Also check file properties
    import wave
    print(f"\n=== Audio file properties ===")
    try:
        with wave.open(audio_file, 'rb') as wav:
            print(f"  Channels: {wav.getnchannels()}")
            print(f"  Sample rate: {wav.getframerate()}")
            print(f"  Duration: {wav.getnframes() / wav.getframerate():.2f} seconds")
            print(f"  Sample width: {wav.getsampwidth()} bytes")
    except Exception as e:
        print(f"  Error reading WAV: {e}")

if __name__ == "__main__":
    # Test with the recorded audio
    test_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_audio.wav"
    test_transcription(test_file)