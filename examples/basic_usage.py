#!/usr/bin/env python3
"""
Basic usage examples for Whisper API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_client import WhisperClient
import json

def example_basic_transcription():
    """Simple transcription without speaker detection"""
    print("=== Basic Transcription ===")
    
    client = WhisperClient()
    
    # Check service health
    health = client.health_check()
    if not health.get('ok'):
        print(f"Service not available: {health}")
        return
    
    # Transcribe audio
    result = client.transcribe("sample.wav")
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Duration: {result['duration']} seconds")

def example_with_diarization():
    """Transcription with speaker identification"""
    print("\n=== Transcription with Speaker Diarization ===")
    
    client = WhisperClient()
    
    # Transcribe with speaker detection
    result = client.transcribe(
        "meeting.wav",
        diarize=True,
        num_speakers=2  # Optional: if you know the number of speakers
    )
    
    # Show results grouped by speaker
    speakers = {}
    for segment in result['segments']:
        speaker = segment.get('speaker', 'UNKNOWN')
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(segment['text'])
    
    for speaker, texts in speakers.items():
        print(f"\n{speaker}:")
        print(" ".join(texts))

def example_format_transcript():
    """Different ways to format the output"""
    print("\n=== Formatting Options ===")
    
    client = WhisperClient()
    result = client.transcribe("conversation.wav", diarize=True)
    
    # Dialogue format
    print("\n1. Dialogue Format:")
    print(client.format_transcript(result, style="dialogue"))
    
    # Timeline format
    print("\n2. Timeline Format:")
    print(client.format_transcript(result, style="timeline"))
    
    # Speakers summary
    print("\n3. Speakers Summary:")
    print(client.format_transcript(result, style="speakers"))

def example_async_transcription():
    """Using async endpoint for long files"""
    print("\n=== Async Transcription ===")
    
    client = WhisperClient()
    
    # Start async job
    job = client.transcribe_url(
        "https://example.com/long-audio.wav",
        diarize=True,
        wait=False  # Don't wait for completion
    )
    
    print(f"Job ID: {job['id']}")
    print(f"Status: {job['status']}")
    
    # Later, check status
    import time
    time.sleep(5)
    
    # Get results
    result = client.transcribe_url(
        "https://example.com/long-audio.wav",
        diarize=True,
        wait=True,  # Wait for completion
        timeout=300  # 5 minutes timeout
    )
    
    if result['status'] == 'completed':
        print(f"Transcription complete: {result['text'][:100]}...")

def example_export_formats():
    """Export in different subtitle formats"""
    print("\n=== Export Formats ===")
    
    client = WhisperClient()
    
    # Get SRT subtitles
    srt = client.transcribe("video.wav", format="srt")
    with open("subtitles.srt", "w") as f:
        f.write(srt)
    print("Saved subtitles.srt")
    
    # Get VTT subtitles
    vtt = client.transcribe("video.wav", format="vtt")
    with open("subtitles.vtt", "w") as f:
        f.write(vtt)
    print("Saved subtitles.vtt")

if __name__ == "__main__":
    print("Whisper API Usage Examples")
    print("=" * 40)
    
    # Note: Replace with actual audio files
    print("\nNote: These examples assume you have audio files:")
    print("  - sample.wav")
    print("  - meeting.wav") 
    print("  - conversation.wav")
    print("  - video.wav")
    print("\nAnd that the Whisper API is running on http://localhost:8767")
    
    # Uncomment to run examples:
    # example_basic_transcription()
    # example_with_diarization()
    # example_format_transcript()
    # example_async_transcription()
    # example_export_formats()