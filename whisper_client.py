#!/usr/bin/env python3
"""
Whisper API Client Library
Can be used from any Python version (3.8+) to call the Whisper API service

Example usage:
    from whisper_client import WhisperClient
    
    client = WhisperClient()
    result = client.transcribe("meeting.wav", diarize=True)
    
    for segment in result['segments']:
        print(f"{segment['speaker']}: {segment['text']}")
"""

import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import time


class WhisperClient:
    def __init__(self, api_url: str = "http://localhost:8765"):
        """Initialize Whisper API client
        
        Args:
            api_url: Base URL of the Whisper API service
        """
        self.api_url = api_url.rstrip('/')
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is running and get status"""
        try:
            response = requests.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": str(e)}
    
    def transcribe(
        self,
        audio_file: str,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        format: Literal["json", "text", "srt", "vtt"] = "json"
    ) -> Any:
        """Transcribe an audio file with optional speaker diarization
        
        Args:
            audio_file: Path to audio file
            diarize: Enable speaker diarization
            num_speakers: Number of speakers (optional, helps diarization)
            language: Language code (e.g., 'en', 'es', 'fr')
            format: Output format
            
        Returns:
            Transcription result (dict for json, string for other formats)
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f, 'audio/wav')}
            data = {
                'diarize': str(diarize).lower(),
                'format': format
            }
            
            if num_speakers:
                data['num_speakers'] = str(num_speakers)
            if language:
                data['language'] = language
                
            response = requests.post(
                f"{self.api_url}/v1/transcribe",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            if format == "json":
                return response.json()
            else:
                return response.text
    
    def transcribe_url(
        self,
        audio_url: str,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        wait: bool = True,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Transcribe audio from URL using async endpoint
        
        Args:
            audio_url: URL of audio file
            diarize: Enable speaker diarization
            num_speakers: Number of speakers
            language: Language code
            wait: Wait for completion
            timeout: Max seconds to wait
            
        Returns:
            Transcription result
        """
        # Start async job
        request_data = {
            "audio_url": audio_url,
            "speaker_labels": diarize,
            "format": "json"
        }
        
        if num_speakers:
            request_data["num_speakers"] = num_speakers
        if language:
            request_data["language_code"] = language
            
        response = requests.post(
            f"{self.api_url}/v2/transcript",
            json=request_data
        )
        response.raise_for_status()
        job = response.json()
        
        if not wait:
            return job
            
        # Poll for completion
        job_id = job['id']
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_url}/v2/transcript/{job_id}")
            response.raise_for_status()
            result = response.json()
            
            if result['status'] in ['completed', 'error']:
                return result
                
            time.sleep(2)
            
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def format_transcript(self, result: Dict[str, Any], style: str = "dialogue") -> str:
        """Format transcription result for display
        
        Args:
            result: Transcription result from transcribe()
            style: Format style ('dialogue', 'timeline', 'speakers')
            
        Returns:
            Formatted transcript string
        """
        if 'segments' not in result:
            return result.get('text', '')
            
        segments = result['segments']
        
        if style == "dialogue":
            # Group consecutive segments by speaker
            output = []
            current_speaker = None
            current_text = []
            
            for seg in segments:
                speaker = seg.get('speaker', 'UNKNOWN')
                if speaker != current_speaker:
                    if current_text:
                        output.append(f"{current_speaker}: {' '.join(current_text)}")
                    current_speaker = speaker
                    current_text = [seg['text']]
                else:
                    current_text.append(seg['text'])
                    
            if current_text:
                output.append(f"{current_speaker}: {' '.join(current_text)}")
                
            return '\n\n'.join(output)
            
        elif style == "timeline":
            output = []
            for seg in segments:
                speaker = seg.get('speaker', 'UNKNOWN')
                start = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
                output.append(f"[{start}] {speaker}: {seg['text']}")
            return '\n'.join(output)
            
        elif style == "speakers":
            # Group all text by speaker
            speakers = {}
            for seg in segments:
                speaker = seg.get('speaker', 'UNKNOWN')
                if speaker not in speakers:
                    speakers[speaker] = []
                speakers[speaker].append(seg['text'])
                
            output = []
            for speaker, texts in speakers.items():
                output.append(f"=== {speaker} ===")
                output.append(' '.join(texts))
                output.append('')
                
            return '\n'.join(output)
            
        else:
            return json.dumps(segments, indent=2)


# CLI interface when run directly
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper API Client")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--speakers", type=int, help="Number of speakers")
    parser.add_argument("--language", help="Language code (e.g., 'en')")
    parser.add_argument("--format", choices=["json", "text", "srt", "vtt", "dialogue", "timeline"], 
                       default="dialogue", help="Output format")
    parser.add_argument("--api-url", default="http://localhost:8765", help="API URL")
    
    args = parser.parse_args()
    
    client = WhisperClient(args.api_url)
    
    # Check service health
    health = client.health_check()
    if not health.get('ok'):
        print(f"Error: Service not available - {health.get('error')}")
        sys.exit(1)
        
    if args.diarize and not health.get('diarization', {}).get('pipeline_loaded'):
        print("Warning: Diarization requested but not available on server")
        
    # Transcribe
    try:
        if args.format in ["dialogue", "timeline"]:
            result = client.transcribe(
                args.audio_file,
                diarize=args.diarize,
                num_speakers=args.speakers,
                language=args.language,
                format="json"
            )
            print(client.format_transcript(result, style=args.format))
        else:
            result = client.transcribe(
                args.audio_file,
                diarize=args.diarize,
                num_speakers=args.speakers,
                language=args.language,
                format=args.format if args.format != "dialogue" else "json"
            )
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)