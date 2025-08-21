#!/usr/bin/env python3
"""
Whisper FastAPI Service - Production Ready
Provides transcription and speaker diarization via REST API
"""

import os
import sys
import asyncio
import tempfile
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Whisper imports
from faster_whisper import WhisperModel
import numpy as np

# Optional diarization imports with detailed error reporting
DIARIZATION_AVAILABLE = False
DIARIZATION_ERROR = None
try:
    import torch
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
    print(f"✓ Diarization modules loaded (torch {torch.__version__})")
except ImportError as e:
    DIARIZATION_ERROR = str(e)
    print(f"Warning: Diarization not available - {e}")
    print("  To enable diarization:")
    print("  1. Run: ./setup_venv.sh")
    print("  2. Or install: pip install torch pyannote.audio")
except Exception as e:
    DIARIZATION_ERROR = str(e)
    print(f"Warning: Unexpected error loading diarization: {e}")

# Configuration from environment
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")  # Changed to tiny which works
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_DIARIZE = os.environ.get("WHISPER_DIARIZE", "true").lower() == "true"
WHISPER_DEFAULT_FORMAT = os.environ.get("WHISPER_DEFAULT_FORMAT", "json")

# Token management
TOKEN_FILE = Path.home() / ".config" / "whisper" / "token"
HF_TOKEN = None
if TOKEN_FILE.exists():
    try:
        with open(TOKEN_FILE) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token and token != "hf_PUT_YOUR_VALID_TOKEN_HERE":
                        HF_TOKEN = token
                        os.environ["HF_TOKEN"] = token
                        print(f"Loaded HF token: {token[:10]}...{token[-4:]}")
    except Exception as e:
        print(f"Warning: Could not load HF token: {e}")

# Initialize models
print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE} with {WHISPER_COMPUTE}")
model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

# Initialize diarization pipeline if available
diarization_pipeline = None
diarization_load_error = None

if DIARIZATION_AVAILABLE and WHISPER_DIARIZE:
    if not HF_TOKEN:
        diarization_load_error = "No HuggingFace token found"
        print("Warning: Diarization disabled - no HF token")
        print("  Add token to ~/.config/whisper/token")
    else:
        try:
            print(f"Loading diarization pipeline (token: {HF_TOKEN[:10]}...)")
            
            # Try different model versions based on pyannote version
            models_to_try = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization-3.0", 
                "pyannote/speaker-diarization@2.1"
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"  Trying model: {model_name}")
                    diarization_pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=HF_TOKEN
                    )
                    
                    # Move to GPU if available
                    if WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
                        diarization_pipeline.to(torch.device("cuda"))
                        print(f"✓ Diarization pipeline loaded on GPU: {model_name}")
                    else:
                        print(f"✓ Diarization pipeline loaded on CPU: {model_name}")
                    break
                    
                except Exception as model_error:
                    print(f"    Failed: {str(model_error)[:100]}")
                    diarization_load_error = str(model_error)
                    continue
                    
        except Exception as e:
            diarization_load_error = str(e)
            print(f"Error loading diarization: {e}")
            
            # Provide specific guidance based on error
            if "401" in str(e) or "Unauthorized" in str(e):
                print("\n  Fix: Token might be invalid or expired")
                print("  1. Get new token: https://huggingface.co/settings/tokens")
                print("  2. Update ~/.config/whisper/token")
            elif "403" in str(e) or "Forbidden" in str(e):
                print("\n  Fix: Accept model license")
                print("  1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("  2. Click 'Agree and access repository'")
            
            diarization_pipeline = None

# Job storage for async processing
jobs_storage: Dict[str, Dict] = {}

# FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="GPU-accelerated speech-to-text with speaker diarization",
    version="2.0.0"
)

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class TranscriptionRequest(BaseModel):
    audio_url: str
    language_code: Optional[str] = None
    speaker_labels: Optional[bool] = False
    num_speakers: Optional[int] = None
    format: Optional[str] = "json"

class JobResponse(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: Optional[str] = None
    text: Optional[str] = None
    segments: Optional[List[Dict]] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    diarize: bool = False,
    num_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """Core transcription function"""
    
    # Transcribe with Whisper (with fallback for empty results)
    segments, info = model.transcribe(
        audio_path,
        language=language or WHISPER_LANGUAGE,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=False,  # Disabled VAD - it was filtering everything
        without_timestamps=False,
        initial_prompt="Transcribe the following audio accurately: ",  # Help the model
        compression_ratio_threshold=2.4,  # More lenient
        log_prob_threshold=-1.0,  # More lenient
        no_speech_threshold=0.6  # More likely to detect speech
    )
    
    # Convert to list and extract text
    segments_list = []
    full_text = []
    
    for segment in segments:
        seg_dict = {
            "id": len(segments_list) + 1,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        segments_list.append(seg_dict)
        full_text.append(segment.text.strip())
    
    # Debug logging
    if not segments_list:
        print(f"WARNING: No segments found for {audio_path}")
        print(f"  Info - Language: {info.language}, Duration: {info.duration}")
        # Try to get at least something
        import wave
        try:
            with wave.open(audio_path, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                actual_duration = frames / float(rate)
                print(f"  WAV - Duration: {actual_duration:.2f}s, Frames: {frames}, Rate: {rate}")
        except Exception as e:
            print(f"  Could not read WAV: {e}")
    
    result = {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(full_text),
        "segments": segments_list
    }
    
    # Add speaker diarization if requested AND available
    if diarize:
        if not diarization_pipeline:
            print("Warning: Diarization requested but not available on server")
        elif not segments_list:
            print("Warning: No segments to diarize")
        else:
            try:
                print(f"Running speaker diarization (num_speakers={num_speakers})...")
                diarization = diarization_pipeline(
                    audio_path,
                    num_speakers=num_speakers
                )
            
            # Map segments to speakers
            for segment in segments_list:
                mid_time = (segment["start"] + segment["end"]) / 2
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= mid_time <= turn.end:
                        segment["speaker"] = speaker
                        break
                if "speaker" not in segment:
                    segment["speaker"] = "UNKNOWN"
            
            print("Diarization complete")
        except Exception as e:
            print(f"Diarization failed: {e}")
    
    return result

def format_output(result: Dict, format_type: str) -> str:
    """Format transcription output"""
    
    if format_type == "text":
        return result["text"]
    
    elif format_type == "vtt":
        lines = ["WEBVTT", ""]
        for seg in result["segments"]:
            start = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}.{int((seg['start']%1)*1000):03d}"
            end = f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}.{int((seg['end']%1)*1000):03d}"
            speaker = f"{seg.get('speaker', 'UNKNOWN')}: " if "speaker" in seg else ""
            lines.append(f"{start} --> {end}")
            lines.append(f"{speaker}{seg['text']}")
            lines.append("")
        return "\n".join(lines)
    
    elif format_type == "srt":
        lines = []
        for i, seg in enumerate(result["segments"], 1):
            start = f"{int(seg['start']//3600):02d}:{int(seg['start']//60%60):02d}:{int(seg['start']%60):02d},{int((seg['start']%1)*1000):03d}"
            end = f"{int(seg['end']//3600):02d}:{int(seg['end']//60%60):02d}:{int(seg['end']%60):02d},{int((seg['end']%1)*1000):03d}"
            speaker = f"{seg.get('speaker', 'UNKNOWN')}: " if "speaker" in seg else ""
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"{speaker}{seg['text']}")
            lines.append("")
        return "\n".join(lines)
    
    else:  # json
        return result

async def process_transcription_job(job_id: str, audio_path: str, params: Dict):
    """Background task for async transcription"""
    
    try:
        jobs_storage[job_id]["status"] = JobStatus.PROCESSING
        jobs_storage[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        result = transcribe_audio(
            audio_path,
            language=params.get("language"),
            diarize=params.get("diarize", False),
            num_speakers=params.get("num_speakers")
        )
        
        jobs_storage[job_id]["status"] = JobStatus.COMPLETED
        jobs_storage[job_id]["text"] = result["text"]
        jobs_storage[job_id]["segments"] = result["segments"]
        jobs_storage[job_id]["metadata"] = {
            "language": result["language"],
            "duration": result["duration"]
        }
        
    except Exception as e:
        jobs_storage[job_id]["status"] = JobStatus.ERROR
        jobs_storage[job_id]["error"] = str(e)
    
    finally:
        jobs_storage[job_id]["updated_at"] = datetime.utcnow().isoformat()
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Whisper Transcription API",
        "version": "2.0.0",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "features": {
            "transcription": "always available",
            "diarization": "available (optional per request)" if diarization_pipeline else "not available (install pyannote)",
            "diarization_loaded": bool(diarization_pipeline)
        },
        "endpoints": {
            "health": "/health",
            "transcribe_sync": "/v1/transcribe",
            "transcribe_async": "/v2/transcript"
        },
        "usage": {
            "basic": "POST /v1/transcribe with file",
            "with_diarization": "POST /v1/transcribe with file and diarize=true"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = {
        "ok": True,
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "diarization": {
            "modules_available": DIARIZATION_AVAILABLE,
            "pipeline_loaded": bool(diarization_pipeline),
            "token_present": bool(HF_TOKEN),
            "error": diarization_load_error or DIARIZATION_ERROR
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add CUDA info if available
    if DIARIZATION_AVAILABLE:
        try:
            import torch
            health_status["cuda"] = {
                "available": torch.cuda.is_available(),
                "version": torch.version.cuda if torch.cuda.is_available() else None,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except:
            pass
    
    return health_status

@app.post("/v1/transcribe")
async def transcribe_v1(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    diarize: Optional[bool] = Form(False),
    num_speakers: Optional[int] = Form(None),
    format: Optional[str] = Form("json")
):
    """Synchronous transcription endpoint"""
    
    # Get audio file
    if file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            audio_path = tmp.name
    
    elif audio_url:
        # Download from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(response.content)
                audio_path = tmp.name
    
    else:
        raise HTTPException(status_code=400, detail="No audio file or URL provided")
    
    try:
        # Process transcription
        result = transcribe_audio(
            audio_path,
            language=language,
            diarize=diarize,
            num_speakers=num_speakers
        )
        
        # Format output
        formatted = format_output(result, format)
        
        if format == "text":
            return PlainTextResponse(formatted)
        else:
            return JSONResponse(formatted if isinstance(formatted, dict) else {"output": formatted})
    
    finally:
        # Clean up
        try:
            os.unlink(audio_path)
        except:
            pass

@app.post("/v2/transcript")
async def create_transcript_v2(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """Asynchronous transcription endpoint (AssemblyAI compatible)"""
    
    # Generate job ID
    job_id = hashlib.md5(f"{request.audio_url}{time.time()}".encode()).hexdigest()
    
    # Create job entry
    jobs_storage[job_id] = {
        "id": job_id,
        "status": JobStatus.QUEUED,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": None,
        "text": None,
        "segments": None,
        "error": None,
        "metadata": None
    }
    
    # Download audio file
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.audio_url)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(response.content)
                audio_path = tmp.name
    except Exception as e:
        jobs_storage[job_id]["status"] = JobStatus.ERROR
        jobs_storage[job_id]["error"] = f"Failed to download audio: {str(e)}"
        return JobResponse(**jobs_storage[job_id])
    
    # Queue background task
    background_tasks.add_task(
        process_transcription_job,
        job_id,
        audio_path,
        {
            "language": request.language_code,
            "diarize": request.speaker_labels,
            "num_speakers": request.num_speakers
        }
    )
    
    return JobResponse(**jobs_storage[job_id])

@app.get("/v2/transcript/{job_id}")
async def get_transcript_v2(job_id: str):
    """Get async job status and results"""
    
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(**jobs_storage[job_id])

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info"
    )