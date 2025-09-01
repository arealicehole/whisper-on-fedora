#!/usr/bin/env python3
"""
Whisper FastAPI Service - Docker Version with CUDA 12.4
Optimized for Blackwell GPU using compatible cuDNN version
"""

import os
import sys
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# PyTorch imports
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import httpx

# Whisper imports
from faster_whisper import WhisperModel

# Configuration from environment
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_PORT = int(os.environ.get("WHISPER_PORT", "8767"))

# Check GPU availability
print(f"🚀 Starting Whisper API in Docker with CUDA 12.4")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    
    # Test CUDA operations
    try:
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()
        print("✅ CUDA operations working")
    except Exception as e:
        print(f"⚠️  CUDA test failed: {e}")
else:
    print("⚠️  No GPU detected - falling back to CPU")
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE = "int8"

# Initialize model
print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE} with {WHISPER_COMPUTE}")
try:
    model = WhisperModel(
        WHISPER_MODEL, 
        device=WHISPER_DEVICE, 
        compute_type=WHISPER_COMPUTE,
        download_root="/app/models"  # Use Docker volume for models
    )
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    sys.exit(1)

# Job storage for async processing
jobs_storage: Dict[str, Dict] = {}

# FastAPI app
app = FastAPI(
    title="Whisper Transcription API (Docker)",
    description="GPU-accelerated speech-to-text with CUDA 12.4",
    version="2.1.0"
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
    
    # Transcribe with Whisper
    segments, info = model.transcribe(
        audio_path,
        language=language or WHISPER_LANGUAGE,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=2000,
            speech_pad_ms=400,
        ),
        without_timestamps=False,
        initial_prompt="",
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6
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
    
    result = {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(full_text),
        "segments": segments_list
    }
    
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
            lines.append(f"{start} --> {end}")
            lines.append(seg['text'])
            lines.append("")
        return "\n".join(lines)
    
    elif format_type == "srt":
        lines = []
        for i, seg in enumerate(result["segments"], 1):
            start = f"{int(seg['start']//3600):02d}:{int(seg['start']//60%60):02d}:{int(seg['start']%60):02d},{int((seg['start']%1)*1000):03d}"
            end = f"{int(seg['end']//3600):02d}:{int(seg['end']//60%60):02d}:{int(seg['end']%60):02d},{int((seg['end']%1)*1000):03d}"
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(seg['text'])
            lines.append("")
        return "\n".join(lines)
    
    else:  # json
        return result

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Whisper Transcription API (Docker)",
        "version": "2.1.0",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "endpoints": {
            "health": "/health",
            "transcribe_sync": "/v1/transcribe"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "ok": True,  # For recall compatibility
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if torch.cuda.is_available():
        health_status["gpu"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB",
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
        }
    
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

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=WHISPER_PORT,
        log_level="info"
    )