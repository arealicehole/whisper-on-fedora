#!/usr/bin/env python3
"""
Whisper FastAPI Service - Docker Version with Diarization Support
Optimized for CUDA 12.4 with PyAnnote compatibility fixes
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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# PyTorch imports
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import httpx

# Whisper imports
from faster_whisper import WhisperModel

# Try to import PyAnnote with compatibility fixes
DIARIZATION_AVAILABLE = False
DIARIZATION_ERROR = None
diarization_pipeline = None

try:
    # Check torchaudio compatibility
    import torchaudio
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Try to import PyAnnote
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
    print("✅ PyAnnote diarization imported successfully")
except ImportError as e:
    DIARIZATION_ERROR = str(e)
    print(f"⚠️ PyAnnote not available: {e}")
except Exception as e:
    DIARIZATION_ERROR = str(e)
    print(f"⚠️ PyAnnote error: {e}")

# Configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16" if WHISPER_DEVICE == "cuda" else "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_PORT = int(os.environ.get("WHISPER_PORT", "8767"))

# HuggingFace token for PyAnnote
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # Try to load from file
    token_file = Path.home() / ".config" / "whisper" / "token"
    if token_file.exists():
        try:
            with open(token_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        HF_TOKEN = line.split("=", 1)[1].strip()
                        break
        except:
            pass

# GPU info
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    print(f"CUDA version: {torch.version.cuda}")
    try:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    except:
        print("cuDNN version: Not available")

# Initialize Whisper model
print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE} with {WHISPER_COMPUTE}")
try:
    model = WhisperModel(
        WHISPER_MODEL, 
        device=WHISPER_DEVICE, 
        compute_type=WHISPER_COMPUTE,
        download_root="/app/models"
    )
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    sys.exit(1)

# Initialize diarization if available and token is present
if DIARIZATION_AVAILABLE and HF_TOKEN:
    try:
        print(f"Loading PyAnnote diarization pipeline...")
        print(f"  Token: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]}")
        
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
            print("✅ PyAnnote diarization loaded on GPU")
        else:
            print("✅ PyAnnote diarization loaded on CPU")
            
    except Exception as e:
        print(f"⚠️ Failed to load diarization pipeline: {e}")
        diarization_pipeline = None
        DIARIZATION_ERROR = str(e)
        
        # Common error guidance
        if "401" in str(e) or "Unauthorized" in str(e):
            print("  Fix: Token might be invalid or expired")
            print("  1. Get token: https://huggingface.co/settings/tokens")
            print("  2. Set HF_TOKEN environment variable")
        elif "403" in str(e) or "Forbidden" in str(e):
            print("  Fix: Accept model license")
            print("  1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  2. Click 'Agree and access repository'")
elif DIARIZATION_AVAILABLE and not HF_TOKEN:
    print("⚠️ Diarization available but no HF_TOKEN found")
    print("  Set HF_TOKEN environment variable to enable diarization")

# FastAPI app
app = FastAPI(
    title="Whisper API with Diarization (Docker)",
    description="GPU-accelerated speech-to-text with speaker diarization",
    version="3.0.0"
)

def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    diarize: bool = False,
    num_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """Core transcription function with optional diarization"""
    
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
    
    # Convert to list
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
    
    # Add speaker diarization if requested and available
    if diarize and diarization_pipeline and segments_list:
        try:
            print(f"Running diarization (num_speakers={num_speakers})...")
            
            # Run diarization
            diarization = diarization_pipeline(
                audio_path,
                num_speakers=num_speakers
            )
            
            # Map segments to speakers
            for segment in segments_list:
                mid_time = (segment["start"] + segment["end"]) / 2
                
                # Find speaker at this time
                speaker_found = False
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= mid_time <= turn.end:
                        segment["speaker"] = speaker
                        speaker_found = True
                        break
                
                if not speaker_found:
                    segment["speaker"] = "UNKNOWN"
            
            print("✅ Diarization complete")
            result["diarization_applied"] = True
            
        except Exception as e:
            print(f"⚠️ Diarization failed: {e}")
            result["diarization_error"] = str(e)
            result["diarization_applied"] = False
    elif diarize and not diarization_pipeline:
        result["diarization_applied"] = False
        result["diarization_error"] = "Diarization not available"
    
    return result

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Whisper API with Diarization (Docker)",
        "version": "3.0.0",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "diarization": {
            "available": bool(diarization_pipeline),
            "library_loaded": DIARIZATION_AVAILABLE,
            "token_present": bool(HF_TOKEN),
            "error": DIARIZATION_ERROR
        },
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/transcribe"
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
        "diarization": {
            "available": bool(diarization_pipeline),
            "modules_available": DIARIZATION_AVAILABLE,
            "pipeline_loaded": bool(diarization_pipeline),
            "token_present": bool(HF_TOKEN),
            "error": DIARIZATION_ERROR
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if torch.cuda.is_available():
        health_status["gpu"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB",
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
    """Transcription endpoint with optional diarization"""
    
    # Get audio file
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            audio_path = tmp.name
    elif audio_url:
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
        if format == "text":
            return PlainTextResponse(result["text"])
        elif format == "json":
            return JSONResponse(result)
        else:
            # Handle other formats if needed
            return JSONResponse(result)
    
    finally:
        # Clean up
        try:
            os.unlink(audio_path)
        except:
            pass

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting Whisper API on port {WHISPER_PORT}")
    print(f"Diarization: {'ENABLED' if diarization_pipeline else 'DISABLED'}")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=WHISPER_PORT,
        log_level="info"
    )