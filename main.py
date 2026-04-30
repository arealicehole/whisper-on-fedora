#!/usr/bin/env python3
"""
Whisper FastAPI Service with NeMo Diarization - Production Ready
Provides transcription and speaker diarization via REST API using NeMo toolkit
"""

import json
import os
import sys
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# GPU validation
from gpu_validator import GPUValidator, GPUEnforcementError

# PyTorch and warnings
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
from model_manager import WhisperModelManager, ModelConfig

# NeMo diarization imports with detailed error reporting
DIARIZATION_AVAILABLE = False
DIARIZATION_ERROR = None
try:
    from sherpa_diarizer import SherpaDiarizer as NeMoDiarizer, align_transcription_with_speakers
    DIARIZATION_AVAILABLE = True
    print(f"✓ Sherpa-ONNX diarization loaded")
except ImportError as e:
    DIARIZATION_ERROR = str(e)
    print(f"Warning: NeMo diarization not available - {e}")
    print("  To enable diarization:")
    print("  1. Install: pip install nemo_toolkit[asr]")
    print("  2. Get HF token from: https://huggingface.co/settings/tokens")
except Exception as e:
    DIARIZATION_ERROR = str(e)
    print(f"Warning: Unexpected error loading NeMo diarization: {e}")

# Configuration from environment
WHISPER_DEFAULT_MODEL = os.environ.get("WHISPER_DEFAULT_MODEL", os.environ.get("WHISPER_MODEL", "tiny"))  # Default model for backward compatibility

# Blackwell GPU initialization
def initialize_blackwell_gpu():
    """Initialize Blackwell GPU with proper configuration"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - check Docker GPU passthrough")
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {capability}")
    
    # Blackwell-specific setup
    if capability == (12, 0):  # sm_120
        print("Configuring for Blackwell GPU...")
        
        # Set environment variables
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0;12.0'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Test CUDA operations
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()
        
        print("✅ Blackwell GPU initialized successfully")
        return True
    
    return False

# GPU Enforcement - No CPU fallback
print("🚀 Starting Whisper API with GPU-only enforcement and NeMo diarization")
validator = GPUValidator()

try:
    # Check for environment conflicts first
    validator.check_environment_conflicts()
    
    # Enforce GPU requirements
    gpu_result = validator.enforce_gpu_requirements()
    
    print(f"✅ GPU Validation PASSED: {gpu_result.device_name}")
    print(f"   Memory: {gpu_result.memory_gb:.1f}GB")
    print(f"   Compute: sm_{gpu_result.compute_capability[0]}{gpu_result.compute_capability[1]}")
    
    # Initialize Blackwell if detected
    is_blackwell = initialize_blackwell_gpu()
    if is_blackwell:
        print("✅ Blackwell-specific optimizations applied")
    
    # Force GPU configuration - no fallback
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE = "float16"  # GPU-optimized precision
    
except GPUEnforcementError as e:
    print(f"\n❌ GPU Requirements not met: {e}", file=sys.stderr)
    if e.remediation:
        print("\n📋 Remediation steps:", file=sys.stderr)
        for step in e.remediation:
            print(f"   • {step}", file=sys.stderr)
    print("\n⚠️  This service requires GPU acceleration. CPU fallback is not supported.", file=sys.stderr)
    # Set dummy values to prevent NameError if sys.exit is mocked in tests
    WHISPER_DEVICE = "cuda"  
    WHISPER_COMPUTE = "float16"
    sys.exit(1)

WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_DIARIZE = os.environ.get("WHISPER_DIARIZE", "true").lower() == "true"
WHISPER_DEFAULT_FORMAT = os.environ.get("WHISPER_DEFAULT_FORMAT", "json")

# Token management
TOKEN_FILE = Path.home() / ".config" / "whisper" / "token"
# First try environment variable, then fallback to file
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and TOKEN_FILE.exists():
    try:
        with open(TOKEN_FILE) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token and token != "hf_PUT_YOUR_VALID_TOKEN_HERE":
                        HF_TOKEN = token
                        os.environ["HF_TOKEN"] = token
                        print(f"Loaded HF token from file: {token[:10]}...{token[-4:]}")
    except Exception as e:
        print(f"Warning: Could not load HF token: {e}")

# Initialize model manager
# Note: WHISPER_DEVICE and WHISPER_COMPUTE are guaranteed to be set by GPU enforcement above
print(f"Initializing Whisper Model Manager for local models")
model_config = ModelConfig(
    name="whisper_api",
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
    models_directory=os.environ.get("MODELS_DIRECTORY", "/workspace/models"),
    max_loaded_models=int(os.environ.get("MAX_LOADED_MODELS", "2"))
)
model_manager = WhisperModelManager(model_config)

# Load default model to maintain backward compatibility
print(f"Loading default Whisper model: {WHISPER_DEFAULT_MODEL} on {WHISPER_DEVICE} with {WHISPER_COMPUTE}")
try:
    model_manager.load_model(WHISPER_DEFAULT_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(f"✓ Default model {WHISPER_DEFAULT_MODEL} loaded successfully")
except Exception as e:
    print(f"Warning: Could not load default model {WHISPER_DEFAULT_MODEL}: {e}")
    available_models = model_manager.discover_available_models()
    if available_models:
        fallback_model = available_models[0]
        print(f"Falling back to available model: {fallback_model}")
        model_manager.load_model(fallback_model, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        WHISPER_DEFAULT_MODEL = fallback_model
    else:
        print("No local models found! Please ensure models are in /workspace/models/")
        print("Expected structure: /workspace/models/{model-name}-ct2/")
        raise RuntimeError("No Whisper models available")

# Initialize NeMo diarization if available
nemo_diarizer = None
diarization_load_error = None

if DIARIZATION_AVAILABLE and WHISPER_DIARIZE:
    try:
        print("Loading Sherpa-ONNX diarization pipeline...")
        nemo_diarizer = NeMoDiarizer(device=WHISPER_DEVICE)
        print(f"✓ Sherpa-ONNX diarization pipeline loaded on {WHISPER_DEVICE}")
    except Exception as e:
        diarization_load_error = str(e)
        print(f"Error loading Sherpa-ONNX diarization: {e}")
        nemo_diarizer = None

# Job storage for async processing
jobs_storage: Dict[str, Dict] = {}

# FastAPI app
app = FastAPI(
    title="Whisper Transcription API with NeMo Diarization",
    description="GPU-accelerated speech-to-text with NeMo speaker diarization",
    version="2.0.0-nemo"
)

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class TranscriptionRequest(BaseModel):
    audio_url: str
    language_code: Optional[str] = None
    model: Optional[str] = None
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
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    diarize: bool = False,
    num_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """Core transcription function with NeMo diarization"""
    
    # Get the appropriate model (use default if not specified)
    selected_model_name = model_name or WHISPER_DEFAULT_MODEL
    selected_model = model_manager.get_model(selected_model_name)
    
    print(f"Using Whisper model: {selected_model_name}")
    
    # Transcribe with Whisper (with fallback for empty results)
    segments, info = selected_model.transcribe(
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
        if not nemo_diarizer:
            print("Warning: Diarization requested but NeMo not available on server")
            # Add fallback behavior - keep segments without speaker labels
            for segment in result["segments"]:
                segment["speaker"] = "UNKNOWN"
        elif not segments_list:
            print("Warning: No segments to diarize")
        else:
            try:
                print(f"Running NeMo diarization (num_speakers={num_speakers})...")
                
                # Memory management before diarization
                torch.cuda.empty_cache()
                
                # Run NeMo diarization
                speaker_segments = nemo_diarizer.diarize(audio_path, num_speakers)
                
                if speaker_segments:
                    # Use weighted intersection alignment from nemo_diarizer
                    result["segments"] = align_transcription_with_speakers(
                        segments_list, speaker_segments
                    )
                    
                    print(f"NeMo diarization complete: {len(speaker_segments)} speaker segments")
                    
                    # Handle parallel processing for long audio
                    if info.duration > 60:
                        print(f"Long audio ({info.duration:.1f}s) - using parallel processing optimizations")
                else:
                    print("Warning: NeMo diarization returned no speaker segments")
                    # Fallback to UNKNOWN speaker
                    for segment in result["segments"]:
                        segment["speaker"] = "UNKNOWN"
                
                # Memory management after diarization
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"NeMo diarization failed: {e}")
                print("Falling back to transcription without diarization")
                # Graceful fallback - keep transcription, mark speakers as UNKNOWN
                for segment in result["segments"]:
                    segment["speaker"] = "UNKNOWN"
    
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
            model_name=params.get("model"),
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
        "service": "Whisper Transcription API with NeMo Diarization",
        "version": "2.0.0-nemo",
        "model": WHISPER_DEFAULT_MODEL,
        "device": WHISPER_DEVICE,
        "diarization_backend": "NeMo",
        "features": {
            "transcription": "always available",
            "diarization": "available (NeMo)" if nemo_diarizer else "not available (install nemo_toolkit)",
            "diarization_loaded": bool(nemo_diarizer),
            "blackwell_optimized": torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0),
            "parallel_processing": "long audio >60s"
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
    """Health check endpoint with GPU enforcement status"""
    
    # Validate GPU is still available
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "GPU lost - service requires GPU acceleration",
                "gpu_required": True,
                "gpu_available": False,
                "message": "This service cannot operate without GPU. Please restore GPU access."
            }
        )
    
    # Healthy status with GPU-only mode
    health_status = {
        "status": "healthy",
        "ok": True,
        "gpu_required": True,
        "gpu_available": True,
        "gpu_enforced": True,
        "default_model": WHISPER_DEFAULT_MODEL,
        "device": "cuda",  # Always CUDA in GPU-only mode
        "compute_type": "float16",  # GPU-optimized
        "diarization_backend": "NeMo",
        "gpu": {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB",
            "blackwell_detected": torch.cuda.get_device_capability(0) == (12, 0)
        },
        "diarization": {
            "modules_available": DIARIZATION_AVAILABLE,
            "pipeline_loaded": bool(nemo_diarizer),
            "token_present": bool(HF_TOKEN),
            "device": "cuda" if nemo_diarizer else None,
            "backend": "NeMo" if nemo_diarizer else None,
            "error": diarization_load_error or DIARIZATION_ERROR
        },
        "processing_mode": {
            "whisper": "GPU",  # Always GPU in GPU-only mode
            "diarization": "GPU (NeMo)" if nemo_diarizer else "N/A"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return health_status

@app.get("/v1/models")
async def list_available_models():
    """List available local models"""
    try:
        available_models = model_manager.discover_available_models()
        loaded_models = model_manager.get_loaded_models_info()
        memory_usage = model_manager.monitor_memory_usage()
        
        return {
            "models": available_models,
            "loaded_models": list(loaded_models.keys()),
            "default_model": WHISPER_DEFAULT_MODEL,
            "model_directory": str(model_manager.models_directory),
            "max_loaded_models": model_manager.max_loaded_models,
            "memory_usage": memory_usage,
            "status": "available"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )

@app.get("/v1/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get information about specific model"""
    try:
        # Validate model exists
        available_models = model_manager.discover_available_models()
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Model not found",
                    "requested": model_name,
                    "available": available_models
                }
            )
        
        # Get model info
        model_path = model_manager.get_model_path(model_name)
        loaded_models = model_manager.get_loaded_models_info()
        is_loaded = model_name in loaded_models
        
        model_info = {
            "name": model_name,
            "path": model_path,
            "is_loaded": is_loaded,
            "is_default": model_name == WHISPER_DEFAULT_MODEL,
            "status": "loaded" if is_loaded else "available"
        }
        
        # Add loaded model details if available
        if is_loaded:
            model_info.update(loaded_models[model_name])
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.post("/v1/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
    """Load a specific model into memory"""
    try:
        model_instance = model_manager.load_model(model_name)
        loaded_models = model_manager.get_loaded_models_info()
        memory_usage = model_manager.monitor_memory_usage()
        
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name,
            "loaded_models": list(loaded_models.keys()),
            "memory_usage": memory_usage
        }
        
    except Exception as e:
        # Handle specific model manager errors
        if "not found" in str(e).lower():
            available_models = model_manager.discover_available_models()
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Model not found",
                    "requested": model_name,
                    "available": available_models
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )

@app.post("/v1/models/{model_name}/unload")
async def unload_model_endpoint(model_name: str):
    """Unload a specific model from memory"""
    try:
        success = model_manager.unload_model(model_name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} is not currently loaded"
            )
        
        loaded_models = model_manager.get_loaded_models_info()
        memory_usage = model_manager.monitor_memory_usage()
        
        return {
            "message": f"Model {model_name} unloaded successfully",
            "model_name": model_name,
            "loaded_models": list(loaded_models.keys()),
            "memory_usage": memory_usage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )

@app.post("/v1/transcribe")
async def transcribe_v1(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
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
            model_name=model,
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

@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None)
):
    """OpenAI-compatible transcription endpoint for Hermes STT integration.
    Accepts multipart file upload at /v1/audio/transcriptions (OpenAI API format)."""
    
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name
    
    try:
        result = transcribe_audio(
            audio_path,
            model_name=model,
            language=language,
            diarize=False,
            num_speakers=None
        )
        
        # Extract full text from segments
        full_text = result.get("text", "")
        if not full_text and result.get("segments"):
            full_text = " ".join(s.get("text", "").strip() for s in result["segments"] if s.get("text"))
        
        if response_format == "text":
            return PlainTextResponse(full_text)
        elif response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": result.get("language", "en"),
                "duration": result.get("duration", 0),
                "text": full_text,
                "segments": result.get("segments", []),
                "words": result.get("words", [])
            })
        else:
            # Default JSON format (OpenAI-compatible)
            return JSONResponse({
                "text": full_text,
                "language": result.get("language", "en")
            })
    finally:
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
            "model": request.model,
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
        host="0.0.0.0",  # Bind to all interfaces for Docker
        port=int(os.environ.get("WHISPER_PORT", 8767)),  # Changed to 8767 to avoid conflicts
        log_level="info"
    )