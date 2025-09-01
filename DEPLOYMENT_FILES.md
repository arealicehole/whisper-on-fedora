# Whisper API Blackwell GPU Deployment Files

## Essential Files Required for Deployment

### 1. Docker Configuration
- `Dockerfile.blackwell` - Builds the container with NGC base image
- `docker-compose.blackwell.yml` - Orchestrates the service

### 2. Python Application Files  
- `main.py` (main_nemo.py renamed) - Main FastAPI application
- `nemo_diarizer.py` - Sortformer v1 diarization module
- `gpu_validator.py` - GPU validation and Blackwell detection

### 3. Environment Variables
Required in docker-compose.yml or .env:
```
HF_TOKEN=hf_xxx  # Your HuggingFace token for Sortformer model
WHISPER_MODEL=small
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
```

### 4. Model Files (Auto-downloaded on first run)
- Whisper model: Cached in container at `/root/.cache/whisper/`
- Sortformer v1: Cached at `/workspace/models/diar_sortformer_4spk-v1.nemo`

## Docker Image Details

Yes, this is a Docker image! Specifically:

**Base Image**: `nvcr.io/nvidia/pytorch:25.02-py3`
- NVIDIA NGC container with PyTorch 2.8.0+cu128
- Pre-optimized for Blackwell GPUs (sm_120)
- Includes CUDA 12.8 and cuDNN

**Built Image**: `whisper-blackwell:latest`
- Built from Dockerfile.blackwell
- Includes faster-whisper, NeMo toolkit, and all dependencies

## API Compatibility

The API is **99% compatible** with the original Whisper API:

### Endpoints (Same as original):
- `GET /health` - Health check with GPU status
- `POST /v1/transcribe` - Main transcription endpoint  
- `POST /v2/transcript` - Async transcription (for large files)

### Request Parameters (All the same):
```python
# All these work exactly as before:
- file: Audio file (WAV, MP3, M4A, etc.)
- response_format: "json", "text", "srt", "vtt", "verbose_json"
- language: Language code (optional)
- temperature: Sampling temperature
- prompt: Initial prompt
- timestamp_granularities: ["segment", "word"]

# Enhanced for diarization:
- diarize: true/false (enables speaker diarization)
- num_speakers: Optional number of speakers
```

### Response Format (Enhanced):
Original segments now include speaker field:
```json
{
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello everyone",
      "speaker": "SPEAKER_00"  // NEW: Speaker identification
    }
  ]
}
```

## Deployment Steps

1. **Copy these files to target machine:**
```bash
# Essential files
whisper-api/
├── Dockerfile.blackwell
├── docker-compose.blackwell.yml
├── main_nemo.py (rename to main.py in container)
├── nemo_diarizer.py
└── gpu_validator.py
```

2. **Build the image:**
```bash
docker compose -f docker-compose.blackwell.yml build
```

3. **Run the service:**
```bash
# Set your HF token
export HF_TOKEN=hf_xxx
docker compose -f docker-compose.blackwell.yml up -d
```

4. **Verify it's working:**
```bash
curl http://localhost:8771/health | jq .
```

## Key Differences from Original

1. **GPU-Only**: Refuses to run on CPU (enforced)
2. **Diarization**: NeMo Sortformer v1 speaker identification
3. **Blackwell Optimized**: Uses NGC container for sm_120 support
4. **Memory Efficient**: Better GPU memory management
5. **Port**: Runs on 8771 by default (configurable)

## Notes

- First run downloads models (~1-2GB)
- Requires NVIDIA GPU with 8GB+ VRAM
- Optimal with Blackwell GPUs (RTX 5060 Ti, etc.)
- HF token required for Sortformer model access