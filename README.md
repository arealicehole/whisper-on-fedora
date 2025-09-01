# Whisper API with GPU Diarization (Blackwell)

## What's Here
- `Dockerfile.blackwell` - Docker image for Blackwell GPU (RTX 5060 Ti)
- `docker-compose.blackwell.yml` - Compose config (port 8771)
- `main.py` - FastAPI server with whisper + NeMo diarization
- `nemo_diarizer.py` - Sortformer v1 GPU diarization (4 speakers max)
- `gpu_validator.py` - GPU validation & Blackwell detection
- `whisper_client.py` - Test client

## Quick Start
```bash
# Set your HuggingFace token
export HF_TOKEN=hf_xxxxx

# Run it
docker compose -f docker-compose.blackwell.yml up

# Test it
python whisper_client.py --file audio.mp3 --diarize
```

## Docker Hub
```bash
docker pull arealicehole/whisper-blackwell:d4-n1
```

Supports up to 4 speakers. Uses NVIDIA NGC container. GPU only, no CPU fallback.