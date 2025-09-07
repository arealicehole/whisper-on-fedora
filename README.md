# Whisper API with GPU Diarization (Blackwell)

## What You Get
Complete GPU-accelerated transcription + speaker diarization API. Everything runs in Docker - no local Python setup needed.

## Requirements
- NVIDIA GPU (optimized for Blackwell/RTX 5060 Ti)
- Docker with NVIDIA Container Toolkit
- HuggingFace token (for diarization models)

## What's Included in Docker Image
- **faster-whisper** - GPU-accelerated OpenAI Whisper
- **NeMo toolkit** - NVIDIA's diarization framework  
- **Sortformer v1** - Speaker identification (4 speakers max)
- **PyTorch 2.5.1** - With CUDA 12.4 support
- **FastAPI server** - Production-ready API
- All dependencies pre-installed and configured

## Quick Start
```bash
# Clone repo
git clone https://github.com/arealicehole/whisper-on-fedora.git
cd whisper-on-fedora

# Set your HuggingFace token
export HF_TOKEN=hf_xxxxx

# Run it (pulls image automatically)
docker compose -f docker-compose.blackwell.yml up

# Test it
python whisper_client.py --file audio.mp3 --diarize
```

## API Endpoints
- `POST /v1/transcribe` - Transcribe with optional diarization
- `GET /health` - Service health check
- Runs on port **8771**

## Docker Hub
```bash
docker pull arealicehole/whisper-blackwell:d4-n2
```

### Version History
- `d4-n2` - Fixed: NeMo toolkit properly installed for diarization
- `d4-n1` - Initial release (NeMo not installed)

GPU only, no CPU fallback. Models download automatically on first run.