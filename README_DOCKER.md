# Docker Deployment with NeMo Diarization

## Docker Hub
```bash
docker pull arealicehole/whisper-blackwell:d4-n2
```

### Version History
- `d4-n2` - Fixed: NeMo toolkit properly installed for diarization  
- `d4-n1` - Initial release (NeMo not installed)

## Quick Start with Docker Compose
```bash
# Set your HuggingFace token
export HF_TOKEN=hf_xxxxx

# Run with docker-compose
docker compose -f docker-compose.blackwell.yml up
```

## What's Fixed in d4-n2
- NeMo toolkit properly installed
- Speaker diarization fully functional
- Config generated at runtime if needed
- ~600MB larger image due to NeMo dependencies

GPU only, no CPU fallback. Models download automatically on first run.