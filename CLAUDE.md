# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Whisper API Project Overview

This is a FastAPI-based Whisper transcription service that provides GPU-accelerated speech-to-text capabilities with optional speaker diarization. The service exposes REST API endpoints for both synchronous and asynchronous audio transcription.

## Quick Usage Guide

### Starting the Service
```bash
# One-command start (handles environment automatically)
./start_whisper.sh start

# Check if running
./start_whisper.sh status
```

### Using the Service

**Basic Transcription (no speakers):**
```bash
curl -X POST http://localhost:8765/v1/transcribe -F "file=@audio.wav"
```

**With Speaker Identification:**
```bash
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=true"
```

**From Python (any version):**
```python
from whisper_client import WhisperClient
client = WhisperClient()
result = client.transcribe("audio.wav", diarize=True)
```

**Key Point**: Diarization is optional per request - use `diarize=false` (default) for speed, `diarize=true` for speaker identification.

## Core Architecture

### Main Components

1. **FastAPI Application** (`main.py`): 
   - RESTful API service with two transcription modes (sync/async)
   - Uses `faster-whisper` for transcription (GPU-accelerated)
   - Optional speaker diarization via `pyannote.audio` pipeline
   - Job queue system for async processing

2. **Test Script** (`test_transcribe.py`): 
   - Debugging utility for testing transcription with different settings
   - Tests VAD (Voice Activity Detection) filter configurations

### Key Dependencies

- `faster-whisper`: GPU-accelerated Whisper implementation
- `FastAPI`: Web framework for REST API
- `pyannote.audio`: Speaker diarization (optional)
- `httpx`: Async HTTP client for downloading audio
- `uvicorn`: ASGI server
- `numpy`: Audio processing
- CUDA/cuDNN: GPU acceleration libraries

### Environment Configuration

The service uses environment variables for configuration:
- `WHISPER_MODEL`: Model size (default: "tiny")
- `WHISPER_DEVICE`: Computing device ("cuda" or "cpu")
- `WHISPER_COMPUTE`: Compute type (e.g., "float16")
- `WHISPER_LANGUAGE`: Default language (default: "en")
- `WHISPER_DIARIZE`: Enable speaker diarization (default: "true")
- `WHISPER_DEFAULT_FORMAT`: Default output format (default: "json")

### Authentication

- HuggingFace token required for diarization features
- Token stored in `~/.config/whisper/token` file
- Format: `HF_TOKEN=hf_xxxxx`

## Common Development Tasks

### Running the Service

```bash
# Start the API server
python main.py
# Server runs on http://127.0.0.1:8765
```

### Testing Transcription

```bash
# Test with a specific audio file
python test_transcribe.py /path/to/audio.wav
```

### API Endpoints

- `GET /`: Service information
- `GET /health`: Health check
- `POST /v1/transcribe`: Synchronous transcription
- `POST /v2/transcript`: Asynchronous transcription (AssemblyAI compatible)
- `GET /v2/transcript/{job_id}`: Get async job status/results

### Important Transcription Settings

The service has been configured with specific settings to improve transcription accuracy:
- VAD filter disabled by default (was filtering out valid speech)
- Initial prompt added to guide model
- More lenient thresholds for compression ratio and log probability
- Lower no-speech threshold for better speech detection

### Output Formats

Supports multiple output formats via the `format` parameter:
- `json`: Full transcription with segments and metadata
- `text`: Plain text output
- `vtt`: WebVTT subtitle format
- `srt`: SRT subtitle format

## Development Notes

- The service uses tempfile for handling uploaded/downloaded audio files
- Background tasks handle async transcription jobs
- Job results are stored in memory (`jobs_storage` dictionary)
- Audio files are cleaned up after processing
- CUDA library paths are configured in test script for GPU support

## Speaker Diarization Setup

Diarization requires specific version combinations due to pyannote compatibility issues:

### Quick Setup
```bash
# Option 1: Install in current environment (Python 3.11 recommended)
./install_diarization.sh

# Option 2: Create dedicated virtual environment
./setup_venv.sh
source ~/.venvs/whisper-diarize/bin/activate
```

### Known Working Combinations
- Python 3.11 + torch 2.2.0 + pyannote.audio 3.1.1 (recommended)
- Python 3.10 + torch 2.1.0 + pyannote.audio 3.0.1
- Python 3.12 + torch 2.3.0 + pyannote.audio 3.1.1 (may have issues)

### Troubleshooting Diarization

1. **Test current setup:**
   ```bash
   python test_diarization.py
   ```

2. **Common issues:**
   - Missing HF token: Add to `~/.config/whisper/token`
   - License not accepted: Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Version conflicts: Use setup_venv.sh for clean environment

3. **Check service health:**
   ```bash
   curl http://localhost:8765/health | jq .diarization
   ```

The service will attempt multiple model versions and provide detailed error messages if diarization fails to load.