# Whisper API with GPU Diarization & Dynamic Model Selection

Complete GPU-accelerated transcription + speaker diarization API with **local model storage** and **dynamic model switching**. Everything runs in Docker - no local Python setup needed.

## ✨ Key Features

- 🚀 **Local Model Storage** - No internet downloads during transcription
- 🔄 **Dynamic Model Selection** - Switch between models via API parameter  
- 🎯 **GPU-Only Processing** - Optimized for Blackwell RTX 5060 Ti (16GB VRAM)
- 🎤 **Speaker Diarization** - NVIDIA NeMo with up to 4 speakers
- 📦 **Docker Ready** - Complete containerized solution
- 🔧 **Production API** - FastAPI with health checks and monitoring

## Requirements
- NVIDIA GPU (optimized for Blackwell/RTX 5060 Ti)
- Docker with NVIDIA Container Toolkit  
- HuggingFace token (for diarization models only)
- 2GB+ available GPU memory

## Quick Start

### 1. Get HuggingFace Token (for diarization)
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Accept terms at https://huggingface.co/nvidia/diar_sortformer_4spk-v1

### 2. Run with Docker
```bash
# Set your token  
export HF_TOKEN=hf_xxxxx

# Clone and run
git clone https://github.com/arealicehole/whisper-on-fedora.git
cd whisper-on-fedora
docker compose -f docker-compose.blackwell.yml up
```

That's it! API is now running at http://localhost:8771

### 3. Test with Model Selection
```bash
# Use tiny model (fastest)
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "model=tiny"

# Use small model (better quality)  
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "model=small"
```

## API Usage

### Basic Transcription
```bash
# Use default model (tiny)
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@podcast.mp3"

# Specify model for better quality
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@podcast.mp3" \
  -F "model=small"
```

### Model Management
```bash
# List available models
curl http://localhost:8771/v1/models

# Get model information  
curl http://localhost:8771/v1/models/small/info

# Pre-load a model into memory
curl -X POST http://localhost:8771/v1/models/base/load

# Unload a model to free memory
curl -X POST http://localhost:8771/v1/models/base/unload
```

### With Speaker Labels
```bash
# Add speaker identification
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@meeting.mp3" \
  -F "diarize=true" \
  -F "model=small"
```

### Output Formats
```bash
# JSON (default) - Full details with timestamps
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=json"

# Plain text - Just the words
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=text"

# SRT subtitles - For video
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=srt" > subtitles.srt

# WebVTT subtitles - For web video
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=vtt" > captions.vtt
```

## API Parameters

| Parameter | Values | Description | Default |
|-----------|--------|-------------|---------|
| `file` | Any audio/video file | MP3, WAV, M4A, MP4, etc | Required |
| `model` | `tiny`, `base`, `small` | Local model to use | `tiny` |
| `format` | `json`, `text`, `srt`, `vtt` | Output format | `json` |
| `diarize` | `true`, `false` | Enable speaker labels | `false` |
| `language` | `en`, `es`, `fr`, etc | Audio language | `en` |

### Available Models
| Model | Size | VRAM | Speed | Quality | Use Case |
|-------|------|------|-------|---------|----------|
| `tiny` | 39MB | 1GB | Fastest | Good | Draft transcripts, real-time |
| `base` | 74MB | 1GB | Fast | Better | General purpose |  
| `small` | 244MB | 2GB | Medium | Best | Production quality |

**Note**: Only `tiny`, `base`, and `small` models are included in the Docker image. Larger models require manual download due to authentication requirements.

## Response Examples

### JSON Response (default)
```json
{
  "text": "Hello, this is a test recording.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test recording.",
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "en",
  "duration": 2.5
}
```

### Text Response
```
Hello, this is a test recording.
```

### SRT Response
```
1
00:00:00,000 --> 00:00:02,500
[SPEAKER_00]: Hello, this is a test recording.
```

## Python Client

```python
# Install client
pip install httpx

# Use it
from whisper_client import WhisperClient

client = WhisperClient("http://localhost:8771")

# Simple transcription with model selection
result = client.transcribe("audio.mp3", model="small")
print(result["text"])

# With speakers and specific model
result = client.transcribe("meeting.mp3", diarize=True, model="base")
for segment in result["segments"]:
    print(f"{segment['speaker']}: {segment['text']}")

# Check available models
import httpx
response = httpx.get("http://localhost:8771/v1/models")
models = response.json()
print(f"Available models: {models['models']}")
print(f"Loaded models: {models['loaded_models']}")
```

## Model Management API

### List Models
```bash
curl http://localhost:8771/v1/models
```
**Response:**
```json
{
  "models": ["tiny", "base", "small"],
  "loaded_models": ["tiny"],
  "default_model": "tiny", 
  "memory_usage": {
    "allocated_gb": 0.49,
    "total_gb": 15.48,
    "usage_percent": 3.15
  }
}
```

### Model Information  
```bash
curl http://localhost:8771/v1/models/small/info
```
**Response:**
```json
{
  "name": "small",
  "path": "/workspace/models/models--Systran--faster-whisper-small/snapshots/...",
  "is_loaded": true,
  "is_default": false,
  "status": "loaded",
  "memory_usage_mb": 244.0
}
```

### Load/Unload Models
```bash
# Load model into memory
curl -X POST http://localhost:8771/v1/models/base/load

# Free memory by unloading
curl -X POST http://localhost:8771/v1/models/base/unload
```

## Common Tasks

### Transcribe a Meeting with Speaker Identification
```bash
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@zoom_recording.mp4" \
  -F "model=small" \
  -F "diarize=true" \
  -F "format=json" > meeting_transcript.json
```

### Generate High-Quality Subtitles
```bash
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@video.mp4" \
  -F "model=small" \
  -F "format=srt" > video_subtitles.srt
```

### Batch Processing with Different Models
```bash
# Fast processing for previews
for file in *.mp3; do
  echo "Quick transcription: $file..."
  curl -X POST http://localhost:8771/v1/transcribe \
    -F "file=@$file" \
    -F "model=tiny" \
    -F "format=text" > "${file%.mp3}_draft.txt"
done

# High-quality processing  
for file in *.mp3; do
  echo "Final transcription: $file..."
  curl -X POST http://localhost:8771/v1/transcribe \
    -F "file=@$file" \
    -F "model=small" \
    -F "format=text" > "${file%.mp3}_final.txt"
done
```

### Monitor Model Performance
```bash
# Check current memory usage
curl http://localhost:8771/v1/models | jq '.memory_usage'

# Pre-load models for faster processing
curl -X POST http://localhost:8771/v1/models/small/load

# Check loaded models
curl http://localhost:8771/v1/models | jq '.loaded_models'
```

## Supported File Types
- **Audio**: MP3, WAV, M4A, FLAC, OGG, OPUS, WMA, AAC
- **Video**: MP4, AVI, MOV, MKV, WEBM (audio extracted automatically)
- **Other**: AMR, 3GP, any format FFmpeg supports

## Health Check
```bash
# Check if service is running
curl http://localhost:8771/health
```

**Response shows GPU status and model information:**
```json
{
  "status": "healthy",
  "ok": true,
  "gpu_required": true,
  "gpu_available": true,
  "default_model": "tiny",
  "device": "cuda",
  "compute_type": "float16",
  "gpu": {
    "device_name": "NVIDIA GeForce RTX 5060 Ti",
    "memory_allocated": "0.49GB",
    "memory_total": "15.48GB",
    "blackwell_detected": true
  },
  "diarization": {
    "modules_available": true,
    "pipeline_loaded": true,
    "backend": "NeMo"
  },
  "processing_mode": {
    "whisper": "GPU",
    "diarization": "GPU (NeMo)"
  }
}
```

## Docker Management

### Run in Background
```bash
docker compose -f docker-compose.blackwell.yml up -d
```

### View Logs
```bash
docker compose -f docker-compose.blackwell.yml logs -f
```

### Stop Service
```bash
docker compose -f docker-compose.blackwell.yml down
```

### Update to Latest
```bash
docker pull arealicehole/whisper-blackwell:d5-l1
docker compose -f docker-compose.blackwell.yml up -d
```

## Troubleshooting

### Container Hangs on Startup with VPN
If the container hangs during startup when using a VPN (WireGuard, OpenVPN, etc.), it's likely due to MTU mismatch. VPNs typically use MTU 1420 while Docker defaults to 1500, causing SSL handshake timeouts and preventing NeMo models from loading.

**Fix:**
1. Edit Docker daemon configuration:
   ```bash
   sudo nano /etc/docker/daemon.json
   ```
2. Add MTU setting (adjust value to match your VPN):
   ```json
   {
     "bridge": "docker0",
     "mtu": 1420,
     "dns": ["8.8.8.8", "8.8.4.4"]
   }
   ```
3. Restart Docker and recreate container:
   ```bash
   sudo systemctl restart docker
   docker compose -f docker-compose.blackwell.yml down
   docker compose -f docker-compose.blackwell.yml up -d
   ```

**Symptoms of MTU issue:**
- Container starts but health checks fail
- Logs show "Matplotlib is building font cache" hanging
- NeMo import never completes
- SSL/HTTPS connections timeout inside container

### GPU Not Found
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If using Fedora, ensure nvidia runtime is configured
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Diarization Not Working
- Make sure HF_TOKEN is set: `echo $HF_TOKEN`
- Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
- Check health endpoint shows diarization available

### Out of Memory
- Use smaller model: add `-F "model=tiny"`
- Process shorter segments of long files
- Monitor GPU: `watch nvidia-smi`

### Port Already Used
```bash
# Change port in docker-compose.blackwell.yml
# From: "8771:8767"
# To:   "8080:8767"
```

## Performance Tips

### Model Selection Strategy
- **`tiny`** (39MB): Use for real-time, drafts, or when speed is critical
- **`base`** (74MB): Good balance of speed vs quality for general use
- **`small`** (244MB): Best quality available, recommended for production

### Memory Management
- **Pre-load models**: Use `/v1/models/{name}/load` for frequently used models
- **Monitor usage**: Check `/v1/models` for memory consumption
- **Auto-eviction**: System automatically unloads least-used models when memory is full
- **Disable diarization**: Skip `diarize=true` for 2x faster processing

### Batch Processing
- Use `tiny` model for quick previews, `small` for final transcripts
- Pre-load your target model before batch processing
- Monitor GPU memory with `nvidia-smi` during long jobs

## Docker Hub
```bash
docker pull arealicehole/whisper-blackwell:d5-l1
```

### Version History
- **`d5-l1`** - 🆕 **Local Model Storage + Dynamic Selection**
  - ✅ No internet downloads during transcription
  - ✅ Dynamic model switching via API
  - ✅ Model management endpoints (`/v1/models`)
  - ✅ Memory-efficient LRU model caching
  - ✅ Includes `tiny`, `base`, `small` models
- `d4-n2` - Fixed: NeMo toolkit properly installed for diarization
- `d4-n1` - Initial release (NeMo not installed)

## Environment Variables

### Model Configuration
- `WHISPER_DEFAULT_MODEL` - Default model to load (default: `tiny`)
- `MODELS_DIRECTORY` - Path to local models (default: `/workspace/models`)
- `MAX_LOADED_MODELS` - Maximum models in memory (default: `2`)

### Diarization
- `HF_TOKEN` - HuggingFace token for NeMo diarization models
- `WHISPER_DIARIZE` - Enable diarization by default (default: `true`)

### GPU Settings
- `WHISPER_DEVICE` - Force device (default: `cuda`)
- `WHISPER_COMPUTE` - Compute precision (default: `float16`)

**Example:**
```bash
export WHISPER_DEFAULT_MODEL=small
export MAX_LOADED_MODELS=3
export HF_TOKEN=hf_xxxxx
docker compose -f docker-compose.blackwell.yml up
```

## What's Included

### Core Components
- **faster-whisper 1.0.3** - GPU-accelerated OpenAI Whisper with CTranslate2
- **NeMo toolkit** - NVIDIA's diarization framework with Sortformer v1
- **WhisperModelManager** - Dynamic model loading with LRU memory management
- **PyTorch 2.7.0** - With CUDA 12.4 and Blackwell GPU support
- **FastAPI server** - Production-ready REST API with monitoring

### Pre-installed Models
- **tiny** (39MB) - Fast transcription for drafts and real-time
- **base** (74MB) - Balanced speed and quality
- **small** (244MB) - Best quality for production use

### Key Features
- 🚀 **100% Local Processing** - No internet required for Whisper transcription
- 🔄 **Runtime Model Switching** - Change models without container restart
- 🎯 **GPU-Only Enforcement** - No CPU fallback for maximum performance  
- 📊 **Memory Management** - Automatic LRU eviction and monitoring
- 🎤 **Speaker Diarization** - Up to 4 speakers with NeMo Sortformer
- 🔧 **Production Ready** - Health checks, error handling, and monitoring

**System Requirements:** NVIDIA GPU with 2GB+ VRAM, Docker with GPU support