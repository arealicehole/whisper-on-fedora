# Whisper API with GPU Diarization (Blackwell)

Complete GPU-accelerated transcription + speaker diarization API. Everything runs in Docker - no local Python setup needed.

## Requirements
- NVIDIA GPU (optimized for Blackwell/RTX 5060 Ti)
- Docker with NVIDIA Container Toolkit
- HuggingFace token (for diarization models)

## Quick Start

### 1. Get HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1

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

### 3. Test It Works
```bash
# Quick test (returns JSON)
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=json"

# Just get the text
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=text"
```

## API Usage

### Basic Transcription
```bash
# Transcribe any audio file
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@podcast.mp3"
```

### With Speaker Labels
```bash
# Add speaker identification
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@meeting.mp3" \
  -F "diarize=true"
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
| `format` | `json`, `text`, `srt`, `vtt` | Output format | `json` |
| `diarize` | `true`, `false` | Enable speaker labels | `false` |
| `language` | `en`, `es`, `fr`, etc | Audio language | `en` |
| `model` | `tiny`, `base`, `small`, `medium`, `large` | Model size (bigger = more accurate) | `small` |

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

# Simple transcription
result = client.transcribe("audio.mp3")
print(result["text"])

# With speakers
result = client.transcribe("meeting.mp3", diarize=True)
for segment in result["segments"]:
    print(f"{segment['speaker']}: {segment['text']}")
```

## Common Tasks

### Transcribe a Meeting
```bash
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@zoom_recording.mp4" \
  -F "diarize=true" \
  -F "format=json" > meeting_transcript.json
```

### Generate Subtitles
```bash
curl -X POST http://localhost:8771/v1/transcribe \
  -F "file=@video.mp4" \
  -F "format=srt" > video_subtitles.srt
```

### Process Multiple Files
```bash
for file in *.mp3; do
  echo "Processing $file..."
  curl -X POST http://localhost:8771/v1/transcribe \
    -F "file=@$file" \
    -F "format=text" > "${file%.mp3}.txt"
done
```

## Supported File Types
- **Audio**: MP3, WAV, M4A, FLAC, OGG, OPUS, WMA, AAC
- **Video**: MP4, AVI, MOV, MKV, WEBM (audio extracted automatically)
- **Other**: AMR, 3GP, any format FFmpeg supports

## Health Check
```bash
# Check if service is running
curl http://localhost:8771/health

# Response shows GPU status
{
  "status": "healthy",
  "gpu_available": true,
  "diarization": {
    "modules_available": true,
    "backend": "NeMo"
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
docker pull arealicehole/whisper-blackwell:d4-n2
docker compose -f docker-compose.blackwell.yml up -d
```

## Troubleshooting

### GPU Not Found
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
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
- `tiny` model: Fastest, good for drafts
- `base` model: Good balance
- `small` model: Default, accurate
- `large` model: Most accurate, slowest
- Disable diarization when not needed (2x faster)

## Docker Hub
```bash
docker pull arealicehole/whisper-blackwell:d4-n2
```

### Version History
- `d4-n2` - Fixed: NeMo toolkit properly installed for diarization
- `d4-n1` - Initial release (NeMo not installed)

## What's Included
- **faster-whisper** - GPU-accelerated OpenAI Whisper
- **NeMo toolkit** - NVIDIA's diarization framework  
- **Sortformer v1** - Speaker identification (4 speakers max)
- **PyTorch 2.5.1** - With CUDA 12.4 support
- **FastAPI server** - Production-ready API
- All dependencies pre-installed and configured

GPU only, no CPU fallback. Models download automatically on first run.