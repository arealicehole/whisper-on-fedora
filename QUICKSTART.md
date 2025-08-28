# 🚀 Whisper API - Quick Start Guide

Get the Whisper API running in 5 minutes!

## Prerequisites
- Linux/macOS with NVIDIA GPU (4GB+ VRAM)
- Python 3.11
- CUDA drivers installed

## Setup (One Time)

### 1. Clone & Enter Directory
```bash
git clone [repository-url]
cd whisper-api
```

### 2. Create Virtual Environment
```bash
./setup_venv.sh
# Or manually: python3.11 -m venv venv && source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure HuggingFace Token (Optional - for speaker diarization)
```bash
mkdir -p ~/.config/whisper
echo "HF_TOKEN=your_token_here" > ~/.config/whisper/token
```
Get token from: https://huggingface.co/settings/tokens

## Run the Service

### Start
```bash
./start_whisper.sh start
```

### Check Health
```bash
curl http://localhost:8765/health | jq .status
# Expected: "healthy"
```

### Stop
```bash
./start_whisper.sh stop
```

## Use the API

### Basic Transcription
```bash
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3" | jq .text
```

### With Speaker Identification
```bash
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "diarize=true" | jq .
```

### From Python
```python
from whisper_client import WhisperClient

client = WhisperClient()
result = client.transcribe("audio.mp3")
print(result["text"])
```

### Async Transcription (for long files)
```bash
# Submit job
JOB_ID=$(curl -X POST http://localhost:8765/v2/transcript \
  -F "file=@long_audio.mp3" | jq -r .id)

# Check status
curl http://localhost:8765/v2/transcript/$JOB_ID | jq .status

# Get results when ready
curl http://localhost:8765/v2/transcript/$JOB_ID | jq .result
```

## Output Formats

Add `format` parameter to control output:
- `json` (default) - Full details with segments
- `text` - Plain text only
- `srt` - Subtitle format with timecodes
- `vtt` - Web video text tracks

```bash
# Get plain text
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=text"

# Get subtitles
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "format=srt" > subtitles.srt
```

## Troubleshooting

### Port Already in Use
```bash
# Find and kill existing process
lsof -i :8765
kill [PID]
```

### GPU Not Found
```bash
# Check NVIDIA driver
nvidia-smi

# Should show your GPU and CUDA version
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate
which python  # Should point to venv/bin/python
```

### File Upload Issues
```bash
# Always use @ before filename
# ✅ Correct: -F "file=@audio.mp3"  
# ❌ Wrong:   -F "file=audio.mp3"
```

## Model Sizes

Set `WHISPER_MODEL` environment variable before starting:
- `tiny` - Fastest, least accurate (39MB)
- `base` - Balanced (74MB) 
- `small` - Good accuracy (244MB) - Default
- `medium` - Better accuracy (769MB)
- `large` - Best accuracy (1550MB)

```bash
export WHISPER_MODEL=base
./start_whisper.sh start
```

## Monitoring

```bash
# GPU usage
nvidia-smi -l 1

# Service logs  
tail -f ~/.whisper-api.log

# API endpoints
http://localhost:8765/docs    # Interactive API docs
http://localhost:8765/health  # Health check
```

## Next Steps

- Read [ONBOARDING.md](ONBOARDING.md) for detailed documentation
- Check [examples/](examples/) for more usage patterns
- Review [PRPs/](PRPs/) for feature specifications

---

**Need help?** Check the full documentation in `docs-archive/` or open an issue!