# ⚡ Whisper API - Quick Start Guide

Get the Whisper API running in under 5 minutes!

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with 4GB+ VRAM (required)
- **Python**: 3.11.x (MUST be 3.11 for compatibility)
- **CUDA**: Installed NVIDIA drivers
- **OS**: Linux or macOS

## 🚀 Fast Setup

### 1. Clone & Setup Environment (1 minute)

```bash
# Clone repository
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api

# Use dev branch for full features
git checkout dev

# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies (2 minutes)

#### For Standard GPUs (RTX 30xx, 40xx, etc.):
```bash
pip install -r requirements_api.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyannote.audio==3.3.1
```

#### For Blackwell GPUs (RTX 5060 Ti):
```bash
# Install PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements_blackwell.txt
pip install --no-deps pyannote.audio==3.3.1
```

### 3. Configure (Optional - for diarization) (1 minute)

```bash
# Get HuggingFace token from: https://huggingface.co/settings/tokens
mkdir -p ~/.config/whisper
echo "YOUR_HF_TOKEN" > ~/.config/whisper/token

# Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1
```

### 4. Run the API (30 seconds)

```bash
python main.py
```

The API will start on http://localhost:8767

## ✅ Verify Installation

### Test API Health
```bash
curl http://localhost:8767/health | jq
```

Expected output:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model": "tiny",
  "device": "cuda"
}
```

### Test Transcription
```bash
# Download sample audio
wget https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3 -O test.mp3

# Transcribe
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@test.mp3" | jq
```

### Test with Speaker Identification
```bash
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@test.mp3" \
  -F "diarize=true" | jq
```

## 🔧 Quick Configuration

### Environment Variables (.env)
```bash
# Copy template
cp .env.development .env

# Key settings:
WHISPER_MODEL=base       # tiny/base/small/medium/large
WHISPER_DEVICE=cuda      # Always use cuda
WHISPER_DIARIZE=true     # Enable speaker identification
PORT=8767                # API port
```

### Model Size Selection
| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| tiny | Fastest | Good | 1GB |
| base | Fast | Better | 2GB |
| small | Balanced | Great | 2GB |
| medium | Slower | Excellent | 4GB |
| large | Slowest | Best | 8GB |

## 📝 Basic Usage Examples

### Python Client
```python
from whisper_client import WhisperClient

# Initialize client
client = WhisperClient("http://localhost:8767")

# Simple transcription
result = client.transcribe("audio.mp3")
print(result["text"])

# With speaker identification
result = client.transcribe("meeting.wav", diarize=True)
for segment in result["segments"]:
    print(f"{segment['speaker']}: {segment['text']}")
```

### cURL Commands
```bash
# Basic transcription
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@audio.mp3"

# With options
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "format=srt" \
  -F "language=en"

# Async processing for large files
curl -X POST http://localhost:8767/v2/transcript \
  -F "file=@large_audio.mp3"
# Returns: {"job_id": "uuid"}

# Check status
curl http://localhost:8767/v2/transcript/{job_id}
```

## 🐛 Troubleshooting

### GPU Not Found
```bash
# Check CUDA installation
nvidia-smi

# Verify in Python
python -c "import torch; print(torch.cuda.is_available())"
```

### Port Already in Use
```bash
# Find and kill process
lsof -i :8767
kill -9 [PID]

# Or use different port
PORT=8768 python main.py
```

### Diarization Not Working
1. Ensure Python 3.11 (not 3.12+)
2. Check HuggingFace token is set
3. Accept model license online
4. Verify with: `python test_diarization_final.py`

### Out of Memory
- Use smaller model (tiny or base)
- Process shorter audio segments
- Upgrade GPU with more VRAM

## 🔍 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service info |
| `GET /health` | Health check |
| `GET /docs` | Interactive API docs |
| `POST /v1/transcribe` | Sync transcription |
| `POST /v2/transcript` | Async job creation |
| `GET /v2/transcript/{id}` | Get job status |

## 📚 Next Steps

- Read [ONBOARDING.md](ONBOARDING.md) for detailed developer guide
- Explore test scripts: `test_*.py`
- Check API docs: http://localhost:8767/docs
- Review [README.md](README.md) for full features

## 🚦 Service Management

### Start Service
```bash
python main.py
# Or with auto-reload for development
uvicorn main:app --reload --port 8767
```

### Stop Service
```bash
# Ctrl+C in terminal
# Or find and kill process
pkill -f "python main.py"
```

### Run as Background Service
```bash
# Using nohup
nohup python main.py > whisper.log 2>&1 &

# Using screen
screen -S whisper
python main.py
# Detach with Ctrl+A, D
```

## ✨ Quick Tips

1. **Start with `tiny` model** for testing (fastest)
2. **Use `base` model** for production (best balance)
3. **Enable diarization** only when needed (adds overhead)
4. **Process MP3 directly** (no conversion needed)
5. **Check GPU memory** with `nvidia-smi` during processing

---

**Ready to transcribe!** 🎉 The API is now running at http://localhost:8767

For issues, check [ONBOARDING.md](ONBOARDING.md) Section 8: Potential Gotchas