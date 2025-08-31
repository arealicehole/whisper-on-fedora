# 🎙️ Whisper API

Production-ready, GPU-accelerated speech-to-text REST API using OpenAI's Whisper with optional speaker diarization.

## 💻 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (REQUIRED - no CPU fallback)
- **Python**: 3.11.x (MUST be 3.11 - not 3.10 or 3.12+)
- **CUDA**: 11.8+ (12.8+ for Blackwell GPUs)
- **RAM**: 8GB system memory
- **Storage**: 10GB free space
- **OS**: Linux (Ubuntu 22.04+, Fedora 40+) or macOS

### Recommended Setup
- **GPU**: NVIDIA RTX 3060 or better (12GB VRAM)
- **Python**: 3.11.9
- **CUDA**: 12.1
- **RAM**: 16GB
- **Storage**: 20GB SSD

> ⚠️ **Python Version Warning**
> 
> This project REQUIRES Python 3.11.x for PyAnnote compatibility.
> - ❌ Python 3.10 and below: Missing required features
> - ❌ Python 3.12 and above: Breaks PyAnnote dependencies
> - ✅ Python 3.11.x: Fully compatible
> 
> Verify with: `python --version`

## ✨ Features

- ⚡ **GPU-Accelerated**: 10x faster transcription with CUDA optimization
- 👥 **Speaker Diarization**: Optional speaker identification per request  
- 🔄 **Flexible Processing**: Sync and async endpoints for any file size
- 📝 **Multiple Formats**: JSON, text, SRT, VTT outputs
- 🚀 **Production Ready**: Robust error handling and health monitoring

## 🎮 Blackwell GPU Support (RTX 5060 Ti)

For NVIDIA Blackwell architecture GPUs, use PyTorch nightly builds:

```bash
# Install PyTorch nightly with Blackwell support
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Use special requirements file
pip install -r requirements_blackwell.txt

# The project includes automatic patches for compatibility in sitecustomize.py
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api

# Quick setup (Python 3.11 required!)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_api.txt

# Run the API
python main.py

# Test it works
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@audio.mp3" | jq .text
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.



## 🛠️ Installation

### Standard GPU Setup
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_api.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyannote.audio==3.3.1
```

### Blackwell GPU Setup (RTX 5060 Ti)
```bash
# Use PyTorch nightly for Blackwell support
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements_blackwell.txt
pip install --no-deps pyannote.audio==3.3.1
```

### Configure Diarization (Optional)
```bash
mkdir -p ~/.config/whisper
echo "HF_TOKEN=your_token" > ~/.config/whisper/token
```
Get token from [HuggingFace](https://huggingface.co/settings/tokens)

## 🐳 Docker Deployment

### For Blackwell GPUs (RTX 5060 Ti)
```bash
cd docker
docker-compose up -d whisper-blackwell
```

### For Standard GPUs
```bash
docker build -t whisper-api .
docker run --gpus all -p 8767:8767 whisper-api
```

### Docker Compose Configuration
The project includes optimized Docker configurations with:
- NVIDIA GPU runtime support
- Pre-configured environment variables
- Automatic model caching
- Health checks

## 📡 API Usage

### Basic Transcription
```bash
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@audio.mp3"
```

### With Speaker Identification
```bash
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@audio.mp3" \
  -F "diarize=true"
```

### Python Client
```python
from whisper_client import WhisperClient
client = WhisperClient()
result = client.transcribe("audio.mp3", diarize=True)
print(result["text"])
```

### Real-World Examples

#### Transcribe a Podcast with Speakers
```bash
curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@podcast.mp3" \
  -F "diarize=true" \
  -F "format=srt" > podcast.srt
```

#### Process Meeting Recording
```python
from whisper_client import WhisperClient
import json

client = WhisperClient()
result = client.transcribe("meeting.m4a", diarize=True)

# Export speaker segments
with open("meeting_transcript.json", "w") as f:
    json.dump(result["segments"], f, indent=2)

# Print speaker summary
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s] {segment['speaker']}: {segment['text']}")
```

#### Batch Processing Multiple Files
```bash
# Process all MP3 files in a directory
for file in *.mp3; do
  echo "Processing $file..."
  curl -X POST http://localhost:8767/v1/transcribe \
    -F "file=@$file" \
    -F "format=text" \
    -o "${file%.mp3}.txt"
done
```

#### Async Processing for Large Files
```python
import time
from whisper_client import WhisperClient

client = WhisperClient()

# Submit large file for processing
job = client.transcribe_async("conference_2hrs.mp4", diarize=True)
print(f"Job ID: {job['job_id']}")

# Poll for completion
while True:
    status = client.get_job_status(job['job_id'])
    if status['status'] == 'completed':
        print("Transcription complete!")
        print(status['result']['text'])
        break
    elif status['status'] == 'failed':
        print(f"Job failed: {status['error']}")
        break
    time.sleep(5)
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check with GPU status |
| `/v1/transcribe` | POST | Synchronous transcription |
| `/v2/transcript` | POST | Async job submission |
| `/v2/transcript/{id}` | GET | Get async job results |

## ⚙️ Configuration

Environment variables:
- `WHISPER_MODEL`: Model size (tiny/base/small/medium/large)
- `WHISPER_LANGUAGE`: Default language (en)
- `WHISPER_DIARIZE`: Enable diarization by default (true/false)

## 📂 Project Structure

### Branch Strategy
- **main branch**: Production runtime only (10 essential files)
- **dev branch**: Full development environment (140+ files)

### Core Files
```
whisper-api/
├── main.py                 # FastAPI application
├── gpu_validator.py        # GPU enforcement & validation
├── sitecustomize.py        # Blackwell GPU patches
├── startup.py              # Service initialization
├── whisper_client.py       # Python client library
├── requirements_api.txt    # Core dependencies
├── requirements_blackwell.txt # Blackwell-specific deps
├── LICENSE                 # MIT license
├── QUICKSTART.md          # 5-minute setup guide
├── ONBOARDING.md          # Comprehensive developer guide
└── README.md              # This file
```

See [docs-archive/BRANCH_STRATEGY.md](docs-archive/BRANCH_STRATEGY.md) for details.

## 🔨 Development Setup

### Branch Structure
- `main`: Production runtime (minimal, 10 files)
- `dev`: Development branch (full codebase, 140+ files)

### Developer Quick Start
```bash
# Clone and switch to dev branch
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api
git checkout dev

# Setup development environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements_api.txt

# Run with auto-reload
uvicorn main:app --reload --port 8767
```

### Testing
```bash
# Test suite
python test_gpu_basic.py        # GPU detection
python test_whisper_only.py     # Transcription
python test_diarization_final.py # Speaker identification
python test_mp3_diarization.py  # MP3 processing
```

See [ONBOARDING.md](ONBOARDING.md) for complete developer guide.

## 🐛 Troubleshooting

### Python Version Issues
```bash
# Check Python version (MUST be 3.11.x)
python --version

# Install Python 3.11 if needed
# Ubuntu/Debian:
sudo apt install python3.11 python3.11-venv

# Fedora:
sudo dnf install python3.11

# macOS:
brew install python@3.11
```

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Test GPU in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# For GPU enforcement errors
python gpu_validator.py
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8767

# Kill process
kill -9 [PID]

# Or use different port
PORT=8768 python main.py
```

### Diarization Not Working
- Ensure Python 3.11 (not 3.12+)
- Check HuggingFace token: `cat ~/.config/whisper/token`
- Accept model license at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Test with: `python test_diarization_final.py`

### Memory Issues
- Reduce model size: Use `tiny` or `base` instead of `large`
- Process shorter segments: Split long audio files
- Monitor GPU memory: `watch -n 1 nvidia-smi`
- Clear GPU cache in Python: `torch.cuda.empty_cache()`

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements_api.txt

# For Blackwell GPUs, reinstall PyTorch nightly
pip install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

## ⚡ Performance Tuning

### Model Selection Guide
- **Podcasts/Meetings**: Use `base` or `small` for good balance
- **Quick drafts**: Use `tiny` for fastest processing
- **Legal/Medical**: Use `medium` or `large` for accuracy
- **Multiple languages**: Use `large` model
- **Real-time needs**: Use `tiny` with streaming

### Speed Optimizations
- **Disable diarization** when not needed (2-3x faster)
- **Use smaller models** for initial drafts
- **Process in chunks** for very long files (>2 hours)
- **Pre-convert to 16kHz WAV** for marginal speed gain
- **Increase batch size** for multiple files

### GPU Memory Management
```python
# Monitor GPU usage
nvidia-smi -l 1

# Clear cache between large files
import torch
torch.cuda.empty_cache()

# Use smaller model if OOM
export WHISPER_MODEL=tiny  # Instead of large
```

## 📊 Model Performance

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Good | 1GB |
| base | 74M | Fast | Better | 2GB |
| small | 244M | Balanced | Great | 2GB |
| medium | 769M | Slower | Excellent | 4GB |
| large | 1550M | Slowest | Best | 8GB |

## 🚀 Production Deployment

### Recommended Setup
1. Use `main` branch for minimal runtime
2. Run behind reverse proxy (nginx/caddy)
3. Use process manager (systemd/pm2)
4. Monitor with health endpoint
5. Set up log rotation

### Systemd Service Example
```ini
[Unit]
Description=Whisper API Service
After=network.target

[Service]
Type=simple
User=whisper
WorkingDirectory=/opt/whisper-api
Environment="PATH=/opt/whisper-api/venv/bin"
Environment="WHISPER_MODEL=base"
ExecStart=/opt/whisper-api/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Install as Service
```bash
sudo cp whisper-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable whisper-api
sudo systemctl start whisper-api
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://localhost:8767;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        client_max_body_size 500M;  # For large audio files
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow existing code patterns
4. Test with multiple audio formats
5. Submit pull request

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🔗 Links

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 🙏 Acknowledgments

Built on OpenAI's Whisper, Pyannote, and the Faster-Whisper implementation.

---

**Status**: ✅ Production Ready | **API Version**: v1/v2 | **Port**: 8767