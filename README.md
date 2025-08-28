# 🎙️ Whisper API

Production-ready, GPU-accelerated speech-to-text REST API using OpenAI's Whisper with optional speaker diarization.

## ✨ Features

- ⚡ **GPU-Accelerated**: 10x faster transcription with CUDA optimization
- 👥 **Speaker Diarization**: Optional speaker identification per request  
- 🔄 **Flexible Processing**: Sync and async endpoints for any file size
- 📝 **Multiple Formats**: JSON, text, SRT, VTT outputs
- 🚀 **Production Ready**: Robust error handling and health monitoring

## 🚀 Quick Start

```bash
# Setup (one time)
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api
./setup_venv.sh
pip install -r requirements.txt

# Run
./start_whisper.sh start

# Use
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3" | jq .text
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.

## 📋 Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (required)
- **Python**: 3.11 (for diarization compatibility)
- **CUDA**: Drivers installed and functional
- **OS**: Linux/macOS

## 🛠️ Installation

### Standard Setup
```bash
./setup_venv.sh
pip install -r requirements.txt
```

### Configure Diarization (Optional)
```bash
mkdir -p ~/.config/whisper
echo "HF_TOKEN=your_token" > ~/.config/whisper/token
```
Get token from [HuggingFace](https://huggingface.co/settings/tokens)

## 📡 API Usage

### Basic Transcription
```bash
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.mp3"
```

### With Speaker Identification
```bash
curl -X POST http://localhost:8765/v1/transcribe \
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

```
whisper-api/
├── main.py              # FastAPI application
├── gpu_validator.py     # GPU enforcement
├── whisper_client.py    # Python client library
├── start_whisper.sh     # Service management
├── requirements.txt     # Dependencies
├── QUICKSTART.md       # Quick setup guide
├── ONBOARDING.md       # Developer guide
└── examples/           # Usage examples
```

## 🔧 Development

See [ONBOARDING.md](ONBOARDING.md) for comprehensive developer documentation.

## 🐛 Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check CUDA installation
```

### Port Already in Use
```bash
./start_whisper.sh stop
# Or: lsof -i :8765 && kill [PID]
```

### Diarization Not Working
- Ensure Python 3.11
- Valid HuggingFace token configured
- Accept license at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)

## 📊 Model Performance

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Good | 1GB |
| base | 74M | Fast | Better | 2GB |
| small | 244M | Balanced | Great | 2GB |
| medium | 769M | Slower | Excellent | 4GB |
| large | 1550M | Slowest | Best | 8GB |

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

**Status**: ✅ Production Ready | **API Version**: v1/v2 | **Port**: 8765