# Whisper API - Speech-to-Text with Optional Speaker Diarization

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance REST API for audio transcription using OpenAI's Whisper model with optional speaker diarization (speaker identification).

## ✨ Why This Project?

- **Unified API**: Single service for both transcription and speaker identification
- **Flexible**: Choose features per request - fast transcription or detailed speaker analysis
- **Production Ready**: Includes systemd service, Docker support, and comprehensive error handling
- **Well Documented**: Clear examples for every use case

## Features

- 🚀 **GPU-accelerated transcription** using faster-whisper
- 🎯 **Optional speaker diarization** - identify who's speaking when
- 🔄 **Sync and async API endpoints** 
- 📝 **Multiple output formats** (JSON, text, SRT, VTT)
- 🎛️ **Per-request control** - choose features for each transcription

## Quick Start

### 1. Initial Setup (One Time)

```bash
# Clone or navigate to the project
cd /home/ice/whisper-api

# Set up Python 3.11 environment (required for diarization)
./setup_isolated_python.sh

# Activate environment and install dependencies
source ~/.venvs/whisper-diarize/bin/activate
pip install -r requirements_diarization.txt

# Add your HuggingFace token (required for diarization)
mkdir -p ~/.config/whisper
echo "HF_TOKEN=hf_your_token_here" > ~/.config/whisper/token
```

**Note**: Get your HuggingFace token from https://huggingface.co/settings/tokens and accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1

### 2. Start the Service

```bash
# Simple start
./start_whisper.sh start

# Check status
./start_whisper.sh status

# View logs
./start_whisper.sh logs
```

The service runs on `http://localhost:8765`

### 3. Use the API

#### Basic Transcription (Fast, No Speaker Detection)

```bash
# Using curl
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav"

# Using the CLI wrapper
./whisper-cli.sh audio.wav

# From Python
from whisper_client import WhisperClient
client = WhisperClient()
result = client.transcribe("audio.wav")
print(result['text'])
```

#### With Speaker Diarization (Identifies Who's Speaking)

```bash
# Using curl - just add diarize=true
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "num_speakers=2"

# Using the CLI wrapper
./whisper-cli.sh audio.wav --diarize --speakers 2

# From Python
result = client.transcribe("audio.wav", diarize=True, num_speakers=2)
for segment in result['segments']:
    print(f"{segment['speaker']}: {segment['text']}")
```

## API Endpoints

### `GET /` - Service Info
Shows service status and available features

### `GET /health` - Health Check
Detailed status including diarization availability

### `POST /v1/transcribe` - Synchronous Transcription

**Parameters:**
- `file` - Audio file (WAV, MP3, etc.)
- `diarize` - Enable speaker detection (optional, default: false)
- `num_speakers` - Number of speakers if known (optional)
- `language` - Language code (optional, default: "en")
- `format` - Output format: json|text|srt|vtt (default: "json")

**Example Response with Diarization:**
```json
{
  "text": "Full transcript here...",
  "language": "en",
  "duration": 120.5,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello everyone",
      "speaker": "SPEAKER_00"
    },
    {
      "id": 2,
      "start": 3.5,
      "end": 5.8,
      "text": "Hi, thanks for having me",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

### `POST /v2/transcript` - Asynchronous Transcription

For long audio files, returns a job ID to check status.

## Client Libraries

### Python Client

```python
from whisper_client import WhisperClient

# Initialize client
client = WhisperClient("http://localhost:8765")

# Basic transcription
result = client.transcribe("meeting.wav")

# With all options
result = client.transcribe(
    "meeting.wav",
    diarize=True,          # Enable speaker detection
    num_speakers=3,        # Expected number of speakers
    language="en",         # Language code
    format="json"          # Output format
)

# Format for display
print(client.format_transcript(result, style="dialogue"))
```

### CLI Wrapper

```bash
# Show usage
./whisper-cli.sh --help

# Basic transcription
./whisper-cli.sh audio.wav

# With diarization
./whisper-cli.sh audio.wav --diarize --speakers 2 --format srt
```

## Installation Options

### Option 1: Manual Start
```bash
./start_whisper.sh start  # Start
./start_whisper.sh stop   # Stop
./start_whisper.sh status # Check status
```

### Option 2: System Service (Auto-start on boot)
```bash
sudo ./install_service.sh
sudo systemctl start whisper-api
sudo systemctl status whisper-api
```

### Option 3: Docker
```bash
docker-compose up -d
```

## Configuration

Environment variables (optional):
- `WHISPER_MODEL` - Model size: tiny|base|small|medium|large (default: small)
- `WHISPER_DEVICE` - Device: cuda|cpu (default: cuda)
- `WHISPER_LANGUAGE` - Default language (default: en)
- `WHISPER_DIARIZE` - Make diarization available (default: true)

## Performance Notes

- **Basic transcription**: Fast, real-time or better
- **With diarization**: Adds 20-50% processing time
- **GPU recommended** for best performance
- **Model sizes**: 
  - tiny: Fastest, least accurate
  - small: Good balance (default)
  - large: Most accurate, slowest

## Troubleshooting

### Check Service Health
```bash
curl http://localhost:8765/health | jq .
```

### Test Diarization Setup
```bash
python test_diarization.py
```

### Common Issues

1. **Diarization not working**: 
   - Check HuggingFace token in `~/.config/whisper/token`
   - Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Run `python test_diarization.py` for diagnostics

2. **CUDA not available**:
   - Check GPU drivers: `nvidia-smi`
   - Reinstall PyTorch with CUDA support

3. **Port already in use**:
   - Stop existing service: `./start_whisper.sh stop`
   - Or change port in main.py

## Project Structure

```
whisper-api/
├── main.py                    # FastAPI application
├── whisper_client.py          # Python client library
├── whisper-cli.sh            # CLI wrapper
├── start_whisper.sh          # Service launcher
├── test_diarization.py       # Diarization diagnostic tool
├── test_transcribe.py        # Transcription test script
├── requirements_diarization.txt  # Python dependencies
├── setup_isolated_python.sh  # Environment setup
├── Dockerfile                # Docker deployment
├── docker-compose.yml        # Docker compose config
└── whisper-api.service       # Systemd service file
```

## Requirements

- Python 3.11 (for pyannote compatibility)
- NVIDIA GPU with CUDA (optional but recommended)
- 4-8GB RAM depending on model size
- HuggingFace account (free) for diarization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/arealicehole/whisper-on-fedora.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes and test
5. Submit a pull request

### Running Tests

```bash
# Test transcription
python test_transcribe.py sample.wav

# Test diarization setup
python test_diarization.py

# Test API endpoints
curl http://localhost:8765/health
```

## Roadmap

- [ ] Add real-time streaming transcription
- [ ] Support for more audio formats
- [ ] Web UI for testing
- [ ] Batch processing endpoint
- [ ] WebSocket support for live audio
- [ ] Multi-language diarization
- [ ] Speaker embedding export
- [ ] Fine-tuning support

## Credits

This project uses:
- [Whisper](https://github.com/openai/whisper) by OpenAI - MIT License
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein - MIT License
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) by Hervé Bredin - MIT License

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](QUICKSTART.md)
- 🐛 [Report Issues](https://github.com/arealicehole/whisper-on-fedora/issues)
- 💬 [Discussions](https://github.com/arealicehole/whisper-on-fedora/discussions)
- 📧 Contact: @arealicehole

## Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Note**: This project requires acceptance of the pyannote model license for speaker diarization features. Visit [the model page](https://huggingface.co/pyannote/speaker-diarization-3.1) to accept the license.