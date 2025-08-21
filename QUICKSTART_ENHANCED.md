# Whisper API - Quick Start Guide 🚀

Get the Whisper API running in 5 minutes!

## Prerequisites Check
```bash
# Check Python version (need 3.11)
python3.11 --version

# Check GPU (optional but recommended)
nvidia-smi

# Check available disk space (need ~5GB)
df -h .
```

## 1. Fast Setup (3 commands)

```bash
# 1. Setup Python environment
./setup_isolated_python.sh

# 2. Activate and install
source ~/.venvs/whisper-diarize/bin/activate
pip install -r requirements.txt

# 3. Start the service
./start_whisper.sh start
```

## 2. Configure Diarization (Optional)

For speaker identification features:

```bash
# Get HuggingFace token from: https://huggingface.co/settings/tokens
mkdir -p ~/.config/whisper
echo "HF_TOKEN=hf_your_token_here" > ~/.config/whisper/token

# Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
```

## 3. Verify Installation

```bash
# Check service health
curl http://localhost:8765/health | python -m json.tool

# Should see:
# {
#   "ok": true,
#   "model": "tiny",
#   "device": "cuda",
#   ...
# }
```

## 4. Your First Transcription

### Quick Test
```bash
# Create test audio (or use your own)
echo "import numpy as np; import wave; t = np.linspace(0, 3, 48000); \
audio = np.sin(2*np.pi*440*t); \
with wave.open('test.wav', 'wb') as w: \
  w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); \
  w.writeframes((audio*32767).astype(np.int16).tobytes())" | python

# Transcribe it
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@test.wav" \
  -o result.json

# View result
cat result.json | python -m json.tool
```

### Real Audio File
```bash
# Basic transcription (fast)
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@your_audio.wav"

# With speaker detection (slower)
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@your_audio.wav" \
  -F "diarize=true"
```

## 5. Python Client Usage

```python
from whisper_client import WhisperClient

# Initialize
client = WhisperClient("http://localhost:8765")

# Transcribe
result = client.transcribe("meeting.wav", diarize=True)

# Display
for segment in result['segments']:
    speaker = segment.get('speaker', 'Unknown')
    print(f"{speaker}: {segment['text']}")
```

## Service Management

| Action | Command |
|--------|---------|
| Start | `./start_whisper.sh start` |
| Stop | `./start_whisper.sh stop` |
| Restart | `./start_whisper.sh restart` |
| Status | `./start_whisper.sh status` |
| Logs | `./start_whisper.sh logs` |

## Output Formats

| Format | Usage | Output |
|--------|-------|--------|
| JSON | `format=json` (default) | Full structured data |
| Text | `format=text` | Plain text only |
| SRT | `format=srt` | Subtitle format |
| VTT | `format=vtt` | Web subtitle format |

## Common Issues & Fixes

### Port Already in Use
```bash
lsof -i:8765
kill <PID>
./start_whisper.sh start
```

### CUDA Not Working (RTX 5060 Ti)
```bash
# Run automatic fix
./fix_cuda_fallback.sh
```

### Diarization Not Loading
```bash
# Check setup
python test_diarization.py

# Verify token
cat ~/.config/whisper/token
```

### Out of Memory
```bash
# Use smaller model
export WHISPER_MODEL=tiny
./start_whisper.sh restart
```

## Performance Tips

1. **Model Selection**:
   - `tiny`: Fastest, least accurate (39MB)
   - `base`: Good balance (74MB)
   - `small`: Better accuracy (244MB) - default
   - `medium`: High accuracy (769MB)
   - `large`: Best accuracy (1550MB)

2. **Speed Optimization**:
   - Skip diarization for speed: `diarize=false`
   - Use GPU: Automatic if available
   - Batch multiple files: Use async endpoint

3. **Quality Optimization**:
   - Use `diarize=true` for meetings
   - Specify `num_speakers` if known
   - Use larger models for accuracy

## API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/v1/transcribe` | POST | Sync transcription |
| `/v2/transcript` | POST | Async job creation |
| `/v2/transcript/{id}` | GET | Get job result |

## Next Steps

1. ✅ Service is running
2. ✅ Made first transcription
3. 📖 Read [ONBOARDING.md](ONBOARDING.md) for deep dive
4. 🔧 Check [examples/](examples/) for integration samples
5. 🚀 Build something awesome!

---

**Need Help?**
- Logs: `tail -f ~/.whisper-api.log`
- Test: `python test_transcribe.py`
- Docs: See README.md for full details