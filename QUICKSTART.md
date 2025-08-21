# Whisper API Quick Start Guide

## Setup (One Time)

### Option 1: Quick Start
```bash
# Install Python 3.11 environment
./setup_isolated_python.sh

# Activate and install dependencies
source ~/.venvs/whisper-diarize/bin/activate
pip install -r requirements_diarization.txt
```

### Option 2: System Service (Auto-start on boot)
```bash
sudo ./install_service.sh
sudo systemctl start whisper-api
```

## Starting the Service

### Manual Start
```bash
./start_whisper.sh start
```

### Check Status
```bash
./start_whisper.sh status
```

## Using the API

### Basic Transcription (Fast, No Speakers)
```bash
# Default - just transcription
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav"
```

### With Speaker Diarization (Slower, Identifies Speakers)
```bash
# Add diarize=true to enable speaker detection
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "num_speakers=2"
```

### Using the CLI
```bash
# Basic
./whisper-cli.sh audio.wav

# With diarization
./whisper-cli.sh audio.wav --diarize --speakers 3
```

### From Python
```python
from whisper_client import WhisperClient

client = WhisperClient()

# Basic transcription
result = client.transcribe("audio.wav")

# With diarization
result = client.transcribe("audio.wav", diarize=True, num_speakers=2)

# Pretty print
print(client.format_transcript(result, style="dialogue"))
```

## Key Points

1. **Diarization is OPTIONAL** - Each API call can choose whether to use it
   - `diarize=false` (default): Fast transcription only
   - `diarize=true`: Slower but identifies speakers

2. **Service runs once** - Start it and leave it running
   - Manual: `./start_whisper.sh start`
   - Auto: Install as systemd service

3. **Any program can use it** - It's just HTTP
   - Your Python scripts (any version)
   - Shell scripts (curl)
   - Web apps (JavaScript)
   - Any language that can make HTTP requests

## Troubleshooting

### Check if service is running
```bash
./start_whisper.sh status
curl http://localhost:8765/health | jq .
```

### Check diarization status
```bash
curl http://localhost:8765/health | jq .diarization
```

### View logs
```bash
./start_whisper.sh logs
# or
tail -f ~/.whisper-api.log
```

### Test diarization setup
```bash
python test_diarization.py
```