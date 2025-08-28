# 🚀 Whisper API - Developer Onboarding Guide

Welcome to the Whisper API project! This guide will help you understand, set up, and start contributing to this GPU-accelerated speech-to-text service.

## 1. Project Overview

### What is Whisper API?
A production-ready REST API service that provides high-performance audio transcription using OpenAI's Whisper model with optional speaker diarization (speaker identification).

### Core Functionality
- **Audio Transcription**: Convert speech to text using GPU-accelerated Whisper models
- **Speaker Diarization**: Identify who's speaking when (optional per request)
- **Multiple Output Formats**: JSON, plain text, SRT subtitles, VTT captions
- **Async Processing**: Support for both synchronous and asynchronous transcription

### Tech Stack
- **Language**: Python 3.11+ (required for pyannote compatibility)
- **Framework**: FastAPI 0.115.0 with Uvicorn ASGI server
- **AI/ML**: 
  - `faster-whisper` 1.0.3 (GPU-accelerated Whisper implementation)
  - `pyannote.audio` 3.3.1 (speaker diarization)
  - PyTorch 2.3.0 with CUDA support
- **Audio Processing**: librosa, soundfile
- **GPU**: NVIDIA CUDA (required, no CPU fallback)

### Architecture Pattern
- **Service-Oriented**: Single FastAPI service with REST endpoints
- **GPU-First**: Enforces GPU-only operation for consistent performance
- **Stateless Design**: Each request is independent (async jobs stored in memory)

## 2. Repository Structure

```
whisper-api/
│
├── Core Files (Root)
│   ├── main.py                 # FastAPI application & endpoints
│   ├── gpu_validator.py        # GPU enforcement & validation
│   ├── whisper_client.py       # Python client library
│   ├── whisper-cli.sh          # CLI wrapper script
│   │
│   ├── start_whisper.sh        # Service management script
│   ├── setup_venv.sh           # Virtual environment setup
│   │
│   ├── requirements.txt        # Main dependencies
│   └── requirements_diarization.txt  # Alternative deps for Python 3.11
│
├── examples/                   # Client usage examples
│   └── basic_usage.py         # Python client demonstrations
│
├── PRPs/                      # Product Requirement Prompts
│   ├── ai_docs/              # AI/LLM documentation
│   ├── templates/            # PRP templates
│   └── *.md                  # Feature specifications
│
├── docs-archive/              # Organized documentation
│   ├── core-docs/            # Main documentation
│   ├── troubleshooting/      # GPU/CUDA guides
│   ├── implementation/       # Technical details
│   └── docker-docs/          # Container documentation
│
└── docker-historic/           # Archived Docker files
    ├── dockerfiles/          # Container definitions
    ├── compose/              # Docker Compose configs
    └── scripts/              # Container scripts
```

### Key Organizational Patterns
- **Minimal Root**: Only essential files in root directory
- **Documentation Archive**: All docs organized in `docs-archive/`
- **Docker Separation**: Container files archived in `docker-historic/`
- **PRP-Driven Development**: Feature specs in `PRPs/` directory

## 3. Getting Started

### Prerequisites
- **OS**: Linux (Ubuntu 20.04+, Fedora 35+) or macOS
- **Python**: 3.11 (required for pyannote.audio compatibility)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: Drivers installed and functional
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB for models and dependencies

### Environment Setup

#### Step 1: Clone and Navigate
```bash
git clone [repository-url]
cd whisper-api
```

#### Step 2: Create Virtual Environment
```bash
# Automated setup (recommended)
./setup_venv.sh

# Or manually
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Standard installation
pip install -r requirements.txt

# Or for specific Python 3.11 compatibility
pip install -r requirements_diarization.txt
```

#### Step 4: Configure HuggingFace Token (for diarization)
```bash
mkdir -p ~/.config/whisper
echo "HF_TOKEN=your_huggingface_token_here" > ~/.config/whisper/token
```
Get your token from: https://huggingface.co/settings/tokens

#### Step 5: Accept Pyannote License (for diarization)
Visit and accept: https://huggingface.co/pyannote/speaker-diarization-3.1

### Running the Project

#### Start the API Service
```bash
# Using management script (recommended)
./start_whisper.sh start

# Or directly
python main.py
```

#### Verify Service Health
```bash
curl http://localhost:8765/health | jq .
```

#### Stop the Service
```bash
./start_whisper.sh stop
```

### Running Tests
```bash
# Test transcription
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@test_audio.wav" | jq .

# Test with speaker diarization
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@test_audio.wav" \
  -F "diarize=true" | jq .
```

## 4. Key Components

### Entry Points
- **main.py**: FastAPI application entry point
  - Line 419: `uvicorn.run()` starts the server
  - Default port: 8765

### Core Business Logic
- **Transcription Pipeline** (main.py:188-248)
  - Audio file validation
  - Whisper model inference
  - Optional diarization processing
  - Format conversion (JSON/text/SRT/VTT)

### GPU Enforcement
- **gpu_validator.py**: Ensures GPU-only operation
  - No CPU fallback allowed
  - Validates CUDA availability
  - Checks memory requirements (4GB minimum)

### API Endpoints
- `GET /` - Service information
- `GET /health` - Health check with GPU status
- `POST /v1/transcribe` - Synchronous transcription
- `POST /v2/transcript` - Asynchronous transcription
- `GET /v2/transcript/{job_id}` - Get async results

### Configuration Management
- Environment variables control behavior:
  - `WHISPER_MODEL`: Model size (tiny/base/small/medium/large)
  - `WHISPER_DEVICE`: Always "cuda" (enforced)
  - `WHISPER_LANGUAGE`: Default language
  - `WHISPER_DIARIZE`: Enable diarization by default

## 5. Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

### Adding a New Feature
1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes following existing patterns
3. Test locally with various audio formats
4. Update documentation if needed
5. Create pull request with clear description

### Code Style
- Python: Follow PEP 8
- Use type hints for function parameters
- Document complex logic with inline comments
- Keep functions focused and under 50 lines

### Testing Requirements
- Test with multiple audio formats (WAV, MP3, M4A)
- Verify GPU memory usage stays within limits
- Check both sync and async endpoints
- Test error cases (invalid audio, missing files)

## 6. Architecture Decisions

### Design Patterns
- **Singleton GPU Validator**: Single instance ensures consistent GPU checks
- **Async Processing**: Background tasks for long-running transcriptions
- **Factory Pattern**: Model loading with fallback versions

### State Management
- **Stateless API**: No session management
- **In-Memory Job Storage**: Async jobs stored in dictionary (not persistent)
- **Temporary Files**: Cleaned up after processing

### Error Handling
- **GPU Enforcement**: Hard fail if GPU not available
- **Graceful Diarization Fallback**: Continue without speaker identification
- **Detailed Error Messages**: Include remediation steps

### Security Measures
- **File Validation**: Check audio file headers
- **Size Limits**: Prevent oversized uploads
- **Token Management**: HuggingFace tokens stored securely
- **No Exposed Secrets**: Environment variables for sensitive data

## 7. Common Tasks

### Add a New Whisper Model Size
```python
# In main.py, update WHISPER_MODEL options
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # Change default

# Verify GPU memory requirements for larger models
# tiny: 39M, base: 74M, small: 244M, medium: 769M, large: 1550M
```

### Change Default Port
```python
# In main.py, line 419
uvicorn.run(app, host="127.0.0.1", port=8765)  # Change port number
```

### Add Output Format
```python
# In main.py, add to format_output() function
elif format == "your_format":
    return YourFormatResponse(content=formatted_data)
```

### Debug Audio Processing
```bash
# Enable debug logs
export WHISPER_DEBUG=true
python main.py

# Check GPU memory usage
nvidia-smi -l 1  # Update every second
```

## 8. Potential Gotchas

### GPU/CUDA Issues
- **RTX 5060 Ti Warning**: Blackwell architecture (sm_120) shows PyTorch warnings but works
- **CUDA Mismatch**: Ensure PyTorch CUDA version matches system CUDA
- **Memory Errors**: Large models require more VRAM (8GB+ for large model)

### Diarization Limitations
- **Python Version**: Must use Python 3.11 for pyannote compatibility
- **License Required**: Must accept pyannote license on HuggingFace
- **Token Required**: HuggingFace token needed for model download
- **GPU Incompatibility**: Some newer GPUs may not support diarization models

### Common Errors
- **Port Already in Use**: Kill existing process or change port
- **Import Errors**: Ensure virtual environment is activated
- **File Not Found**: Use absolute paths or `@` prefix in curl
- **No GPU Found**: Check NVIDIA drivers with `nvidia-smi`

### Performance Considerations
- **Model Loading**: First request takes longer (model initialization)
- **Memory Leaks**: Monitor long-running instances
- **Concurrent Requests**: Limited by GPU memory
- **File Cleanup**: Temporary files accumulate if service crashes

## 9. Documentation and Resources

### Project Documentation
- `docs-archive/core-docs/`: Original README and guides
- `docs-archive/troubleshooting/`: GPU and CUDA solutions
- `PRPs/`: Feature specifications and planning

### External Resources
- [Faster Whisper Docs](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Documentation](https://github.com/pyannote/pyannote-audio)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### API Documentation
- Interactive docs: http://localhost:8765/docs (when running)
- OpenAPI schema: http://localhost:8765/openapi.json

## 10. Next Steps - New Developer Checklist

### Week 1: Environment Setup
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies successfully
- [ ] Configure HuggingFace token
- [ ] Run the API service
- [ ] Successfully transcribe a test audio file

### Week 2: Understanding the Codebase
- [ ] Read through main.py completely
- [ ] Understand GPU validation flow
- [ ] Trace a request through the transcription pipeline
- [ ] Review client library (whisper_client.py)
- [ ] Experiment with different model sizes

### Week 3: Making Contributions
- [ ] Fix a small bug or typo
- [ ] Add a helpful comment or docstring
- [ ] Improve error messages
- [ ] Add a new example to examples/
- [ ] Update documentation based on your learning

### Areas to Start Contributing
1. **Documentation**: Improve based on your onboarding experience
2. **Examples**: Add more client usage examples
3. **Error Handling**: Enhance error messages and recovery
4. **Testing**: Add test scripts for edge cases
5. **Performance**: Profile and optimize bottlenecks

## Support and Communication

### Getting Help
1. Check `docs-archive/troubleshooting/` for common issues
2. Review PRPs for feature context
3. Search existing GitHub issues
4. Check GPU compatibility guides

### Reporting Issues
When reporting issues, include:
- Python version
- GPU model and CUDA version
- Error messages and stack traces
- Steps to reproduce
- Audio file characteristics (format, duration)

---

## Quick Command Reference

```bash
# Service Management
./start_whisper.sh start|stop|status|restart

# Testing
curl -X POST http://localhost:8765/v1/transcribe -F "file=@audio.mp3"

# Monitoring
nvidia-smi                         # GPU status
curl http://localhost:8765/health  # Service health
tail -f ~/.whisper-api.log        # Service logs

# Python Client
python -c "from whisper_client import WhisperClient; print(WhisperClient().transcribe('audio.wav'))"
```

Welcome to the team! 🎉 Start with the checklist and don't hesitate to explore and improve the codebase.