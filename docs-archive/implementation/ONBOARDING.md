# Developer Onboarding Guide - Whisper API

Welcome to the Whisper API project! This comprehensive guide will help you understand the codebase and get productive quickly.

## 1. Project Overview

### Project Purpose
**Whisper API** is a production-ready REST API service that provides GPU-accelerated speech-to-text transcription with optional speaker diarization (speaker identification). It wraps OpenAI's Whisper model with enterprise features and optimizations.

### Main Functionality
- **Audio Transcription**: Convert speech to text using Whisper AI models
- **Speaker Diarization**: Identify and label different speakers in audio
- **Multiple Output Formats**: JSON, plain text, SRT, VTT subtitles
- **Sync & Async Processing**: Handle both real-time and batch workloads
- **GPU Acceleration**: Leverages NVIDIA GPUs for faster processing

### Tech Stack
- **Language**: Python 3.11 (required for pyannote compatibility)
- **Framework**: FastAPI (async REST API framework)
- **AI/ML Libraries**:
  - `faster-whisper`: GPU-accelerated Whisper implementation
  - `pyannote.audio`: Speaker diarization pipeline
  - `torch`: PyTorch for deep learning operations
- **Server**: Uvicorn (ASGI server)
- **Audio Processing**: soundfile, librosa, numpy
- **Infrastructure**: Docker, systemd service support

### Architecture Pattern
- **Service-Oriented Architecture**: Single microservice with clear API boundaries
- **Pipeline Pattern**: Audio → Transcription → Diarization → Formatting
- **Hybrid Processing**: GPU for transcription, CPU for diarization (RTX 5060 Ti compatibility)

### Key Dependencies
```
faster-whisper==1.0.3      # GPU-accelerated transcription
pyannote.audio==3.3.1      # Speaker diarization
fastapi==0.115.0           # REST API framework
torch==2.3.0+cpu           # Deep learning (CPU mode for compatibility)
uvicorn[standard]==0.32.0  # ASGI server
numpy==1.26.4              # Numerical operations
librosa==0.10.2            # Audio analysis
```

## 2. Repository Structure

```
whisper-api/
│
├── main.py                    # 🎯 Main FastAPI application
├── whisper_client.py          # Python client library
├── diarization_handler.py     # Enhanced diarization with error recovery
├── whisper-cli.sh            # CLI wrapper for API calls
│
├── tests/                     # Test suite
│   └── test_diarization_comprehensive.py
│
├── examples/                  # Usage examples
│   ├── basic_usage.py
│   └── README.md
│
├── PRPs/                      # Project Requirements Plans
│   ├── templates/
│   └── diarization-testing-hardening.md
│
├── Setup & Configuration
│   ├── setup.sh              # Initial setup script
│   ├── setup_venv.sh         # Virtual environment setup
│   ├── install_diarization.sh # Diarization dependencies
│   ├── fix_cuda_fallback.sh  # CUDA compatibility fix
│   └── start_whisper.sh      # Service management
│
├── Testing & Diagnostics
│   ├── test_transcribe.py    # Transcription tests
│   ├── test_diarization.py   # Diarization diagnostics
│   ├── test_hybrid_mode.py   # Hybrid mode verification
│   └── cuda_diagnostic.py    # CUDA compatibility checker
│
├── Configuration Files
│   ├── requirements.txt      # Python dependencies
│   ├── requirements_diarization.txt
│   ├── Dockerfile            # Container definition
│   ├── docker-compose.yml    # Docker orchestration
│   └── whisper-api.service   # Systemd service unit
│
└── Documentation
    ├── README.md             # Project overview
    ├── QUICKSTART.md         # Quick setup guide
    ├── CLAUDE.md             # AI assistant instructions
    └── CONTRIBUTING.md       # Contribution guidelines
```

### Directory Purposes
- **Root**: Core application files and entry points
- **tests/**: Comprehensive test suite for all components
- **examples/**: Code samples and integration examples
- **PRPs/**: Structured project planning documents
- **Scripts**: Setup, configuration, and management utilities

## 3. Getting Started

### Prerequisites
- **OS**: Linux (Fedora/Ubuntu/Debian tested)
- **Python**: 3.11 (required for pyannote compatibility)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **CUDA**: 12.1+ for newer GPUs, 11.8 for older
- **RAM**: 4-8GB depending on model size
- **Disk**: ~5GB for models and dependencies

### Environment Setup

1. **Clone and navigate to project**:
```bash
cd /home/ice/whisper-api
```

2. **Install Python 3.11** (if needed):
```bash
# Fedora
sudo dnf install python3.11 python3.11-devel

# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv python3.11-dev
```

3. **Create virtual environment**:
```bash
python3.11 -m venv ~/.venvs/whisper-diarize
source ~/.venvs/whisper-diarize/bin/activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Configure HuggingFace token** (for diarization):
```bash
mkdir -p ~/.config/whisper
echo "HF_TOKEN=your_token_here" > ~/.config/whisper/token
```
Get token from: https://huggingface.co/settings/tokens
Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1

### Running the Project

**Development mode**:
```bash
python main.py
# API runs on http://localhost:8765
```

**Production mode**:
```bash
./start_whisper.sh start
```

**Docker mode**:
```bash
docker-compose up -d
```

### Running Tests
```bash
# Test transcription
python test_transcribe.py sample.wav

# Test diarization setup
python test_diarization.py

# Run comprehensive test suite
pytest tests/ -v

# Test hybrid mode (RTX 5060 Ti)
python test_hybrid_mode.py
```

### Building for Production
```bash
# Build Docker image
docker build -t whisper-api:latest .

# Install as systemd service
sudo ./install_service.sh
```

## 4. Key Components

### Entry Points

**main.py** (lines 500-506):
- FastAPI application initialization
- Uvicorn server configuration
- Port: 8765 (default)

### Core Business Logic

**transcribe_audio()** (main.py:176-263):
- Core transcription function
- Handles Whisper processing
- Optional diarization integration
- Segment extraction and formatting

**DiarizationHandler** (diarization_handler.py):
- Enhanced error recovery
- Speaker embedding cache
- Retry logic with exponential backoff
- CUDA fallback mechanisms

### API Endpoints

1. **GET /** - Service information
2. **GET /health** - Health check with diarization status
3. **POST /v1/transcribe** - Synchronous transcription
   - Parameters: file, diarize, num_speakers, language, format
4. **POST /v2/transcript** - Asynchronous transcription
5. **GET /v2/transcript/{job_id}** - Get async job results

### Database/Storage
- **In-memory job storage**: `jobs_storage` dictionary
- **Speaker embedding cache**: LRU cache in diarization handler
- **Temporary files**: Uses tempfile for audio processing

### Configuration Management
- Environment variables for runtime config
- `~/.config/whisper/token` for HF authentication
- Hybrid mode detection for GPU compatibility

### Authentication
- HuggingFace token for pyannote models
- No API authentication (add if needed for production)

## 5. Development Workflow

### Git Branch Strategy
```bash
main         # Production-ready code
develop      # Integration branch
feature/*    # New features
fix/*        # Bug fixes
test/*       # Experimental changes
```

### Creating a New Feature

1. **Create branch**:
```bash
git checkout -b feature/your-feature
```

2. **Make changes** following patterns in codebase

3. **Test locally**:
```bash
python test_transcribe.py
curl -X POST http://localhost:8765/v1/transcribe -F "file=@test.wav"
```

4. **Run tests**:
```bash
pytest tests/
```

5. **Commit with meaningful message**:
```bash
git add .
git commit -m "feat: add support for new audio format"
```

### Code Style
- **Python**: Follow PEP 8
- **Docstrings**: Use for all public functions
- **Type hints**: Preferred for function signatures
- **Line length**: 100 characters max

### Testing Requirements
- Unit tests for new functions
- Integration tests for API changes
- Performance benchmarks for optimizations

### CI/CD Pipeline
- Currently manual deployment
- Future: GitHub Actions for automated testing
- Docker builds for containerized deployment

## 6. Architecture Decisions

### Design Patterns

**Pipeline Pattern**:
- Audio → Preprocessing → Transcription → Diarization → Postprocessing
- Each stage can fail independently with fallbacks

**Singleton Pattern**:
- DiarizationHandler uses singleton for resource management
- Model loading happens once at startup

**Factory Pattern**:
- Output formatting based on format parameter

### State Management
- Stateless API design
- Job state tracked in memory (consider Redis for production)
- Model state persisted across requests

### Error Handling Strategy
```python
try:
    # Core operation
except CUDAError:
    # Fallback to CPU
except ModelLoadError:
    # Try alternative model
except Exception:
    # Log and return error response
```

### Logging
- Console logging in development
- File logging via nohup in production
- Structured logging for debugging

### Security Measures
- Input validation on file uploads
- Temporary file cleanup
- Token storage in separate config file
- No direct command execution

### Performance Optimizations
- GPU acceleration for transcription
- Speaker embedding caching
- Batch processing support
- Memory-mapped model loading

## 7. Common Tasks

### Adding a New API Endpoint

1. Define in main.py:
```python
@app.post("/v1/your-endpoint")
async def your_endpoint(
    param: str = Form(...)
):
    # Implementation
    return {"result": "..."}
```

2. Add to API info in root endpoint
3. Document in README

### Creating a New Audio Processor

1. Create processor function:
```python
def process_audio_custom(audio_path: str) -> Dict:
    # Load audio
    # Process
    # Return results
```

2. Integrate with main pipeline
3. Add error handling

### Adding a Test

1. Create test file in tests/:
```python
def test_your_feature():
    # Arrange
    # Act
    # Assert
    assert result == expected
```

2. Run with pytest

### Debugging Common Issues

**CUDA errors**:
```bash
python cuda_diagnostic.py
./fix_cuda_fallback.sh  # For RTX 5060 Ti
```

**Diarization not loading**:
```bash
python test_diarization.py
# Check HF token and model access
```

**Out of memory**:
- Reduce batch size
- Use smaller model (tiny/base)
- Enable CPU fallback

### Updating Dependencies
```bash
pip list --outdated
pip install --upgrade package_name
# Test thoroughly after updates
```

## 8. Potential Gotchas

### Non-Obvious Configurations
- **Hybrid Mode**: RTX 5060 Ti automatically uses CPU for diarization
- **VAD disabled**: Voice Activity Detection filtered too aggressively
- **Model sizes**: tiny=39M, base=74M, small=244M, medium=769M, large=1550M

### Required Environment Variables
```bash
WHISPER_MODEL=tiny       # Model size
WHISPER_DEVICE=cuda      # or cpu
WHISPER_COMPUTE=float16  # Precision
WHISPER_LANGUAGE=en      # Default language
WHISPER_DIARIZE=true     # Enable diarization
HF_TOKEN=hf_xxxxx        # In ~/.config/whisper/token
```

### External Dependencies
- NVIDIA drivers for GPU support
- CUDA toolkit (optional, included in PyTorch)
- FFmpeg for audio format conversion
- Internet for first model download

### Known Issues
1. **RTX 5060 Ti**: Requires hybrid mode (sm_120 not supported)
2. **Python 3.12**: Compatibility issues with pyannote
3. **Memory leaks**: Can occur with very long audio files
4. **Port conflicts**: Default 8765 might be in use

### Performance Bottlenecks
- Model loading: First request takes longer
- Diarization: CPU-bound in hybrid mode
- Large files: Consider chunking for >1 hour audio
- Concurrent requests: Limited by GPU memory

### Technical Debt
- In-memory job storage (needs Redis for production)
- No request authentication
- Limited error recovery in async jobs
- Manual deployment process

## 9. Documentation and Resources

### Existing Documentation
- **README.md**: Comprehensive project overview
- **QUICKSTART.md**: Essential setup steps
- **CLAUDE.md**: AI assistant context
- **examples/README.md**: Integration examples
- **PRPs/**: Detailed technical plans

### API Documentation
Access at `http://localhost:8765/docs` (FastAPI automatic docs)

### Database Schema
No database currently (in-memory storage)

### Deployment Guides
- Docker: See docker-compose.yml
- Systemd: See whisper-api.service
- Manual: Use start_whisper.sh

### Team Conventions
- Meaningful commit messages
- Test before pushing
- Document API changes
- Update requirements.txt for new dependencies

## 10. Next Steps - Developer Checklist

### Essential Tasks
- [ ] Set up Python 3.11 environment
- [ ] Install dependencies
- [ ] Configure HuggingFace token
- [ ] Run `python main.py` successfully
- [ ] Test API with curl: `curl http://localhost:8765/health`
- [ ] Make a test transcription
- [ ] Run the test suite

### Understanding the Codebase
- [ ] Read through main.py
- [ ] Understand transcribe_audio() flow
- [ ] Review API endpoints
- [ ] Explore diarization_handler.py
- [ ] Check examples/basic_usage.py

### First Contribution
- [ ] Fix a small bug or typo
- [ ] Add a test case
- [ ] Improve error messages
- [ ] Update documentation
- [ ] Add an example script

### Advanced Tasks
- [ ] Optimize performance
- [ ] Add new output format
- [ ] Implement authentication
- [ ] Add Redis for job storage
- [ ] Create GitHub Actions workflow

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| CUDA not found | Run `./fix_cuda_fallback.sh` |
| Diarization fails | Check HF token, accept model license |
| Port in use | `lsof -i:8765` and kill process |
| Out of memory | Use smaller model or CPU mode |
| No segments found | Check audio quality, disable VAD |
| Python version | Must use Python 3.11 |

## Support Channels

- GitHub Issues: Report bugs and request features
- Documentation: Check README and QUICKSTART
- Logs: `/tmp/whisper.log` or `~/.whisper-api.log`
- Diagnostics: Run test scripts in root directory

## Performance Expectations

- **Transcription Speed**: 0.1-0.5x real-time (GPU)
- **Diarization Speed**: 0.5-1x real-time (CPU)
- **Memory Usage**: 2-4GB for small model
- **Startup Time**: 5-10 seconds
- **API Latency**: <100ms overhead

---

**Welcome aboard!** Start with the checklist in Section 10 and don't hesitate to explore the codebase. The test files are great examples of how everything works together.