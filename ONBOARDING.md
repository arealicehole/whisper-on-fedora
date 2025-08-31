# 🚀 Whisper API - Developer Onboarding Guide

Welcome to the Whisper API project! This comprehensive guide will help you understand the codebase, set up your development environment, and start contributing effectively.

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Key Components](#key-components)
5. [Development Workflow](#development-workflow)
6. [Architecture Decisions](#architecture-decisions)
7. [Common Tasks](#common-tasks)
8. [Potential Gotchas](#potential-gotchas)
9. [Documentation and Resources](#documentation-and-resources)
10. [Next Steps](#next-steps)

---

## 1. Project Overview

### **What is Whisper API?**
Whisper API is a production-ready, GPU-accelerated REST API service that provides:
- **Speech-to-text transcription** using OpenAI's Whisper model
- **Speaker diarization** (identifying who spoke when) using PyAnnote
- **Multiple output formats** (JSON, text, SRT, VTT)
- **Synchronous and asynchronous processing** endpoints

### **Tech Stack**
- **Language**: Python 3.11 (required for PyAnnote compatibility)
- **Framework**: FastAPI (async web framework)
- **ML Models**: 
  - Faster-Whisper (optimized Whisper implementation)
  - PyAnnote Audio 3.3.1 (speaker diarization)
- **GPU**: PyTorch 2.8.0+cu128 with CUDA support
- **Server**: Uvicorn ASGI server
- **Special Support**: NVIDIA Blackwell GPU architecture (RTX 5060 Ti)

### **Architecture Pattern**
- **Service-Oriented Architecture**: Single service with REST API
- **GPU-First Design**: Mandatory GPU execution, no CPU fallback
- **Job Queue Pattern**: Async processing with job status tracking
- **Dependency Injection**: Configuration via environment variables

### **Key Dependencies**
- `faster-whisper`: Optimized Whisper implementation using CTranslate2
- `pyannote.audio`: State-of-the-art speaker diarization
- `torch`: PyTorch deep learning framework (nightly build for Blackwell)
- `fastapi`: Modern async web framework
- `uvicorn`: Lightning-fast ASGI server

---

## 2. Repository Structure

### **Branch Strategy**
This project uses a **dual-branch strategy**:

#### **main branch** (Production - 10 files)
```
whisper-api/
├── main.py                 # FastAPI application entry point
├── gpu_validator.py        # GPU enforcement and validation
├── sitecustomize.py        # PyTorch patches for Blackwell GPU
├── startup.py              # Service initialization script
├── whisper_client.py       # Python client library
├── requirements_api.txt    # Core dependencies
├── requirements_blackwell.txt # Blackwell-specific deps
├── LICENSE                 # MIT license
├── .gitattributes         # Merge strategy configuration
└── .env.production        # Production environment settings
```

#### **dev branch** (Development - 140+ files)
```
whisper-api/
├── All production files above, plus:
├── PRPs/                   # Project Requirement Plans (40+ files)
│   ├── ai_docs/           # AI-generated documentation
│   └── templates/         # PRP templates
├── docs-archive/          # Documentation archive (38 files)
│   ├── troubleshooting/   # Troubleshooting guides
│   └── BRANCH_STRATEGY.md # Branch management docs
├── .claude/               # Claude AI assistant commands
│   └── commands/          # Custom command definitions
├── test_*.py              # Test scripts (8 files)
├── docker/                # Docker configurations
│   ├── Dockerfile.blackwell
│   └── docker-compose.yml
├── .github/               # GitHub configurations
│   └── workflows/         # GitHub Actions
├── scripts/               # Utility scripts
└── venv/                  # Virtual environment (gitignored)
```

### **Key Directories**
- **`/PRPs`**: Detailed project plans and architectural decisions
- **`/docs-archive`**: Historical documentation and guides
- **`/.claude`**: AI assistant integration for development
- **`/docker`**: Containerization for Blackwell GPU support
- **`/test_*.py`**: Various test scripts for different scenarios

### **Non-Standard Patterns**
- **Dual-branch strategy**: Minimal `main` for production, full `dev` for development
- **PRPs methodology**: Detailed planning documents before implementation
- **Blackwell-specific patches**: Custom PyTorch modifications for new GPU

---

## 3. Getting Started

### **Prerequisites**
```bash
# Required Software
- Python 3.11.x (MUST be 3.11 for PyAnnote compatibility)
- NVIDIA GPU with 4GB+ VRAM
- NVIDIA CUDA drivers (12.8+ for Blackwell)
- Git
- HuggingFace account (for model access)

# System Requirements
- OS: Linux (Fedora 42 tested) or Ubuntu 22.04+
- RAM: 8GB minimum
- Storage: 10GB free space
```

### **Environment Setup**

#### Step 1: Clone and Branch Setup
```bash
# Clone repository
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api

# Switch to dev branch for development
git checkout dev
```

#### Step 2: Python Environment
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Verify Python version
python --version  # Should show 3.11.x
```

#### Step 3: Install Dependencies

For **Standard GPUs** (non-Blackwell):
```bash
pip install -r requirements_api.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyannote.audio==3.3.1
```

For **Blackwell GPUs** (RTX 5060 Ti):
```bash
# Install PyTorch nightly with Blackwell support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements_blackwell.txt

# Install PyAnnote without dependencies to avoid PyTorch downgrade
pip install --no-deps pyannote.audio==3.3.1
pip install -r requirements_blackwell.txt  # Installs PyAnnote deps
```

#### Step 4: Configure HuggingFace Token
```bash
# Create config directory
mkdir -p ~/.config/whisper

# Add your HuggingFace token
echo "your_huggingface_token_here" > ~/.config/whisper/token

# Or set as environment variable
export HF_TOKEN="your_huggingface_token_here"
```

Get token from: https://huggingface.co/settings/tokens

Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1

#### Step 5: Environment Variables
```bash
# Copy development environment template
cp .env.development .env

# Edit as needed
nano .env
```

Default settings:
```env
DEBUG=true
RELOAD=true
LOG_LEVEL=DEBUG
TEST_MODE=true
WHISPER_MODEL=tiny
WHISPER_DEVICE=cuda
WHISPER_DIARIZE=true
PORT=8767
```

### **Running the Project**

#### Development Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Run with auto-reload
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --port 8767
```

#### Production Mode
```bash
# Switch to main branch
git checkout main

# Run without debug
python main.py
```

### **Running Tests**
```bash
# Test GPU availability
python test_gpu_basic.py

# Test transcription only
python test_whisper_only.py

# Test with diarization
python test_diarization_final.py

# Test MP3 processing
python test_mp3_diarization.py
```

### **Building for Production**
```bash
# Using Docker (recommended for Blackwell)
cd docker
docker-compose up -d whisper-blackwell

# Or traditional deployment
git checkout main
pip install -r requirements_api.txt
python main.py
```

---

## 4. Key Components

### **Entry Points**

#### `main.py` (lines: ~600)
Main FastAPI application with all endpoints:
```python
# Key globals
model: WhisperModel  # Faster-Whisper model instance
diarization_pipeline: Pipeline  # PyAnnote diarization
jobs_storage: Dict  # Async job tracking

# Main endpoints
@app.post("/v1/transcribe")  # Sync transcription
@app.post("/v2/transcript")  # Async job creation
@app.get("/v2/transcript/{job_id}")  # Job status
@app.get("/health")  # Health check with GPU info
```

#### `startup.py`
Service initialization and model loading:
- Validates GPU availability
- Loads Whisper model to GPU
- Initializes diarization pipeline
- Sets up error handlers

### **Core Business Logic**

#### `gpu_validator.py`
Enforces GPU-only execution:
```python
class GPUValidator:
    def validate_environment() -> GPUValidationResult
    def enforce_gpu_execution() -> None
    def get_device_info() -> Dict
    
class GPUEnforcementError(Exception):
    # Raised when GPU requirements not met
```

#### `sitecustomize.py`
Blackwell GPU compatibility patches:
- Patches torchvision NMS for PyAnnote
- Handles CUDA architecture mismatches
- Auto-loaded by Python interpreter

### **API Endpoints**

#### Synchronous Transcription
```python
POST /v1/transcribe
- file: Audio file (required)
- diarize: Enable speaker identification
- language: Target language
- format: Output format (json/text/srt/vtt)
```

#### Asynchronous Processing
```python
POST /v2/transcript
- Submit transcription job
- Returns job_id immediately

GET /v2/transcript/{job_id}
- Check job status
- Retrieve results when complete
```

### **Configuration Management**
- Environment variables in `.env` files
- Runtime configuration in `main.py`
- Model selection via `WHISPER_MODEL`
- GPU device via `WHISPER_DEVICE`

### **Authentication/Authorization**
- Currently no auth implemented
- HuggingFace token for model downloads
- Ready for auth middleware addition

---

## 5. Development Workflow

### **Git Branch Conventions**
```bash
main       # Production-ready, minimal runtime
dev        # Active development, all resources
feature/*  # New features
fix/*      # Bug fixes
test/*     # Testing branches
```

### **Creating a New Feature**

1. **Start from dev branch**
```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

2. **Develop and test**
```bash
# Make changes
# Run tests
python test_whisper_only.py
# Commit changes
git add .
git commit -m "feat: add your feature description"
```

3. **Push and create PR**
```bash
git push origin feature/your-feature-name
# Create PR from feature -> dev on GitHub
```

### **Testing Requirements**
- Test GPU functionality first
- Test with multiple audio formats (WAV, MP3, M4A)
- Test both with and without diarization
- Verify memory usage stays within limits

### **Code Style/Linting**
```python
# Python style
- PEP 8 compliance
- Type hints where beneficial
- Docstrings for public functions
- f-strings for formatting

# Naming conventions
- snake_case for functions/variables
- PascalCase for classes
- UPPER_CASE for constants
```

### **PR Process**
1. Create PR from feature branch to `dev`
2. Ensure all tests pass
3. Update relevant documentation
4. Request review from team lead
5. Merge to `dev` after approval
6. Runtime files auto-sync to `main` via GitHub Actions

### **CI/CD Pipeline**
- **GitHub Actions**: `.github/workflows/sync-to-main.yml`
- Automatically syncs runtime files from `dev` to `main`
- Triggers on push to `dev` branch
- Only syncs essential runtime files

---

## 6. Architecture Decisions

### **Design Patterns**

#### **GPU-First Architecture**
- No CPU fallback by design
- Fails fast if GPU unavailable
- Enforces consistent performance

#### **Job Queue Pattern**
```python
jobs_storage = {}  # In-memory job storage
# Could extend to Redis/PostgreSQL for production
```

#### **Singleton Model Loading**
- Models loaded once at startup
- Shared across all requests
- Reduces memory overhead

### **State Management**
- Stateless API design
- Job state in memory (consider persistence for production)
- Model state persistent across requests

### **Error Handling Strategy**
```python
try:
    # GPU operations
except GPUEnforcementError:
    # Return 503 Service Unavailable
except Exception as e:
    # Log and return 500 with details
```

### **Logging and Monitoring**
- Structured logging with levels
- GPU metrics in health endpoint
- Performance metrics per request

### **Security Measures**
- File upload validation
- Size limits on uploads
- Temporary file cleanup
- Token management for HuggingFace

### **Performance Optimizations**
- Faster-Whisper for 10x speed improvement
- GPU memory pre-allocation
- Batch processing capability
- Model caching in VRAM

---

## 7. Common Tasks

### **Adding a New API Endpoint**

1. **Define in main.py**
```python
@app.post("/v1/your-endpoint")
async def your_endpoint(
    file: UploadFile = File(...),
    param: str = Form(None)
):
    # Implementation
    return {"result": "value"}
```

2. **Add to client library** (whisper_client.py)
```python
def your_method(self, file_path: str, param: str = None):
    files = {"file": open(file_path, "rb")}
    data = {"param": param}
    response = requests.post(f"{self.api_url}/v1/your-endpoint", 
                            files=files, data=data)
    return response.json()
```

### **Adding a New Output Format**

1. **Extend format_output function**
```python
def format_output(segments, format_type, diarize=False):
    if format_type == "your_format":
        # Format implementation
        return formatted_string
```

2. **Update endpoint validation**
```python
format: Literal["json", "text", "srt", "vtt", "your_format"]
```

### **Creating a New Test**

1. **Create test file**
```python
# test_your_feature.py
import sys
sys.path.append('.')
from whisper_client import WhisperClient

def test_your_feature():
    client = WhisperClient()
    result = client.your_method("test.wav")
    assert result["status"] == "success"
    
if __name__ == "__main__":
    test_your_feature()
    print("✅ Test passed")
```

### **Debugging Common Issues**

#### GPU Not Detected
```python
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU info
nvidia-smi

# Verify in code
python gpu_validator.py
```

#### Diarization Failures
```python
# Check token
cat ~/.config/whisper/token

# Test pipeline directly
python test_diarization_final.py

# Check PyAnnote version
pip show pyannote.audio  # Should be 3.3.1
```

### **Updating Dependencies**

For standard GPUs:
```bash
pip install --upgrade faster-whisper
pip install --upgrade pyannote.audio
```

For Blackwell GPUs:
```bash
# Update PyTorch nightly
pip install --pre --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 8. Potential Gotchas

### **Non-Obvious Configurations**

#### **Python Version Lock**
- **MUST use Python 3.11** for PyAnnote compatibility
- Python 3.12+ breaks PyAnnote dependencies
- Python 3.10 lacks required features

#### **Blackwell GPU Special Setup**
- Requires PyTorch nightly builds
- Custom patches in `sitecustomize.py`
- May break with PyTorch stable releases

### **Required Environment Variables**
```bash
# Critical for diarization
HF_TOKEN=your_token_here

# GPU enforcement
WHISPER_DEVICE=cuda  # Never set to 'cpu'

# Model size affects VRAM usage
WHISPER_MODEL=base  # tiny/base/small/medium/large
```

### **External Service Dependencies**
- **HuggingFace Hub**: Model downloads
- **PyAnnote Models**: Requires license acceptance
- **NVIDIA Drivers**: Must match CUDA toolkit version

### **Known Issues/Workarounds**

#### Issue: PyAnnote NMS errors on Blackwell
**Workaround**: Applied via `sitecustomize.py` auto-patch

#### Issue: Port 8767 already in use
**Fix**: 
```bash
lsof -i :8767
kill -9 [PID]
```

#### Issue: OOM errors with large files
**Fix**: Use smaller model or increase GPU memory

### **Performance Bottlenecks**
- First request slow (model loading)
- Diarization adds 2-3x processing time
- Large models (medium/large) significantly slower

### **Technical Debt**
- In-memory job storage (needs persistence)
- No authentication system
- Limited error recovery in async jobs
- Manual dependency management for Blackwell

---

## 9. Documentation and Resources

### **Existing Documentation**

#### In Repository
- `README.md` - Project overview and quick start
- `QUICKSTART.md` - Rapid setup guide
- `docs-archive/BRANCH_STRATEGY.md` - Git workflow
- `PRPs/*.md` - Detailed implementation plans
- `docs-archive/troubleshooting/*.md` - Issue resolution

#### API Documentation
- Interactive docs: http://localhost:8767/docs (when running)
- OpenAPI schema: http://localhost:8767/openapi.json

### **Database Schemas**
Currently no database (in-memory storage)

Future schema for job persistence:
```sql
CREATE TABLE transcription_jobs (
    id UUID PRIMARY KEY,
    status VARCHAR(20),
    input_file VARCHAR(255),
    result JSONB,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### **Deployment Guides**
- Docker deployment: `docker/README.md`
- Systemd service: `scripts/whisper.service`
- PM2 configuration: `ecosystem.config.js`

### **Team Conventions**
- Use PRPs for major changes
- Test on Blackwell GPU before merge
- Document GPU-specific changes
- Keep main branch minimal

### **External Resources**
- [Faster Whisper Docs](https://github.com/guillaumekln/faster-whisper)
- [PyAnnote Documentation](https://github.com/pyannote/pyannote-audio)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [PyTorch Blackwell Support](https://pytorch.org/get-started/locally/)

---

## 10. Next Steps - Developer Checklist

### **Week 1: Environment Setup**
- [ ] Set up development environment on `dev` branch
- [ ] Configure Python 3.11 virtual environment
- [ ] Install dependencies for your GPU type
- [ ] Configure HuggingFace token
- [ ] Run `python main.py` successfully
- [ ] Access API docs at http://localhost:8767/docs

### **Week 1: Basic Testing**
- [ ] Run `test_gpu_basic.py` - verify GPU detection
- [ ] Run `test_whisper_only.py` - test transcription
- [ ] Run `test_diarization_final.py` - test speaker identification
- [ ] Test with your own audio file using curl
- [ ] Test using `whisper_client.py`

### **Week 2: Code Familiarization**
- [ ] Understand main.py structure and endpoints
- [ ] Review gpu_validator.py enforcement logic
- [ ] Trace a request through the transcription flow
- [ ] Understand async job processing
- [ ] Review error handling patterns

### **Week 2: First Contribution**
- [ ] Pick a small issue or improvement
- [ ] Create feature branch from `dev`
- [ ] Make changes following code style
- [ ] Test thoroughly
- [ ] Submit PR with clear description

### **Ongoing: Deep Dives**
- [ ] Understand Blackwell GPU optimizations
- [ ] Learn PyAnnote diarization pipeline
- [ ] Explore Faster-Whisper optimizations
- [ ] Review PRPs for architecture decisions
- [ ] Contribute to documentation improvements

---

## 🎯 Key Success Factors

1. **Always work on `dev` branch** for development
2. **Test GPU functionality first** before any changes
3. **Maintain Python 3.11** compatibility
4. **Document Blackwell-specific changes** clearly
5. **Follow the PRP process** for major changes

## 🆘 Getting Help

- Check `docs-archive/troubleshooting/` for common issues
- Review PRPs for implementation details
- Test scripts in root directory for examples
- GitHub Issues for bug reports
- Team chat for quick questions

## 🚀 Ready to Contribute!

You now have everything needed to understand and contribute to the Whisper API project. Start with the checklist above and gradually explore deeper aspects of the codebase.

Remember: This is a GPU-first, production-ready API service. Performance and reliability are paramount.

Welcome to the team! 🎉