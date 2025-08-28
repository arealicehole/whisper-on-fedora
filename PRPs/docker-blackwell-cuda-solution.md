name: "Docker-Based CUDA Solution for RTX 5060 Ti Blackwell GPU"
description: |
  Complete implementation of Docker containerized Whisper API with full CUDA acceleration
  for both transcription and diarization on RTX 5060 Ti (sm_120 architecture) using
  NVIDIA NGC containers with PyTorch CUDA 12.8 support.

---

## Goal

**Feature Goal**: Deploy a fully containerized Whisper API service with native Blackwell GPU support, eliminating the "no kernel image" error and enabling full CUDA acceleration for both whisper transcription and pyannote diarization.

**Deliverable**: Production-ready Docker container using NVIDIA NGC PyTorch base image with sm_120 support, configured Whisper API service, and persistent model caching.

**Success Definition**: Whisper API runs in Docker with both transcription and diarization using GPU acceleration on RTX 5060 Ti, models persist between restarts, existing audio files and scripts work unchanged.

## User Persona

**Target User**: Developer/researcher with RTX 5060 Ti GPU needing reliable speech transcription with speaker diarization

**Use Case**: Process audio files with GPU acceleration without managing complex Python dependencies

**User Journey**: 
1. Run single Docker command → Container starts
2. Existing whisper_client.py scripts work immediately
3. Models download once and persist
4. Service auto-starts on system boot
5. Full GPU acceleration for all operations

**Pain Points Addressed**: 
- PyTorch stable doesn't support sm_120 architecture
- Dependency conflicts between pyannote and PyTorch versions
- Manual environment management complexity
- Models re-downloading on each setup

## Why

- **Guaranteed Compatibility**: NGC containers are validated by NVIDIA for Blackwell GPUs
- **Zero Dependency Conflicts**: Isolated container environment
- **Production Stability**: Immutable environment that works identically everywhere
- **Existing Code Works**: No changes needed to current scripts/clients
- **Model Persistence**: Download once, use forever

## What

Docker container providing:
- Full CUDA 12.8 support for RTX 5060 Ti (sm_120)
- FastAPI server on port 8765
- Persistent model storage
- Automatic GPU detection and configuration
- Health monitoring endpoints
- Compatible with existing whisper_client.py

### Success Criteria

- [ ] Docker container builds successfully with NGC base image
- [ ] CUDA operations work without "no kernel image" error
- [ ] Diarization runs on GPU, not CPU fallback
- [ ] Models persist between container restarts
- [ ] Existing client code works without modification
- [ ] Service auto-restarts after system reboot
- [ ] Processing speed matches native GPU performance

## All Needed Context

### Context Completeness Check

_This PRP contains everything needed to implement a Docker-based solution for Blackwell GPU support without prior knowledge of the architecture compatibility issue._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
  why: NVIDIA NGC PyTorch containers with Blackwell support details
  critical: Version 25.01+ required for sm_120 architecture

- url: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
  why: NVIDIA Container Toolkit installation for GPU passthrough
  critical: Required for --gpus flag to work in Docker

- file: /home/ice/whisper-api/main.py
  why: Existing FastAPI application to containerize
  pattern: Current endpoints, configuration, dependencies
  gotcha: Hardcoded paths may need adjustment for container

- file: /home/ice/whisper-api/diarization_handler.py
  why: Diarization logic that needs GPU acceleration
  pattern: PyTorch CUDA operations for speaker detection
  gotcha: Must ensure torch.cuda.is_available() returns True

- file: /home/ice/whisper-api/whisper_client.py
  why: Client library that must remain compatible
  pattern: API endpoint calls, expected responses
  gotcha: Assumes localhost:8765, must maintain same port
```

### Current Codebase Structure

```bash
whisper-api/
├── main.py                      # FastAPI server
├── diarization_handler.py       # Speaker detection logic
├── whisper_client.py           # Client library
├── requirements_diarization.txt # Current dependencies
├── ~/.cache/huggingface/       # Existing model cache
├── ~/.cache/whisper/           # Existing whisper models
└── ~/.config/whisper/token     # HuggingFace token
```

### Desired Structure (with Docker)

```bash
whisper-api/
├── docker/
│   ├── Dockerfile.blackwell    # NGC-based container definition
│   ├── docker-compose.yml      # Service orchestration
│   └── .dockerignore          # Exclude unnecessary files
├── scripts/
│   ├── docker-start.sh        # Start containerized service
│   ├── docker-test.sh         # Test GPU support
│   └── docker-logs.sh         # View container logs
├── volumes/
│   ├── models/                # Persistent model storage
│   └── config/                # Configuration files
└── (existing files)
```

### Known Patterns and Conventions

```python
# Key CUDA compatibility check in container
import torch
assert torch.cuda.is_available(), "CUDA not available"
assert 'sm_120' in torch.cuda.get_arch_list(), "Blackwell not supported"

# Model cache paths that need volume mapping
HUGGINGFACE_CACHE = "/root/.cache/huggingface"  # Inside container
WHISPER_CACHE = "/root/.cache/whisper"
CONFIG_PATH = "/root/.config/whisper"
```

## Implementation Blueprint

### Data Models and Structure

Not applicable - using existing models and API structure.

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE docker/Dockerfile.blackwell
  - BASE: nvcr.io/nvidia/pytorch:25.02-py3 (has sm_120 support)
  - INSTALL: System deps (ffmpeg, git)
  - INSTALL: Python packages with CUDA 12.8 index URL
  - COPY: Application files (main.py, diarization_handler.py)
  - EXPOSE: Port 8765
  - HEALTHCHECK: curl http://localhost:8765/health
  - CMD: python main.py

Task 2: CREATE docker/docker-compose.yml
  - SERVICE: whisper-blackwell
  - RUNTIME: nvidia (for GPU access)
  - ENVIRONMENT: NVIDIA_VISIBLE_DEVICES=all, WHISPER_* configs
  - VOLUMES: Map model caches, config, audio directories
  - PORTS: 8765:8765
  - RESTART: unless-stopped
  - DEPLOY: GPU reservation for 1 device

Task 3: CREATE docker/.dockerignore
  - EXCLUDE: __pycache__, *.pyc, .git, audio files
  - EXCLUDE: Local virtual environments
  - INCLUDE: Only necessary Python files

Task 4: CREATE scripts/docker-start.sh
  - CHECK: Docker and nvidia-container-toolkit installed
  - BUILD: docker-compose build (if needed)
  - START: docker-compose up -d
  - VERIFY: Health check endpoint
  - REPORT: Success with API URL

Task 5: CREATE scripts/docker-test.sh
  - RUN: Diagnostic test inside container
  - CHECK: PyTorch version has +cu128
  - CHECK: torch.cuda.get_arch_list() includes sm_120
  - TEST: Simple CUDA tensor operation
  - TEST: Diarization pipeline GPU detection

Task 6: MODIFY docker-compose.yml for model persistence
  - ADD: Volume for ~/.cache/huggingface:/root/.cache/huggingface
  - ADD: Volume for ~/.cache/whisper:/root/.cache/whisper
  - ENSURE: Models downloaded on host are reused
  - BENEFIT: No re-downloading on container restart

Task 7: CREATE systemd service for auto-start
  - CREATE: /etc/systemd/system/whisper-docker.service
  - AFTER: docker.service
  - EXECSTART: docker-compose up
  - EXECSTOP: docker-compose down
  - RESTART: always
  - ENABLE: Start on boot
```

### Implementation Patterns & Key Details

```dockerfile
# Dockerfile.blackwell - Critical sections
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Install pyannote preserving CUDA support
RUN pip install pyannote.audio \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Verify Blackwell support at build time
RUN python -c "import torch; \
    assert 'sm_120' in torch.cuda.get_arch_list(), \
    'Container does not support Blackwell GPU'"
```

```yaml
# docker-compose.yml - GPU configuration
services:
  whisper-blackwell:
    runtime: nvidia
    environment:
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0  # RTX 5060 Ti
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      # Reuse existing model cache
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.cache/whisper:/root/.cache/whisper
      # Config with HF token
      - ~/.config/whisper:/root/.config/whisper:ro
```

```bash
# docker-start.sh - Startup verification
#!/bin/bash
# Test GPU inside container
docker-compose run --rm whisper-blackwell python -c "
import torch
assert torch.cuda.is_available()
assert 'sm_120' in torch.cuda.get_arch_list()
print('✅ Blackwell GPU ready!')
"
```

### Integration Points

```yaml
NETWORKING:
  - host: "localhost:8765 remains unchanged"
  - container: "Internal port 8765 mapped to host"

FILE_ACCESS:
  - audio_files: "./audio_files:/app/audio_files"
  - models: "~/.cache:/root/.cache for persistence"

CONFIGURATION:
  - env_file: ".env for WHISPER_* variables"
  - token: "~/.config/whisper mounted read-only"

MONITORING:
  - logs: "docker-compose logs -f whisper-blackwell"
  - stats: "docker stats whisper-blackwell"
```

## Validation Loop

### Level 1: Build Validation

```bash
# Build the Docker image
cd /home/ice/whisper-api
docker build -f docker/Dockerfile.blackwell -t whisper-blackwell:latest .

# Verify image has correct PyTorch
docker run --rm whisper-blackwell:latest python -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert '+cu128' in torch.__version__, 'Wrong PyTorch version'
"

# Expected: PyTorch version shows +cu128
```

### Level 2: GPU Access Validation

```bash
# Test GPU passthrough
docker run --rm --gpus all whisper-blackwell:latest nvidia-smi

# Test CUDA in container
docker run --rm --gpus all whisper-blackwell:latest python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
major, minor = torch.cuda.get_device_capability(0)
print(f'Compute Capability: sm_{major}{minor}')
"

# Expected: Shows RTX 5060 Ti with sm_120
```

### Level 3: Service Validation

```bash
# Start the service
docker-compose -f docker/docker-compose.yml up -d

# Wait for startup
sleep 10

# Check health endpoint
curl -f http://localhost:8765/health | jq .

# Test transcription
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@test.wav" \
  -F "diarize=false" | jq .

# Test diarization (GPU acceleration)
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@test.wav" \
  -F "diarize=true" | jq .

# Monitor GPU usage during diarization
nvidia-smi dmon -i 0 -s u -c 10

# Expected: GPU utilization increases during processing
```

### Level 4: Performance Validation

```bash
# Compare CPU vs GPU diarization speed
# This should show significant speedup with GPU

# Test file processing time
time curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@large_audio.wav" \
  -F "diarize=true" \
  -o /dev/null

# Check container resource usage
docker stats --no-stream whisper-blackwell

# Verify models are cached (no download on restart)
docker-compose down
docker-compose up -d
# Should start quickly without downloading models

# Expected: 
# - Processing time ~0.1x audio duration
# - Models load from cache instantly
```

## Final Validation Checklist

### Docker Setup Validation

- [ ] Dockerfile builds without errors
- [ ] NGC base image version 25.02 or newer
- [ ] PyTorch shows +cu128 in version string
- [ ] torch.cuda.get_arch_list() includes sm_120

### GPU Functionality

- [ ] nvidia-smi works inside container
- [ ] torch.cuda.is_available() returns True
- [ ] No "no kernel image" errors
- [ ] Diarization uses GPU (check with nvidia-smi)

### Service Operation

- [ ] API accessible on localhost:8765
- [ ] Health check endpoint responds
- [ ] Existing whisper_client.py works unchanged
- [ ] Models persist between restarts
- [ ] Processing speed matches GPU expectations

### Production Readiness

- [ ] Container auto-restarts on failure
- [ ] Service starts on system boot (systemd)
- [ ] Logs accessible via docker-compose logs
- [ ] Resource limits configured appropriately
- [ ] Model cache volumes properly mounted

---

## Anti-Patterns to Avoid

- ❌ Don't use PyTorch stable (lacks sm_120 support)
- ❌ Don't install pyannote without --extra-index-url
- ❌ Don't forget nvidia-container-toolkit
- ❌ Don't rebuild container unnecessarily (use volumes)
- ❌ Don't hardcode paths in container
- ❌ Don't run as root in production (add USER directive)

## Implementation Confidence Score: 9/10

This PRP provides a complete Docker-based solution for Blackwell GPU support. The only variable is the specific NGC container version availability, but 25.02 is confirmed to work.