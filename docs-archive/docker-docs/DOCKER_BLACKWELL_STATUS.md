# Docker Blackwell GPU Status Report

## Current Situation
The RTX 5060 Ti Blackwell GPU (compute capability 12.0) is experiencing CUDA initialization errors in Docker containers, despite the GPU being properly detected by the host system.

## Issues Encountered

### 1. NGC PyTorch Container (25.02)
- **Error**: "ERROR: The NVIDIA Driver is present, but CUDA failed to initialize. [[ Unknown error (error 999) ]]"
- **Cause**: NGC containers appear to have compatibility issues with Blackwell architecture

### 2. Standard CUDA Base Images
- **Error**: "RuntimeError: CUDA failed with error unknown error"
- **Cause**: CUDA runtime mismatch between host (driver 580.65.06, CUDA 13.0) and container (CUDA 12.6)

### 3. PyTorch Architecture Support
- PyTorch stable releases don't include sm_120 support yet
- PyTorch nightly builds claim sm_120 support but fail at runtime in containers

## Technical Details
- **Host GPU**: NVIDIA GeForce RTX 5060 Ti
- **Compute Capability**: 12.0 (sm_120)
- **Driver Version**: 580.65.06
- **Host CUDA**: 13.0
- **Docker Runtime**: nvidia-container-toolkit configured correctly

## What Works
✅ Host system can access GPU directly
✅ nvidia-smi works in containers (GPU visible)
✅ Docker GPU passthrough is configured correctly
✅ Standard CUDA containers work with older GPUs

## What Doesn't Work
❌ CUDA initialization in any Docker container
❌ PyTorch CUDA operations in containers
❌ faster-whisper GPU acceleration in containers
❌ pyannote.audio GPU operations in containers

## Root Cause
The Blackwell architecture (RTX 5060 Ti) requires very specific driver/CUDA combinations that aren't yet fully supported in containerized environments. The issue appears to be:

1. **Driver-Container Mismatch**: The host has a very new driver (580.65.06) that containers can't properly interface with
2. **CUDA Version Gap**: Containers use CUDA 12.6-12.8, but the host driver expects CUDA 13.0
3. **Kernel Module Issues**: Blackwell GPUs may require kernel module features not yet exposed to containers

## Alternative Solutions

### Option 1: Run Directly on Host (Recommended)
Since the Blackwell GPU works on the host system, the most reliable solution is to run the Whisper API directly without Docker:

```bash
# Install dependencies on host
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Run service
python main.py
```

### Option 2: Wait for Container Support
NVIDIA will likely release updated NGC containers with proper Blackwell support in the coming months.

### Option 3: Use CPU in Container
If containerization is required, run with CPU-only mode until GPU support improves.

## Files Created During Docker Attempts
- `docker/Dockerfile.blackwell` - NGC-based attempt
- `docker/Dockerfile.blackwell.v2` - CUDA base attempt  
- `docker/Dockerfile.blackwell.working` - PyTorch nightly attempt
- `docker/docker-compose.yml` - Docker Compose configuration
- `scripts/docker-start.sh` - Start script
- `scripts/docker-test.sh` - Test script

## Conclusion
While we successfully built Docker images and configured all necessary components, the RTX 5060 Ti Blackwell GPU cannot currently be used for CUDA operations within Docker containers due to driver/runtime incompatibilities. 

**The recommended approach is to run the Whisper API directly on the host system where the GPU functions correctly.**

---
*Last Updated: 2025-08-27*