# CUDA Initialization Issue - RTX 5060 Ti Blackwell

## Issue Description

The system has an RTX 5060 Ti (Blackwell architecture, compute capability 12.0) that is properly detected by nvidia-smi and NVML, but PyTorch cannot initialize CUDA, resulting in the error:

```
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment
```

## System Configuration

- **GPU**: NVIDIA GeForce RTX 5060 Ti (Blackwell, sm_120)  
- **Driver**: 580.65.06
- **CUDA Runtime**: 12.9 (nvcc)
- **PyTorch**: 2.9.0.dev20250827+cu129 (nightly)
- **Python**: 3.11 (virtual environment)
- **OS**: Fedora 42

## Diagnosis Results

1. **nvidia-smi**: ✅ Works correctly, shows GPU
2. **NVML (nvidia-ml-py)**: ✅ Can access GPU, reports correct compute capability
3. **Direct CUDA driver API**: ❌ Fails with error 999 (unknown error)
4. **PyTorch CUDA**: ❌ Cannot initialize

## Root Cause

The issue appears to be a system-level permission or configuration problem that prevents user-space CUDA initialization, possibly related to:

1. **Hybrid GPU Setup**: System has both GTX 1060 (nouveau driver) and RTX 5060 Ti (nvidia driver)
2. **Permission Issues**: CUDA device files may not have proper permissions
3. **Driver/Runtime Mismatch**: Despite matching versions, initialization fails

## Current Workaround

The service is configured to automatically fall back to CPU when CUDA initialization fails:

```python
# In main.py, the code gracefully handles CUDA unavailability
if torch.cuda.is_available():
    # Use GPU
    device = "cuda"
else:
    # Fall back to CPU
    device = "cpu"
```

## Permanent Fix Options

### Option 1: System Administrator Fix
```bash
# Run as root/sudo
nvidia-modprobe -u -c=0
chmod 666 /dev/nvidia*
```

### Option 2: Docker Solution
Use the provided Dockerfile.blackwell which runs with proper privileges:
```bash
docker build -f Dockerfile.blackwell -t whisper-blackwell .
docker run --gpus all -p 8765:8765 whisper-blackwell
```

### Option 3: Wait for System Update
Monitor for updates to:
- NVIDIA drivers (> 580.65.06)
- PyTorch stable release with official sm_120 support

## Testing GPU Access

To test if the issue is resolved:

```bash
source ~/.venvs/whisper-blackwell/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output when fixed:
```
CUDA available: True
```

## Performance Impact

When running on CPU:
- Whisper transcription: ~5-10x slower than GPU
- Diarization: ~3-5x slower than GPU
- Still functional but with reduced throughput

## Environment Setup

The Python 3.11 environment with all dependencies is ready at:
```
~/.venvs/whisper-blackwell
```

To activate:
```bash
source ~/.venvs/whisper-blackwell/bin/activate
```

## Next Steps

1. Monitor PyTorch releases for stable Blackwell support
2. Consider Docker deployment for production use
3. Contact system administrator for device permission fixes

## Related Files

- `/home/ice/whisper-api/blackwell_diagnostic.py` - Comprehensive diagnostic tool
- `/home/ice/whisper-api/test_blackwell_gpu.py` - GPU test script
- `/home/ice/whisper-api/test_cuda_init.py` - CUDA initialization test
- `/home/ice/whisper-api/test_nvidia_ml.py` - NVML direct access test