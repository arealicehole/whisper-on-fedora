# PyTorch Setup for NVIDIA Blackwell GPUs (RTX 5060 Ti)

## Critical Information

The NVIDIA RTX 5060 Ti uses the **Blackwell architecture** with compute capability **sm_120** (12.0). This is NOT supported by any stable PyTorch release as of 2025. You MUST use PyTorch nightly builds.

## Installation Commands

### Primary Solution (Tested & Working)
```bash
# Python 3.11 + PyTorch Nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Alternative Solutions
```bash
# CUDA 12.9 variant (newer, may have better optimizations)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# Build from source (most control, longest setup)
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="12.0"
export USE_CUDA=1
python setup.py install
```

## Verification Script

Always verify GPU support after installation:

```python
import torch

# Basic checks
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Detailed GPU info
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # CRITICAL: Check compute capability
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: sm_{capability[0]}{capability[1]}")
    
    if capability == (12, 0):
        print("✅ Blackwell GPU properly detected!")
    else:
        print(f"⚠️ Warning: Expected sm_120, got sm_{capability[0]}{capability[1]}")
    
    # Test actual computation
    try:
        x = torch.randn(100, 100).cuda()
        y = x * 2
        print("✅ GPU computation successful")
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print("❌ FATAL: PyTorch doesn't support your GPU architecture!")
            print("   You need a newer PyTorch nightly build")
```

## Common Errors and Solutions

### Error: "no kernel image is available for execution on the device"
**Cause**: PyTorch doesn't have compiled kernels for sm_120
**Solution**: Install PyTorch nightly with cu128 or cu129

### Error: "CUDA error: device-side assert triggered"
**Cause**: Memory or computation error
**Solution**: Reduce batch size, check for NaN values

### Error: "RuntimeError: CUDA out of memory"
**Cause**: Model too large for VRAM
**Solution**: Use smaller model or enable gradient checkpointing

## Docker Solution (Most Reliable)

If native installation fails, use this tested Docker approach:

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch nightly
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify installation
RUN python -c "import torch; assert torch.cuda.is_available()"
```

## pyannote.audio Compatibility

For speaker diarization with pyannote.audio on Blackwell GPUs:

```bash
# After installing PyTorch nightly, install pyannote
pip install pyannote.audio==3.1.1

# Test diarization GPU support
python -c "
from pyannote.audio import Pipeline
import torch
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='YOUR_HF_TOKEN'
)
if torch.cuda.is_available():
    pipeline = pipeline.to(torch.device('cuda'))
    print('✅ Diarization on GPU ready')
"
```

## Environment Variables

Set these for optimal performance:

```bash
# Force CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optimize for Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"

# Disable CUDA lazy loading (faster startup)
export CUDA_MODULE_LOADING=EAGER

# Use TF32 for better performance (Blackwell supports it)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

## Performance Tips

1. **Use float16 precision**: Blackwell has excellent FP16 performance
   ```python
   model = model.half().cuda()
   ```

2. **Enable CUDA graphs** for repetitive operations:
   ```python
   with torch.cuda.graph():
       output = model(input)
   ```

3. **Monitor memory usage**:
   ```python
   print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
   ```

## Current Status (2025)

- **Stable PyTorch**: NO Blackwell support (up to sm_90 only)
- **PyTorch Nightly**: YES, with cu128/cu129 index
- **Expected Stable Support**: Unknown, likely 2025 Q2-Q3
- **Alternative**: NVIDIA NGC containers often have early support

## References

- [PyTorch GitHub Issue #159207](https://github.com/pytorch/pytorch/issues/159207) - Official sm_120 support tracking
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015) - Community solutions
- [Working Docker Container](https://github.com/dconsorte/pytorch-tensorflow-gpu) - Tested Blackwell support