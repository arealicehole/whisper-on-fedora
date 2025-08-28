# PRP: Fix RTX 5060 Ti Blackwell GPU Support for Whisper API with Full Diarization

## Feature Goal
Enable complete GPU acceleration for both Whisper transcription and speaker diarization on RTX 5060 Ti (Blackwell architecture, sm_120) by implementing proper PyTorch nightly builds with Python 3.11 and cleaning up obsolete code.

## Deliverable
A production-ready Whisper API service that:
- Runs both whisper and diarization on GPU (RTX 5060 Ti)
- Uses Python 3.11 with PyTorch nightly for sm_120 support
- Removes all obsolete workarounds and duplicate code
- Integrates the enhanced diarization handler
- Provides accurate GPU detection and health reporting

## Success Definition
✅ Both whisper and diarization run on GPU
✅ Health endpoint reports actual GPU (not hardcoded)
✅ All obsolete files removed
✅ Clean requirements with Python 3.11
✅ Tests pass with GPU acceleration

---

## Context

### Critical Technical Context

```yaml
hardware:
  gpu: NVIDIA GeForce RTX 5060 Ti
  architecture: Blackwell (NOT Ada Lovelace)
  compute_capability: sm_120 (12.0)
  vram: 16GB
  issue: Not supported by stable PyTorch, requires nightly builds

current_system:
  os: Fedora 42
  current_python: 3.12 (in ~/.venvs/whisper312)
  working_setup_documented: /home/ice/whisper-api/fed/ice_whisper_guide.md
  driver: NVIDIA 580.65.06
  cuda: 13.0 runtime

solution_docs:
  architecture_fix: /home/ice/whisper-api/PyTorch Ada Lovelace GPU Workaround.md
  system_setup: /home/ice/whisper-api/fed/README.md
  
key_files:
  main_app: /home/ice/whisper-api/main.py
  enhanced_handler: /home/ice/whisper-api/diarization_handler.py (17.5KB, unused but good)
  diagnostic: /home/ice/whisper-api/blackwell_diagnostic.py (12.9KB, comprehensive)
```

### Problem Summary

1. **RTX 5060 Ti Blackwell GPU** has compute capability 12.0 (sm_120)
2. **Stable PyTorch doesn't support sm_120** - causes "no kernel image available" errors
3. **Current Python 3.12** has compatibility issues with pyannote
4. **Solution**: Python 3.11 + PyTorch nightly cu128 + proper cleanup

### Working Combinations (Research Confirmed)

```yaml
proven_working:
  - python: 3.11
    pytorch: nightly (--pre)
    cuda_index: cu128
    pyannote: 3.1.1
    status: "✅ Best for Blackwell GPU"
    
  - python: 3.10
    pytorch: nightly
    cuda_index: cu129
    pyannote: 3.0.1
    status: "✅ Alternative option"

external_validation:
  github_issue: https://github.com/pytorch/pytorch/issues/159207
  community_solution: https://github.com/dconsorte/pytorch-tensorflow-gpu
  docker_tested: Ubuntu 24.04 + Python 3.11 + CUDA 12.8
```

### Files to Remove (Cleanup List)

```yaml
dangerous_automation:
  - apply_hybrid_patch.py  # Automated code modification
  
obsolete_backups:
  - main.py.backup
  - main.py.backup_hybrid
  - main.py.fixed
  
redundant_fixes:
  - fix_cuda_fallback.sh     # CPU fallback approach
  - fix_cuda_diarization.sh   # Redundant with main fix
  - fix_cuda_nightly.sh       # Duplicates fix_blackwell_cuda.sh
  - fix_blackwell_pytorch.sh  # Duplicates fix_blackwell_cuda.sh
  - setup_blackwell_env.sh    # Overlaps with other scripts
  - run_docker_blackwell.sh   # Simple wrapper, not needed
  - docker_blackwell_solution.sh  # Overlaps with docker approach
  
keep_for_reference:
  - blackwell_diagnostic.py  # Excellent diagnostic tool
  - test_gpu_whisper.py      # Good testing framework
  - diarization_handler.py   # Enhanced handler to integrate
  - Dockerfile.blackwell     # Docker solution
```

### Code Issues in main.py

```yaml
bugs:
  line_53: Duplicate torch import
  line_379: Hardcoded "NVIDIA GeForce RTX 5060 Ti" instead of detection
  lines_376-380: Fake hybrid mode status (claims CPU for diarization)
  
unused_imports:
  - sys
  - asyncio  
  - json
  - Field from pydantic
  - numpy as np
  
vad_settings:
  current: vad_filter=False (good, was causing empty transcriptions)
  no_speech_threshold: 0.6 (good)
  initial_prompt: Set (good)
```

---

## Implementation Tasks

### 1. Environment Setup - Create Python 3.11 Environment

```bash
# Create new Python 3.11 environment
/usr/bin/python3.11 -m venv ~/.venvs/whisper-blackwell
source ~/.venvs/whisper-blackwell/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch nightly with sm_120 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Capability: {torch.cuda.get_device_capability(0)}')"
# Should show: Capability: (12, 0)
```

### 2. Install Core Dependencies

```bash
# Core Whisper stack (GPU accelerated)
pip install "faster-whisper==1.2.0" "ctranslate2==4.6.0" "whisper-ctranslate2==0.5.4"
pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"

# Diarization (with GPU support)
pip install "pyannote.audio==3.1.1"
pip install "speechbrain==1.0.0" "transformers==4.44.0"

# API dependencies
pip install "fastapi==0.116.1" "uvicorn[standard]==0.35.0" "httpx==0.28.1" "python-multipart==0.0.20" "pydantic"
```

### 3. Fix main.py Issues

**Remove duplicate import and unused imports:**
```python
# Remove line 53 (duplicate torch import)
# Remove unused imports: sys, asyncio, json, numpy, Field
```

**Fix GPU detection (replace hardcoded values):**
```python
# Around line 376-380, replace with:
if torch.cuda.is_available():
    gpu_info = {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
    }
    # Check if diarization is on GPU
    diarization_device = "GPU" if diarization_pipeline and hasattr(diarization_pipeline, 'device') and str(diarization_pipeline.device) != 'cpu' else "CPU"
else:
    gpu_info = {"available": False}
    diarization_device = "CPU"

health_status["gpu"] = gpu_info
health_status["diarization_device"] = diarization_device
```

### 4. Integrate Enhanced Diarization Handler

**Replace inline diarization loading with:**
```python
# Import the enhanced handler
from diarization_handler import DiarizationHandler

# Initialize with proper error handling
diarization_handler = None
if WHISPER_DIARIZE and HF_TOKEN:
    try:
        diarization_handler = DiarizationHandler(
            auth_token=HF_TOKEN,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_auth_token=True
        )
        print(f"✓ Diarization handler initialized on {diarization_handler.device}")
    except Exception as e:
        print(f"Warning: Could not initialize diarization: {e}")
```

### 5. Update Requirements Files

**Create requirements_blackwell.txt:**
```
# Core dependencies with specific versions for Blackwell GPU
--pre
--index-url https://download.pytorch.org/whl/nightly/cu128
torch
torchaudio

# Whisper stack
faster-whisper==1.2.0
ctranslate2==4.6.0
whisper-ctranslate2==0.5.4
nvidia-cublas-cu12
nvidia-cudnn-cu12==9.*

# Diarization
pyannote.audio==3.1.1
speechbrain==1.0.0
transformers==4.44.0

# API
fastapi==0.116.1
uvicorn[standard]==0.35.0
httpx==0.28.1
python-multipart==0.0.20
pydantic

# Audio processing
soundfile==0.12.1
librosa==0.10.2
ffmpeg-python
```

### 6. Clean Up Obsolete Files

```bash
# Remove dangerous/obsolete files
rm apply_hybrid_patch.py
rm main.py.backup main.py.backup_hybrid main.py.fixed
rm fix_cuda_fallback.sh fix_cuda_diarization.sh fix_cuda_nightly.sh
rm fix_blackwell_pytorch.sh setup_blackwell_env.sh
rm run_docker_blackwell.sh docker_blackwell_solution.sh

# Keep these for reference/testing
# blackwell_diagnostic.py - diagnostic tool
# test_gpu_whisper.py - testing
# diarization_handler.py - now integrated
# Dockerfile.blackwell - docker solution
```

### 7. Update Service Configuration

**Update systemd service environment:**
```bash
# Edit ~/.config/systemd/user/whisper-api.service
[Service]
Environment="PATH=/home/ice/.venvs/whisper-blackwell/bin:/usr/local/bin:/usr/bin:/bin"
Environment="WHISPER_MODEL=medium"
Environment="WHISPER_COMPUTE=float16"
Environment="WHISPER_DEVICE=cuda"
Environment="WHISPER_LANGUAGE=en"
Environment="WHISPER_DIARIZE=true"
ExecStart=/home/ice/.venvs/whisper-blackwell/bin/python /home/ice/whisper-api/main.py
```

### 8. Create Comprehensive Test Script

**Create test_blackwell_gpu.py:**
```python
#!/usr/bin/env python3
"""Test RTX 5060 Ti Blackwell GPU support"""

import torch
import sys

def test_gpu():
    print("=" * 60)
    print("RTX 5060 Ti Blackwell GPU Test")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    # Get GPU details
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    
    print(f"GPU Device: {device_name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    
    # Check for Blackwell support
    if capability == (12, 0):
        print("✅ Blackwell architecture detected (sm_120)")
    else:
        print(f"⚠️ Expected sm_120, got sm_{capability[0]}{capability[1]}")
    
    # Test GPU operations
    try:
        print("\nTesting GPU operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✅ GPU computation successful")
        print(f"   Result shape: {z.shape}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
```

---

## Validation Gates

### GPU Support Validation
```bash
# Test GPU detection
python test_blackwell_gpu.py
# Expected: "✅ Blackwell architecture detected (sm_120)"
```

### API Health Check
```bash
# Start service
systemctl --user restart whisper-api.service

# Check health
curl http://127.0.0.1:8765/health | jq .
# Should show actual GPU name, not hardcoded value
```

### Transcription Test
```bash
# Test with sample audio
curl -X POST http://127.0.0.1:8765/v1/transcribe \
  -F "file=@test.wav" \
  -F "diarize=true" | jq .
# Should complete using GPU for both whisper and diarization
```

### Performance Verification
```bash
# Monitor GPU usage during transcription
nvidia-smi dmon -s mu -c 10
# Should show GPU utilization during processing
```

---

## Final Validation Checklist

- [ ] Python 3.11 environment created
- [ ] PyTorch nightly installed with cu128
- [ ] GPU test shows sm_120 capability
- [ ] Obsolete files removed
- [ ] main.py cleaned (no duplicate imports, proper GPU detection)
- [ ] Diarization handler integrated
- [ ] Health endpoint shows real GPU info
- [ ] Both whisper and diarization use GPU
- [ ] Service starts without errors
- [ ] Transcription works with diarization enabled

## Expected Outcome

After implementation:
1. **Full GPU acceleration** for both whisper and diarization on RTX 5060 Ti
2. **Clean codebase** without obsolete workarounds
3. **Proper architecture detection** (no hardcoded GPU names)
4. **Python 3.11 environment** optimized for Blackwell
5. **Production-ready service** with accurate health reporting

## Notes

- The solution uses PyTorch nightly builds which may change. Pin specific versions after testing.
- Monitor PyTorch GitHub for official sm_120 support in stable releases
- Consider Docker deployment (Dockerfile.blackwell) for production isolation
- The diarization_handler.py provides excellent error recovery and should be fully utilized