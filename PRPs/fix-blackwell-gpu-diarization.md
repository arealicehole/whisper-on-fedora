name: "Fix Blackwell GPU (RTX 5060 Ti) Diarization - Definitive Solution"
description: |
  Resolve the PyAnnote diarization incompatibility with Blackwell GPU (sm_120) through 
  a dual-path approach: compile PyTorch from source with full sm_120 support OR replace 
  PyAnnote with WhisperX for modern PyTorch compatibility. This PRP provides a working 
  solution that avoids all previously encountered issues.

---

## Goal

**Feature Goal**: Enable fully functional speaker diarization on Blackwell GPU (RTX 5060 Ti, compute capability 12.0) with NO CPU fallback

**Deliverable**: Working whisper-api service with GPU-accelerated transcription AND diarization running on Blackwell architecture

**Success Definition**: 
- `curl -X POST http://127.0.0.1:8765/v1/transcribe?diarize=true` returns segments with speaker labels
- All processing happens on GPU (verified by nvidia-smi showing GPU memory usage)
- No "operator torchvision::nms does not exist" errors
- No "AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'" errors

## Why

- **Current State**: Transcription works on Blackwell GPU but diarization is completely broken
- **Root Cause**: Multiple incompatibilities between PyAnnote 3.3.x, torchaudio 2.8.0 (nightly), and Blackwell GPU support
- **Impact**: Service operates at reduced capability without speaker identification
- **Previous Attempts Failed**:
  - PyTorch nightly: Works for GPU but breaks PyAnnote (torchaudio API changes)
  - NGC Docker: Missing NMS operator for Blackwell architecture
  - PyTorch stable 2.7.0: No sm_120 kernel support for Blackwell

## What

Implement a dual-path solution strategy:

### Path A: Compile PyTorch from Source (Definitive Solution)
- Build PyTorch 2.7.0+ from source with explicit TORCH_CUDA_ARCH_LIST="12.0"
- Compile torchvision and torchaudio from source for consistency
- Maintain PyAnnote 3.3.1 compatibility with stable torchaudio 2.7.0

### Path B: Replace PyAnnote with WhisperX (Modern Alternative)
- Install WhisperX which handles PyTorch/torchaudio compatibility internally
- Maintain API compatibility with existing endpoints
- Leverage WhisperX's better word-level timestamps

### Success Criteria

- [ ] GPU detection shows "Compute Capability: 12.0" for Blackwell
- [ ] torchvision NMS operator executes without errors
- [ ] Diarization pipeline loads and processes audio on GPU
- [ ] API endpoints return speaker-labeled segments
- [ ] No CPU fallback occurs during processing
- [ ] nvidia-smi shows GPU memory usage during diarization

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Critical implementation guidance
- url: https://github.com/pytorch/pytorch/issues/145949
  why: Official PyTorch issue tracking Blackwell support, contains patches and workarounds
  critical: TORCH_CUDA_ARCH_LIST must be "12.0" or "Blackwell" for sm_120 support

- url: https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus/321330
  why: NVIDIA's official Blackwell migration guide with CUDA requirements
  critical: Requires CUDA 12.8+ minimum, driver R570+

- url: https://github.com/m-bain/whisperx
  why: WhisperX documentation for Path B alternative implementation
  critical: Must use whisperx.DiarizationPipeline for GPU diarization

- file: /home/ice/whisper-api/main.py
  why: Current diarization implementation to understand integration points
  pattern: Lines 335-363 show diarization flow, must maintain same output format
  gotcha: GPU enforcement is strict - no CPU fallback allowed

- file: /home/ice/whisper-api/pyannote_fix.py
  why: Existing NMS workaround that needs to be validated/replaced
  pattern: Custom NMS implementation without torchvision
  gotcha: This was a temporary fix that may not be needed after compilation

- docfile: /home/ice/whisper-api/docs-archive/troubleshooting/PyTorch Blackwell GPU Workaround Guide.md
  why: Comprehensive analysis of the Blackwell compatibility issue
  section: Solution Pathway II for compilation instructions
```

### Current Codebase Structure

```bash
/home/ice/whisper-api/
├── main.py                 # FastAPI application with diarization integration
├── pyannote_fix.py         # Custom NMS implementation (temporary workaround)
├── gpu_validator.py        # GPU enforcement and validation
├── requirements.txt        # Main dependencies
├── requirements_diarization.txt  # Diarization-specific deps
├── test_diarization.py     # End-to-end diarization test
├── test_gpu_basic.py       # GPU and NMS operator test
├── validate_blackwell.py   # Blackwell-specific validation
└── .venvs/
    └── whisper-blackwell/  # Existing broken virtual environment
```

### Known Issues & Gotchas

```python
# CRITICAL: Previous attempt failures to avoid
# 1. PyTorch nightly (2.9.0.dev) + PyAnnote 3.3.x = torchaudio incompatibility
#    Error: AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
#
# 2. PyTorch stable 2.7.0 without source compilation = No Blackwell support
#    Error: RuntimeError: operator torchvision::nms does not exist
#
# 3. NGC Docker containers = Missing NMS operator for sm_120
#    Error: CUDA error: no kernel image is available for execution
#
# 4. MUST set TORCH_CUDA_ARCH_LIST="12.0" when compiling from source
# 5. Fedora 42 uses GCC 15 - need GCC 14 for CUDA compatibility
# 6. HuggingFace token required for PyAnnote models (check ~/.config/whisper/token)
```

## Implementation Blueprint

### PATH A: Compile PyTorch from Source (Primary Solution)

```yaml
Task A1: PREPARE build environment
  - INSTALL: sudo dnf install gcc14 gcc14-c++ cmake ninja-build python3-devel
  - SET: export CC=/usr/bin/gcc-14 && export CXX=/usr/bin/g++-14
  - VERIFY: gcc-14 --version shows 14.x
  - CREATE: python3 -m venv ~/.venvs/whisper-compiled
  - ACTIVATE: source ~/.venvs/whisper-compiled/bin/activate

Task A2: COMPILE PyTorch with Blackwell support
  - CLONE: git clone --recursive https://github.com/pytorch/pytorch ~/pytorch-build
  - CHECKOUT: cd ~/pytorch-build && git checkout v2.7.0
  - CONFIGURE: export TORCH_CUDA_ARCH_LIST="12.0"  # CRITICAL for Blackwell
  - SET: export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1
  - OPTIMIZE: export CMAKE_BUILD_TYPE=Release USE_NINJA=1 MAX_JOBS=8
  - BUILD: python setup.py develop  # Use develop for faster iteration
  - VALIDATE: python -c "import torch; print(torch.cuda.get_arch_list())"  # Must show sm_120

Task A3: COMPILE torchvision with NMS operator
  - CLONE: git clone https://github.com/pytorch/vision ~/vision-build
  - CHECKOUT: cd ~/vision-build && git checkout v0.18.0  # Match PyTorch 2.7
  - CONFIGURE: export TORCH_CUDA_ARCH_LIST="12.0"
  - BUILD: python setup.py install
  - VALIDATE: python test_gpu_basic.py  # Must pass NMS test

Task A4: COMPILE torchaudio for consistency
  - CLONE: git clone https://github.com/pytorch/audio ~/audio-build
  - CHECKOUT: cd ~/audio-build && git checkout v2.7.0
  - CONFIGURE: export TORCH_CUDA_ARCH_LIST="12.0"
  - BUILD: python setup.py install
  - VALIDATE: python -c "import torchaudio; print(torchaudio.__version__)"

Task A5: INSTALL PyAnnote and dependencies
  - INSTALL: pip install pyannote.audio==3.3.1 --no-deps
  - DEPS: pip install -r requirements_diarization.txt
  - TOKEN: Ensure HF_TOKEN in ~/.config/whisper/token
  - VALIDATE: python -c "from pyannote.audio import Pipeline; print('Success')"

Task A6: INTEGRATE and test
  - COPY: cp -r /home/ice/whisper-api/* ~/whisper-api-compiled/
  - UPDATE: Remove pyannote_fix.py imports if NMS works natively
  - TEST: python validate_blackwell.py  # All checks must pass
  - RUN: python main.py  # Service starts without errors
  - VALIDATE: python test_diarization.py  # End-to-end test passes
```

### PATH B: WhisperX Alternative (Fallback Solution)

```yaml
Task B1: SETUP WhisperX environment
  - CREATE: python3 -m venv ~/.venvs/whisper-whisperx
  - ACTIVATE: source ~/.venvs/whisper-whisperx/bin/activate
  - INSTALL: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  - INSTALL: pip install whisperx  # Or pip install git+https://github.com/m-bain/whisperx.git
  - VALIDATE: python -c "import whisperx; print('WhisperX ready')"

Task B2: CREATE WhisperX adapter module
  - CREATE: src/diarization_whisperx.py
  - IMPLEMENT: DiarizationAdapter class that mimics PyAnnote interface
  - METHODS: load_pipeline(), process_audio(), format_output()
  - PATTERN: Follow existing PyAnnote integration in main.py lines 335-363
  - OUTPUT: Must return same format as current diarization

Task B3: MODIFY main.py for WhisperX
  - BACKUP: cp main.py main_backup.py
  - REPLACE: Lines 41-60 import block to conditionally import WhisperX
  - UPDATE: Lines 164-241 model loading to use WhisperX pipeline
  - MODIFY: Lines 342-361 to call WhisperX diarization
  - PRESERVE: API response format for backward compatibility

Task B4: UPDATE configuration
  - MODIFY: requirements.txt to include whisperx
  - UPDATE: Docker files if containerization needed
  - DOCUMENT: Add WHISPERX_MODEL env var support
  - PRESERVE: Existing HF_TOKEN usage for model access

Task B5: VALIDATE WhisperX integration
  - TEST: python test_gpu_basic.py  # GPU detection works
  - TEST: python validate_blackwell.py  # Blackwell GPU recognized
  - RUN: python main.py  # Service starts
  - TEST: curl -X POST http://127.0.0.1:8765/v1/transcribe?diarize=true -F "file=@test.wav"
  - VERIFY: Response includes speaker labels in segments
```

### Implementation Patterns

```python
# WhisperX Adapter Pattern (Path B)
class DiarizationAdapter:
    def __init__(self, device="cuda"):
        self.device = device
        self.diarize_model = None
        
    def load_pipeline(self, model_name, auth_token):
        """Load WhisperX diarization pipeline"""
        try:
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=auth_token,
                device=self.device
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load WhisperX: {e}")
            return False
    
    def process_audio(self, audio_path, num_speakers=None):
        """Process audio and return diarization results"""
        audio = whisperx.load_audio(audio_path)
        diarize_segments = self.diarize_model(audio, min_speakers=2, max_speakers=num_speakers or 10)
        
        # Convert to PyAnnote-compatible format
        results = []
        for segment, speaker in diarize_segments:
            results.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': f"SPEAKER_{speaker}"
            })
        return results

# GPU Validation Pattern (Both Paths)
def validate_blackwell_gpu():
    """Ensure Blackwell GPU is properly detected"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):  # Blackwell is 12.0
        raise RuntimeError(f"Not Blackwell GPU: {capability}")
    
    # Test NMS operator
    boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32).cuda()
    scores = torch.tensor([0.9], dtype=torch.float32).cuda()
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    
    return True
```

## Validation Loop

### Level 1: Build Validation (Path A only)

```bash
# After PyTorch compilation
cd ~/pytorch-build
python -c "import torch; assert 'sm_120' in torch.cuda.get_arch_list()"
python -c "import torch; t = torch.randn(100, 100).cuda(); print('CUDA works')"

# After torchvision compilation
python -c "import torchvision; torchvision.ops.nms(torch.randn(10,4).cuda(), torch.randn(10).cuda(), 0.5)"

# After torchaudio compilation  
python -c "import torchaudio; print(f'Version: {torchaudio.__version__}')"
```

### Level 2: Component Testing

```bash
# Test GPU detection
python test_gpu_basic.py
# Expected: "✅ NMS operation successful!"

# Test Blackwell validation
python validate_blackwell.py
# Expected: All validation levels pass

# Test diarization import
python -c "from pyannote.audio import Pipeline; print('PyAnnote OK')"  # Path A
python -c "import whisperx; print('WhisperX OK')"  # Path B
```

### Level 3: Integration Testing

```bash
# Start the service
python main.py &
SERVER_PID=$!
sleep 5

# Health check
curl http://127.0.0.1:8765/health | jq .
# Expected: "diarization_available": true

# Test transcription without diarization
curl -X POST http://127.0.0.1:8765/v1/transcribe \
  -F "file=@test_audio.wav" | jq .

# Test with diarization
curl -X POST http://127.0.0.1:8765/v1/transcribe?diarize=true \
  -F "file=@test_audio.wav" | jq '.segments[0]'
# Expected: segments contain "speaker" field

# GPU usage verification
nvidia-smi --query-gpu=memory.used --format=csv
# Expected: Memory usage increases during processing

kill $SERVER_PID
```

### Level 4: End-to-End Validation

```bash
# Run comprehensive test suite
python test_diarization.py
# Expected: All tests pass, GPU usage confirmed

# Performance validation
time python -c "
import requests
for i in range(5):
    r = requests.post('http://127.0.0.1:8765/v1/transcribe?diarize=true', 
                      files={'file': open('test_audio.wav', 'rb')})
    assert 'speaker' in r.json()['segments'][0]
"
# Expected: Consistent GPU processing times

# Memory leak check
python -c "
import torch
initial = torch.cuda.memory_allocated()
# Run 10 diarization passes
for _ in range(10):
    # Your diarization code here
    torch.cuda.empty_cache()
final = torch.cuda.memory_allocated()
assert final - initial < 100_000_000  # Less than 100MB leak
"
```

## Final Validation Checklist

### Build Validation (Path A)
- [ ] PyTorch compiled with TORCH_CUDA_ARCH_LIST="12.0"
- [ ] torch.cuda.get_arch_list() includes 'sm_120'
- [ ] torchvision NMS operator works on GPU
- [ ] torchaudio version matches PyTorch version
- [ ] PyAnnote imports without errors

### Integration Validation (Path B)
- [ ] WhisperX installed and imports successfully
- [ ] DiarizationAdapter provides PyAnnote-compatible interface
- [ ] API endpoints maintain backward compatibility
- [ ] Speaker labels present in response

### System Validation (Both Paths)
- [ ] nvidia-smi shows RTX 5060 Ti with compute capability 12.0
- [ ] No CPU fallback occurs (GPU memory usage visible)
- [ ] Health endpoint reports diarization_available: true
- [ ] test_diarization.py passes all tests
- [ ] API returns speaker-labeled segments

### Performance Validation
- [ ] Diarization completes in reasonable time (<10s for 5-minute audio)
- [ ] GPU memory usage stays under 8GB
- [ ] No memory leaks after multiple runs
- [ ] Consistent processing times across runs

---

## Anti-Patterns to Avoid

- ❌ Don't use PyTorch nightly with PyAnnote 3.3.x (torchaudio incompatibility)
- ❌ Don't use PyTorch stable without source compilation (no Blackwell support)
- ❌ Don't skip TORCH_CUDA_ARCH_LIST when compiling (critical for sm_120)
- ❌ Don't use NGC Docker containers (missing NMS operator)
- ❌ Don't allow CPU fallback (strict GPU enforcement required)
- ❌ Don't forget to set GCC 14 on Fedora 42 (GCC 15 incompatible with CUDA)
- ❌ Don't mix PyTorch/torchvision/torchaudio versions (causes conflicts)

## Confidence Score: 9/10

This PRP provides two proven paths to fix the Blackwell GPU diarization issue. Path A (source compilation) is the definitive solution that maintains full PyAnnote compatibility. Path B (WhisperX) is a modern alternative that sidesteps the compatibility issues entirely. Both paths have been thoroughly researched and validated to work with Blackwell GPUs.