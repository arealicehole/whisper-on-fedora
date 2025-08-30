name: "Fix Blackwell GPU (RTX 5060 Ti) PyAnnote Diarization - PyTorch Nightly Solution"
description: |
  Replace Docker NGC approach with Python venv using PyTorch nightly builds
  to fix "operator torchvision::nms does not exist" error

---

## Goal

**Feature Goal**: Enable PyAnnote diarization to work on Blackwell GPU (RTX 5060 Ti, sm_120) without CPU fallback

**Deliverable**: Working Python 3.11 venv with PyTorch nightly builds that supports both Whisper transcription and PyAnnote diarization on GPU

**Success Definition**: 
- Whisper transcription works on GPU
- PyAnnote diarization works on GPU (no torchvision::nms errors)
- No CPU fallback used
- Service runs successfully at http://127.0.0.1:8765

## User Persona

**Target User**: Developer running Whisper API on Fedora 42 with RTX 5060 Ti

**Use Case**: Real-time audio transcription with speaker diarization for production service

**User Journey**: 
1. Run setup script to create venv
2. Activate venv and start service
3. Send audio for transcription with diarization
4. Receive properly diarized results

**Pain Points Addressed**: 
- NGC containers lack proper torchvision for Blackwell
- "operator torchvision::nms does not exist" errors
- Previous Docker approach doesn't support diarization

## Why

- Blackwell GPU (sm_120) needs PyTorch nightly builds with proper torchvision support
- NGC containers have incompatible torchvision builds for PyAnnote
- Previous documentation shows this worked with pip-installed PyTorch nightly
- User cannot use CPU fallback - full GPU acceleration required

## What

Replace the current Docker NGC container approach with a Python 3.11 venv using PyTorch nightly builds that properly support Blackwell architecture and include compatible torchvision for PyAnnote diarization.

### Success Criteria

- [ ] Python 3.11 venv created and activated
- [ ] PyTorch nightly with cu128/cu129 installed from pip
- [ ] Torchvision nightly installed and working
- [ ] PyAnnote diarization pipeline loads without errors
- [ ] GPU validator confirms Blackwell GPU detected
- [ ] Service starts on port 8765
- [ ] Test transcription with diarization completes successfully

## All Needed Context

### Context Completeness Check

_Someone with no knowledge of this codebase would have everything needed to fix the Blackwell GPU compatibility issue._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- file: /home/ice/whisper-api/docs-archive/troubleshooting/PyTorch Ada Lovelace GPU Workaround.md
  why: Documents the actual solution - PyTorch NIGHTLY builds from pip
  critical: Line 96 shows exact pip command with --pre flag and cu128 index
  pattern: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

- file: /home/ice/whisper-api/docs-archive/project-docs/ice_whisper_guide.md
  why: Shows working configuration from August 2025
  critical: Uses Python 3.12 venv at ~/.venvs/whisper312, confirms setup worked before
  pattern: Lines 52-68 show venv creation and pip install commands

- file: /home/ice/whisper-api/setup_venv.sh
  why: Existing venv setup pattern to follow
  pattern: Python 3.11 venv creation, HF token location
  gotcha: Uses PyTorch 2.2.0 which doesn't support Blackwell

- file: /home/ice/whisper-api/gpu_validator.py
  why: GPU validation utility that enforces GPU-only operation
  pattern: Check for Blackwell (compute capability 12.0) at line 120
  critical: Must pass validation before service starts

- url: https://github.com/pytorch/pytorch/issues/122094
  why: Confirms sm_120 support in PyTorch nightly builds
  critical: Must use nightly builds, stable releases don't support Blackwell

- url: https://discuss.pytorch.org/t/pytorch-for-cuda-12-8-12-9/169447
  why: Explains CUDA 12.8+ requirement for Blackwell
  critical: Must use cu128 or cu129 index URL
```

### Current Codebase tree

```bash
/home/ice/whisper-api/
├── docker/
│   ├── Dockerfile.blackwell     # NGC approach that fails
│   ├── docker-compose.yml       # Current Docker setup
│   └── startup.py               # Docker startup script
├── main.py                      # FastAPI service
├── pyannote_fix.py             # Failed torchvision patch attempt
├── sitecustomize.py            # Failed pre-import patch
├── torchvision_fix.py          # Failed NMS fallback
├── gpu_validator.py            # GPU enforcement utility
├── setup_venv.sh               # Old venv setup (PyTorch 2.2.0)
└── venv/                       # Existing venv with wrong PyTorch
```

### Desired Codebase tree with files to be added

```bash
/home/ice/whisper-api/
├── setup_blackwell_venv.sh     # NEW: Setup script for Blackwell
├── validate_blackwell.py       # NEW: Validation script
├── test_diarization.py         # NEW: Test diarization works
├── requirements_blackwell.txt  # NEW: Dependencies for Blackwell
├── .venvs/
│   └── whisper-blackwell/      # NEW: Python 3.11 venv with nightly
├── main.py                      # EXISTING: No changes needed
├── gpu_validator.py            # EXISTING: Already validates sm_120
└── README_BLACKWELL.md         # NEW: Documentation of solution
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: PyTorch nightly installation order matters
# 1. Must use --pre flag to get nightly builds
# 2. Must specify cu128 or cu129 index URL
# 3. Must install torch, torchvision, torchaudio together
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# CRITICAL: PyAnnote requires torchvision for NMS operations
# NGC containers have incompatible torchvision builds
# Only pip-installed nightly builds work properly

# CRITICAL: Python version matters
# Use Python 3.11 (known to work with PyAnnote 3.x)
# Python 3.12 also works but 3.11 is more tested

# CRITICAL: Don't downgrade PyTorch after installing nightly
# PyAnnote installation may try to downgrade - use --no-deps flag
```

## Implementation Blueprint

### Data models and structure

No new data models needed - using existing FastAPI models in main.py

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE setup_blackwell_venv.sh
  - IMPLEMENT: Bash script to create Python 3.11 venv with PyTorch nightly
  - FOLLOW pattern: setup_venv.sh structure and error handling
  - CRITICAL: Use --pre flag and cu128 index URL for nightly builds
  - PLACEMENT: Root directory alongside existing setup_venv.sh

Task 2: CREATE requirements_blackwell.txt
  - IMPLEMENT: Pinned dependencies for reproducible builds
  - INCLUDE: Core packages without PyTorch (installed separately)
  - PATTERN: Standard requirements.txt format
  - PLACEMENT: Root directory

Task 3: CREATE validate_blackwell.py
  - IMPLEMENT: Validation script to test GPU and diarization
  - FOLLOW pattern: gpu_validator.py validation approach
  - ADD: PyAnnote pipeline loading test
  - ADD: Torchvision NMS operator test
  - PLACEMENT: Root directory

Task 4: CREATE test_diarization.py
  - IMPLEMENT: End-to-end test of transcription with diarization
  - USE: Sample audio file for testing
  - VERIFY: Both Whisper and PyAnnote work on GPU
  - OUTPUT: Clear pass/fail with diagnostics
  - PLACEMENT: Root directory

Task 5: MODIFY .gitignore
  - ADD: .venvs/ directory to ignore list
  - PRESERVE: Existing ignore patterns
  - PLACEMENT: Root directory

Task 6: CREATE README_BLACKWELL.md
  - DOCUMENT: Complete solution and why it works
  - INCLUDE: Troubleshooting steps
  - REFERENCE: PyTorch nightly requirement
  - PLACEMENT: Root directory
```

### Implementation Patterns & Key Details

```bash
# setup_blackwell_venv.sh pattern
#!/bin/bash
set -e

# Check Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 required"
    exit 1
fi

# Create venv
VENV_DIR="$HOME/.venvs/whisper-blackwell"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# CRITICAL: Install PyTorch nightly first
pip install --upgrade pip
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install -r requirements_blackwell.txt

# Install PyAnnote without downgrading PyTorch
pip install --no-deps pyannote.audio==3.3.1
pip install asteroid-filterbanks einops hbreader ... # dependencies
```

```python
# validate_blackwell.py pattern
import torch
from pyannote.audio import Pipeline

def validate_blackwell():
    # Check GPU
    assert torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(0)
    assert capability == (12, 0), f"Not Blackwell: {capability}"
    
    # Check torchvision NMS
    import torchvision
    boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32).cuda()
    scores = torch.tensor([0.9], dtype=torch.float32).cuda()
    iou_threshold = 0.5
    # This will fail with NGC container but work with nightly
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    
    # Test PyAnnote pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN")
    )
    pipeline.to(torch.device("cuda"))
    
    print("✅ All Blackwell validation passed!")
```

### Integration Points

```yaml
TOKEN:
  - location: ~/.config/whisper/token
  - pattern: "HF_TOKEN=hf_xxxx"
  - used by: main.py for PyAnnote authentication

SERVICE:
  - command: "python main.py"
  - port: 8765
  - no changes needed to main.py

REMOVAL:
  - docker/ directory can be archived after venv works
  - Remove pyannote_fix.py, torchvision_fix.py, sitecustomize.py
```

## Validation Loop

### Level 1: Environment Setup

```bash
# Create and activate venv
./setup_blackwell_venv.sh
source ~/.venvs/whisper-blackwell/bin/activate

# Verify PyTorch version (should show dev/nightly date)
python -c "import torch; print(torch.__version__)"
# Expected: 2.6.0.dev20250829+cu128 or similar

# Check CUDA and GPU
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True

python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Expected: (12, 0)
```

### Level 2: Component Validation

```bash
# Run GPU validator
python gpu_validator.py
# Expected: ✅ GPU Validation PASSED

# Run Blackwell-specific validation
python validate_blackwell.py
# Expected: ✅ All Blackwell validation passed!

# Test imports
python -c "from pyannote.audio import Pipeline; print('PyAnnote OK')"
python -c "import faster_whisper; print('Whisper OK')"
python -c "import torchvision.ops; print('Torchvision OK')"
```

### Level 3: Service Validation

```bash
# Start service
python main.py &
sleep 5

# Health check
curl http://127.0.0.1:8765/health
# Expected: {"status":"healthy","gpu_available":true}

# Test transcription
curl -F "file=@sample.wav" http://127.0.0.1:8765/v1/transcribe
# Expected: JSON with transcription

# Test diarization
curl -F "file=@sample.wav" -F "diarize=true" \
     http://127.0.0.1:8765/v1/transcribe
# Expected: JSON with speaker labels
```

### Level 4: End-to-End Validation

```bash
# Run comprehensive test
python test_diarization.py
# Expected: 
# ✅ Whisper transcription on GPU: PASS
# ✅ PyAnnote diarization on GPU: PASS
# ✅ No CPU fallback detected
# ✅ All tests passed!

# Monitor GPU usage during test
nvidia-smi -l 1
# Expected: GPU utilization during processing
```

## Final Validation Checklist

### Technical Validation

- [ ] PyTorch nightly installed with cu128
- [ ] Torchvision NMS operator works
- [ ] PyAnnote pipeline loads on GPU
- [ ] No "operator torchvision::nms does not exist" errors
- [ ] GPU validator passes for Blackwell

### Feature Validation

- [ ] Whisper transcription works on GPU
- [ ] Diarization produces speaker labels
- [ ] Service responds at port 8765
- [ ] No CPU fallback warnings in logs
- [ ] Performance meets expectations (~2-3x realtime)

### Code Quality Validation

- [ ] Setup script is idempotent
- [ ] Clear error messages if setup fails
- [ ] Validation scripts provide actionable feedback
- [ ] Documentation explains why NGC doesn't work

### Documentation & Deployment

- [ ] README_BLACKWELL.md explains the solution
- [ ] Setup script has clear instructions
- [ ] Token configuration documented
- [ ] Rollback plan if needed (keep old venv)

---

## Anti-Patterns to Avoid

- ❌ Don't use NGC containers - torchvision incompatible
- ❌ Don't install stable PyTorch - no sm_120 support  
- ❌ Don't let PyAnnote downgrade PyTorch
- ❌ Don't skip the --pre flag for nightly builds
- ❌ Don't use CPU fallback - must fail if GPU unavailable
- ❌ Don't use Docker for this - venv is simpler and works