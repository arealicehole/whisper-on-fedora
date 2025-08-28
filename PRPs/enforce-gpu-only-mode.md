name: "Enforce GPU-Only Mode - Remove CPU Fallback"
description: |
  Enforce strict GPU-only operation for the Whisper API service by removing all CPU fallback mechanisms.
  The RTX 5060 Ti Blackwell GPU has been successfully working before, and CPU performance is unacceptable.
  This PRP ensures the service fails fast with clear error messages when GPU is unavailable, 
  rather than degrading to poor CPU performance.

---

## Goal

**Feature Goal**: Transform the Whisper API service to enforce strict GPU-only operation by removing all CPU fallback mechanisms and implementing comprehensive GPU requirement validation.

**Deliverable**: Modified Whisper API service that requires GPU acceleration, refuses to run on CPU, and provides clear error messages when GPU requirements are not met.

**Success Definition**: Service starts successfully with GPU acceleration OR fails immediately with informative error messages if GPU is unavailable. No CPU fallback occurs under any circumstance.

## User Persona (if applicable)

**Target User**: System administrators and developers deploying Whisper API service

**Use Case**: Production deployment on GPU-equipped servers where high-performance transcription is required

**User Journey**: 
1. Start Whisper API service
2. Service validates GPU availability and capabilities
3. If GPU available: Service starts with full acceleration
4. If GPU unavailable: Service exits with clear error message and remediation steps

**Pain Points Addressed**: 
- Eliminates silent degradation to poor CPU performance
- Prevents resource wastage from inefficient CPU transcription
- Ensures predictable performance characteristics

## Why

- CPU transcription performance is "terrible" - approximately 10x slower than GPU
- The RTX 5060 Ti Blackwell GPU has been working successfully in the past
- Silent fallback to CPU masks configuration issues and creates poor user experience
- Production environments require predictable, high-performance operation
- GPU enforcement ensures optimal resource utilization

## What

Transform the Whisper API service to strictly enforce GPU-only operation:
- Remove all CPU fallback logic from main.py and related files
- Implement comprehensive GPU validation at startup
- Add clear error messages with remediation steps
- Ensure both Whisper and diarization require GPU
- Create validation utilities for GPU requirement checking

### Success Criteria

- [ ] Service refuses to start without available CUDA GPU
- [ ] All CPU fallback code paths are removed
- [ ] Clear error messages explain GPU requirements and how to fix issues
- [ ] GPU validation occurs before model loading to fail fast
- [ ] Both transcription and diarization enforce GPU usage
- [ ] Health endpoint accurately reports GPU-only requirement status

## All Needed Context

### Context Completeness Check

_This PRP provides complete context for implementing GPU enforcement without prior codebase knowledge._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://docs.pytorch.org/docs/stable/notes/cuda.html#checking-cuda-availability
  why: PyTorch CUDA availability checking patterns
  critical: Use torch.cuda.is_available() for validation

- url: https://docs.pytorch.org/docs/stable/cuda_environment_variables.html
  why: CUDA environment variable configuration
  critical: CUDA_VISIBLE_DEVICES controls GPU visibility

- url: https://github.com/SYSTRAN/faster-whisper#gpu-execution
  why: Faster-whisper GPU configuration requirements
  critical: device="cuda" parameter enforces GPU usage

- file: /home/ice/whisper-api/main.py
  why: Primary file with CPU fallback logic to be removed
  pattern: Lines 54-66 contain fallback logic, lines 67-72 have compute type adjustment
  gotcha: Multiple import statements for torch at top

- file: /home/ice/whisper-api/diarization_handler.py
  why: Contains sophisticated GPU fallback mechanisms
  pattern: Lines 147-149, 181-184 have fallback logic
  gotcha: Complex error handling with multiple retry strategies

- file: /home/ice/whisper-api/BLACKWELL_CUDA_IMPLEMENTATION.md
  why: Documents successful GPU-only Docker implementation
  pattern: Shows working GPU enforcement without fallback
  gotcha: Confirms RTX 5060 Ti worked with NGC container

- docfile: /home/ice/whisper-api/PRPs/ai_docs/gpu-enforcement-patterns.md
  why: GPU enforcement patterns and best practices
  section: Code patterns for strict GPU validation
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase

```bash
/home/ice/whisper-api/
├── main.py                      # Primary service file with fallback logic
├── diarization_handler.py       # Diarization with GPU fallback
├── whisper_config_override.py   # Hybrid mode configuration
├── test_transcribe.py           # Test script with hardcoded CUDA
├── test_blackwell_gpu.py        # GPU testing utility
├── test_cuda_init.py            # CUDA initialization test
├── test_gpu_whisper.py          # GPU whisper test
├── test_nvidia_ml.py            # NVML API test
├── blackwell_diagnostic.py      # Comprehensive GPU diagnostic
├── start_whisper.sh             # Service startup script
├── start_whisper_blackwell.sh   # Blackwell-specific startup
├── fix_blackwell_cuda.sh        # GPU fix script
├── fix_cuda_fallback.sh         # Fallback configuration script
├── requirements.txt             # Python dependencies
├── scripts/
│   └── validate_cuda_solution.py # GPU validation script
├── PRPs/                        # Project documentation
│   └── templates/
└── tests/                       # Test suite
```

### Desired Codebase tree with files to be added and responsibility of file

```bash
/home/ice/whisper-api/
├── main.py                      # Modified: GPU-only enforcement, no fallback
├── diarization_handler.py       # Modified: GPU-only enforcement
├── whisper_config_override.py   # Removed or modified: No hybrid mode
├── gpu_validator.py             # NEW: Comprehensive GPU validation utility
├── PRPs/
│   ├── enforce-gpu-only-mode.md # This PRP document
│   └── ai_docs/
│       └── gpu-enforcement-patterns.md # NEW: GPU enforcement documentation
└── tests/
    └── test_gpu_enforcement.py  # NEW: GPU enforcement validation tests
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: PyTorch must detect CUDA before model initialization
# The RTX 5060 Ti requires PyTorch nightly or NGC container for sm_120 support
# Multiple torch imports at the top of files - ensure consistency

# Faster-whisper requires explicit device="cuda" parameter
# PyAnnote pipeline needs .to(torch.device("cuda")) after loading
# CUDA environment variables must be set before importing torch

# The service previously worked with GPU - evidence in BLACKWELL_CUDA_IMPLEMENTATION.md
# Docker solution with NGC container confirmed working without CPU fallback
```

## Implementation Blueprint

### Data models and structure

Create GPU validation and enforcement structures:

```python
# gpu_validator.py - GPU requirement validation utility
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class GPURequirements:
    """GPU requirements specification"""
    min_memory_gb: float = 4.0
    required_compute_capability: tuple = (7, 5)  # sm_75 minimum
    required_device_count: int = 1
    
@dataclass
class GPUValidationResult:
    """GPU validation result with details"""
    is_valid: bool
    device_name: Optional[str] = None
    memory_gb: Optional[float] = None
    compute_capability: Optional[tuple] = None
    error_message: Optional[str] = None
    remediation_steps: Optional[list] = None

class GPUEnforcementError(Exception):
    """Custom exception for GPU requirement failures"""
    def __init__(self, message: str, remediation: list = None):
        super().__init__(message)
        self.remediation = remediation or []
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE PRPs/ai_docs/gpu-enforcement-patterns.md
  - IMPLEMENT: Documentation file with GPU enforcement patterns from research
  - INCLUDE: PyTorch validation patterns, faster-whisper GPU config, error handling
  - FOLLOW pattern: Markdown documentation format
  - CONTENT: Code examples for GPU validation, enforcement, error messages
  - PLACEMENT: PRPs/ai_docs/ directory for AI reference

Task 2: CREATE gpu_validator.py
  - IMPLEMENT: GPUValidator class with comprehensive validation methods
  - FOLLOW pattern: Object-oriented design with clear separation of concerns
  - INCLUDE: validate_cuda_available(), validate_device_count(), validate_memory()
  - INCLUDE: validate_compute_capability(), enforce_gpu_requirements()
  - ERROR HANDLING: GPUEnforcementError with detailed remediation steps
  - PLACEMENT: Root directory alongside main.py

Task 3: MODIFY main.py - Remove CPU fallback logic
  - REMOVE: Lines 54-66 - CPU fallback on CUDA unavailability
  - REMOVE: Lines 67-72 - Compute type adjustment for CPU
  - REPLACE WITH: GPU enforcement using GPUValidator
  - ADD: Import gpu_validator at top
  - ADD: GPU validation before model initialization (line 94)
  - MODIFY: Health endpoint to report GPU-only status
  - PRESERVE: Existing API endpoints and functionality

Task 4: MODIFY diarization_handler.py - Remove CPU fallback
  - REMOVE: Lines 147-149 - CUDA compatibility check with CPU fallback
  - REMOVE: Lines 181-184 - Dynamic CUDA error recovery to CPU
  - REMOVE: Lines 318-324 - GPU OOM handling with batch reduction
  - REPLACE WITH: GPU enforcement that raises exceptions
  - MODIFY: __init__ to validate GPU before pipeline loading
  - ENSURE: Pipeline always loads on GPU with no fallback

Task 5: REMOVE/MODIFY whisper_config_override.py
  - EVALUATE: If file forces hybrid mode, remove it entirely
  - OR MODIFY: Ensure both WHISPER_DEVICE and DIARIZATION_DEVICE are "cuda"
  - REMOVE: Any CPU device assignments
  - ADD: GPU enforcement configuration

Task 6: MODIFY start_whisper.sh startup script
  - ADD: GPU validation before starting service
  - ADD: Clear error messages if GPU not available
  - INCLUDE: python gpu_validator.py check before starting main.py
  - PRESERVE: Existing environment setup and logging

Task 7: CREATE tests/test_gpu_enforcement.py
  - IMPLEMENT: Unit tests for GPU enforcement
  - TEST: GPUValidator validation methods
  - TEST: Main.py refuses to start without GPU (mock torch.cuda)
  - TEST: Diarization handler enforces GPU
  - TEST: Error messages and remediation steps
  - COVERAGE: All GPU enforcement code paths

Task 8: UPDATE fix_blackwell_cuda.sh
  - REMOVE: CPU fallback installation options
  - ENSURE: Only installs GPU-enabled PyTorch
  - ADD: GPU validation after installation
  - FAIL: Script exits if GPU not working after fixes
```

### Implementation Patterns & Key Details

```python
# gpu_validator.py - Core validation implementation
import sys
import torch
from typing import Optional

class GPUValidator:
    def __init__(self, requirements: Optional[GPURequirements] = None):
        self.requirements = requirements or GPURequirements()
    
    def enforce_gpu_requirements(self) -> GPUValidationResult:
        """Enforce GPU requirements - fail if not met"""
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            remediation = [
                "Install CUDA-enabled PyTorch:",
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "For RTX 5060 Ti Blackwell (sm_120), use:",
                "  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
                "Or use NGC container: nvcr.io/nvidia/pytorch:25.02-py3"
            ]
            raise GPUEnforcementError(
                "GPU is required but CUDA is not available",
                remediation
            )
        
        # Validate device count
        device_count = torch.cuda.device_count()
        if device_count < self.requirements.required_device_count:
            raise GPUEnforcementError(
                f"Insufficient GPUs: {device_count} < {self.requirements.required_device_count} required"
            )
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        memory_gb = device_props.total_memory / (1024**3)
        compute_capability = (device_props.major, device_props.minor)
        
        # Validate memory
        if memory_gb < self.requirements.min_memory_gb:
            raise GPUEnforcementError(
                f"Insufficient GPU memory: {memory_gb:.1f}GB < {self.requirements.min_memory_gb}GB required"
            )
        
        # Return successful validation
        return GPUValidationResult(
            is_valid=True,
            device_name=device_name,
            memory_gb=memory_gb,
            compute_capability=compute_capability
        )

# main.py - Modified startup with GPU enforcement
import os
import sys
from gpu_validator import GPUValidator, GPUEnforcementError

# Enforce GPU at startup
def initialize_service():
    """Initialize service with GPU enforcement"""
    
    # Validate GPU requirements
    validator = GPUValidator()
    try:
        result = validator.enforce_gpu_requirements()
        print(f"✓ GPU validation passed: {result.device_name}")
        print(f"  Memory: {result.memory_gb:.1f}GB")
        print(f"  Compute Capability: sm_{result.compute_capability[0]}{result.compute_capability[1]}")
    except GPUEnforcementError as e:
        print(f"✗ GPU Requirements not met: {e}")
        if e.remediation:
            print("\nRemediation steps:")
            for step in e.remediation:
                print(f"  {step}")
        sys.exit(1)
    
    # Initialize models with GPU only
    global model, WHISPER_DEVICE, WHISPER_COMPUTE
    
    WHISPER_DEVICE = "cuda"  # Force GPU, no fallback
    WHISPER_COMPUTE = "float16"  # GPU-optimized precision
    
    # Load Whisper model on GPU
    print(f"Loading Whisper model on GPU...")
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    
    # Load diarization on GPU if enabled
    if WHISPER_DIARIZE and DIARIZATION_AVAILABLE:
        global diarization_pipeline
        pipeline = load_diarization_pipeline()
        pipeline.to(torch.device("cuda"))
        print(f"✓ Diarization pipeline loaded on GPU")

# Modified health endpoint
@app.get("/health")
async def health_check():
    """Health check with GPU enforcement status"""
    
    # Validate GPU is still available
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "GPU lost - service requires GPU acceleration",
                "gpu_required": True,
                "gpu_available": False
            }
        )
    
    return {
        "status": "healthy",
        "gpu_required": True,
        "gpu_available": True,
        "device": torch.cuda.get_device_name(0),
        "transcription": {
            "model": WHISPER_MODEL,
            "device": "cuda",
            "compute": "float16"
        },
        "diarization": {
            "available": DIARIZATION_AVAILABLE,
            "device": "cuda" if DIARIZATION_AVAILABLE else None
        }
    }
```

### Integration Points

```yaml
ENVIRONMENT:
  - remove: WHISPER_DEVICE environment variable override
  - enforce: No FORCE_CPU_ONLY or similar overrides
  - validate: CUDA_VISIBLE_DEVICES not set to empty

STARTUP:
  - add to: start_whisper.sh
  - pattern: "python gpu_validator.py || exit 1"
  - before: Starting main.py service

ERROR_HANDLING:
  - pattern: Raise GPUEnforcementError instead of falling back
  - logging: Log GPU validation results to whisper.log
  - monitoring: Report GPU-only status in metrics
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
cd /home/ice/whisper-api

# Check Python syntax and style
python -m py_compile gpu_validator.py
python -m py_compile main.py
python -m py_compile diarization_handler.py

# Lint checking if ruff available
which ruff && ruff check gpu_validator.py main.py --fix

# Expected: Zero syntax errors. Fix any issues before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test GPU validator independently
python -c "
from gpu_validator import GPUValidator, GPUEnforcementError
try:
    validator = GPUValidator()
    result = validator.enforce_gpu_requirements()
    print('✓ GPU validation passed:', result.device_name)
except GPUEnforcementError as e:
    print('✗ Expected error for testing:', e)
"

# Test GPU enforcement in main module
python -c "
import sys
sys.path.insert(0, '/home/ice/whisper-api')
# Mock torch if needed for testing
import torch
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('✓ Should refuse to start without GPU')
"

# Run unit tests
python -m pytest tests/test_gpu_enforcement.py -v

# Expected: GPU enforcement works correctly, appropriate errors raised
```

### Level 3: Integration Testing (System Validation)

```bash
# Test service startup with GPU enforcement
cd /home/ice/whisper-api

# Attempt to start service (should succeed with GPU or fail with clear error)
python main.py &
SERVICE_PID=$!
sleep 5

# Check if service started
if ps -p $SERVICE_PID > /dev/null; then
    echo "✓ Service started with GPU"
    
    # Test health endpoint
    curl -s http://localhost:8765/health | python -m json.tool
    
    # Test transcription endpoint
    curl -X POST http://localhost:8765/v1/transcribe \
      -F "file=@test_audio.wav" \
      | python -m json.tool
    
    # Stop service
    kill $SERVICE_PID
else
    echo "✓ Service correctly refused to start without GPU"
fi

# Test with CUDA_VISIBLE_DEVICES="" (should fail)
CUDA_VISIBLE_DEVICES="" python main.py 2>&1 | grep -q "GPU is required"
if [ $? -eq 0 ]; then
    echo "✓ Correctly fails when GPU hidden"
fi

# Expected: Service either runs with GPU or fails with clear GPU requirement error
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Test with Docker and NGC container (if available)
if [ -f "docker-compose.yml" ]; then
    docker-compose up -d
    sleep 10
    
    # Test containerized service
    curl -s http://localhost:8765/health | grep -q "gpu_required.*true"
    if [ $? -eq 0 ]; then
        echo "✓ Docker service enforces GPU"
    fi
    
    docker-compose down
fi

# Test startup script enforcement
./start_whisper.sh start
sleep 5

# Check logs for GPU enforcement
grep -q "GPU validation passed" whisper.log && echo "✓ GPU enforced in startup"
grep -q "GPU Requirements not met" whisper.log && echo "✓ GPU enforcement error logged"

# Performance validation (only if GPU available)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    # Time a transcription with GPU enforcement
    time curl -X POST http://localhost:8765/v1/transcribe \
      -F "file=@test_audio.wav" \
      -o /dev/null -s
    echo "✓ GPU-accelerated transcription completed"
fi

# Simulate GPU loss (advanced test)
# This would require root access to unload nvidia kernel module
# nvidia-smi -pm 0  # Disable persistence mode (requires root)

# Expected: All GPU enforcement validations pass, no CPU fallback occurs
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] GPU validator correctly identifies GPU availability
- [ ] Service refuses to start without GPU
- [ ] No CPU fallback code remains in codebase
- [ ] Clear error messages with remediation steps

### Feature Validation

- [ ] Service starts with GPU or fails with clear error
- [ ] Health endpoint reports gpu_required: true
- [ ] Transcription endpoint uses GPU exclusively
- [ ] Diarization pipeline uses GPU exclusively
- [ ] No performance degradation to CPU occurs

### Code Quality Validation

- [ ] All CPU fallback logic removed from main.py
- [ ] All CPU fallback logic removed from diarization_handler.py
- [ ] GPU enforcement follows established patterns
- [ ] Error messages provide actionable remediation
- [ ] Tests cover GPU enforcement scenarios

### Documentation & Deployment

- [ ] GPU requirements clearly documented
- [ ] Startup scripts validate GPU before service start
- [ ] Logs clearly indicate GPU enforcement status
- [ ] Health monitoring reports GPU-only operation

---

## Anti-Patterns to Avoid

- ❌ Don't add any CPU fallback "just in case"
- ❌ Don't catch GPU errors and continue silently
- ❌ Don't allow environment overrides for CPU mode
- ❌ Don't skip GPU validation before model loading
- ❌ Don't use generic error messages without remediation
- ❌ Don't allow partial GPU usage (transcription on GPU, diarization on CPU)

## Confidence Score

**Implementation Success Likelihood: 9/10**

The high confidence score is based on:
- Clear evidence of previous successful GPU operation (BLACKWELL_CUDA_IMPLEMENTATION.md)
- Comprehensive understanding of all fallback mechanisms to remove
- Well-documented GPU enforcement patterns from research
- Existing validation tools and scripts in codebase
- Straightforward removal of fallback logic with GPU enforcement

The implementation is primarily subtractive (removing fallback) with focused additions (GPU validation), making it low-risk and highly achievable.