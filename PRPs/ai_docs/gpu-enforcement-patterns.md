# GPU Enforcement Patterns for Whisper API

This document provides comprehensive patterns and best practices for enforcing GPU-only operation in the Whisper API service.

## Core GPU Validation Patterns

### 1. Basic GPU Availability Check with Exception

```python
import torch
import sys

def enforce_gpu_availability():
    """Enforce GPU availability - exit if not available"""
    if not torch.cuda.is_available():
        print("ERROR: GPU acceleration is required but CUDA is not available.")
        print("\nPossible causes:")
        print("1. No NVIDIA GPU present in the system")
        print("2. NVIDIA drivers not installed")
        print("3. PyTorch installed without CUDA support")
        print("\nRemediation steps:")
        print("1. Verify GPU presence: nvidia-smi")
        print("2. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\nFor RTX 5060 Ti Blackwell (sm_120) support:")
        print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
        sys.exit(1)
```

### 2. Comprehensive GPU Validation

```python
import torch
from typing import Tuple

def validate_gpu_capabilities() -> Tuple[str, float, tuple]:
    """Validate GPU capabilities and return details"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available - GPU required")
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")
    
    # Get primary GPU properties
    device_props = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)
    memory_gb = device_props.total_memory / (1024**3)
    compute_capability = (device_props.major, device_props.minor)
    
    # Validate minimum requirements
    MIN_MEMORY_GB = 4.0
    if memory_gb < MIN_MEMORY_GB:
        raise RuntimeError(
            f"Insufficient GPU memory: {memory_gb:.1f}GB < {MIN_MEMORY_GB}GB required\n"
            f"Device: {device_name}"
        )
    
    # Check for Blackwell architecture (sm_120)
    if compute_capability == (12, 0):
        print(f"⚠️  Blackwell GPU detected (sm_{compute_capability[0]}{compute_capability[1]})")
        print("   Ensure PyTorch nightly or NGC container is used for full support")
    
    return device_name, memory_gb, compute_capability
```

### 3. Custom Exception for Better Error Handling

```python
class GPURequirementError(Exception):
    """Custom exception for GPU requirement failures"""
    
    def __init__(self, message: str, remediation_steps: list = None):
        super().__init__(message)
        self.remediation_steps = remediation_steps or []
    
    def print_remediation(self):
        """Print remediation steps to console"""
        if self.remediation_steps:
            print("\n📋 Remediation Steps:")
            for i, step in enumerate(self.remediation_steps, 1):
                print(f"   {i}. {step}")

# Usage
def check_gpu_with_custom_error():
    if not torch.cuda.is_available():
        raise GPURequirementError(
            "GPU is required but not available",
            remediation_steps=[
                "Check GPU presence: nvidia-smi",
                "Install CUDA drivers: https://developer.nvidia.com/cuda-downloads",
                "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118",
                "For Docker: Use --gpus all flag or nvidia runtime"
            ]
        )
```

## Faster-Whisper GPU Enforcement

### 1. Strict GPU-Only Model Loading

```python
from faster_whisper import WhisperModel
import torch

def load_whisper_model_gpu_only(model_size: str = "medium") -> WhisperModel:
    """Load Whisper model with strict GPU enforcement"""
    
    # Pre-validate GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cannot load Whisper model: GPU is required but CUDA is not available.\n"
            "This service is configured for GPU-only operation due to performance requirements."
        )
    
    try:
        # Force GPU with no fallback
        model = WhisperModel(
            model_size,
            device="cuda",  # Explicitly require CUDA
            compute_type="float16",  # GPU-optimized precision
            device_index=0,  # Use primary GPU
            num_workers=4,  # Parallel processing
            download_root="/models"  # Custom model cache
        )
        
        print(f"✓ Whisper model '{model_size}' loaded on GPU: {torch.cuda.get_device_name(0)}")
        return model
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                f"GPU out of memory loading Whisper model '{model_size}'.\n"
                f"Try a smaller model or increase GPU memory.\n"
                f"Current GPU: {torch.cuda.get_device_name(0)}"
            )
        elif "no kernel image" in str(e).lower():
            raise RuntimeError(
                f"GPU architecture incompatibility detected.\n"
                f"Your GPU ({torch.cuda.get_device_name(0)}) requires PyTorch with appropriate CUDA support.\n"
                f"For Blackwell GPUs, use PyTorch nightly: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
            )
        else:
            raise RuntimeError(f"Failed to load Whisper model on GPU: {e}")
```

### 2. Transcription with GPU Enforcement

```python
def transcribe_audio_gpu_only(model: WhisperModel, audio_path: str) -> dict:
    """Transcribe audio with GPU enforcement"""
    
    # Verify model is on GPU (defensive check)
    if model.model.device.type != 'cuda':
        raise RuntimeError("Model is not on GPU - GPU-only mode requires CUDA device")
    
    # Transcribe with GPU-optimized settings
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,  # GPU can handle larger beam size
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=False,  # Disable for better accuracy
        word_timestamps=True,
        condition_on_previous_text=True
    )
    
    return {
        "segments": list(segments),
        "language": info.language,
        "duration": info.duration
    }
```

## PyAnnote Diarization GPU Enforcement

### 1. Pipeline Loading with GPU Requirement

```python
from pyannote.audio import Pipeline
import torch

def load_diarization_pipeline_gpu_only(auth_token: str) -> Pipeline:
    """Load diarization pipeline with GPU enforcement"""
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cannot load diarization pipeline: GPU is required.\n"
            "Speaker diarization is computationally intensive and requires GPU acceleration."
        )
    
    try:
        # Load pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        # Force to GPU - no fallback
        device = torch.device("cuda:0")
        pipeline.to(device)
        
        print(f"✓ Diarization pipeline loaded on GPU: {torch.cuda.get_device_name(0)}")
        return pipeline
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                "GPU out of memory loading diarization pipeline.\n"
                "Diarization requires approximately 2GB of VRAM.\n"
                "Free GPU memory or reduce other GPU workloads."
            )
        else:
            raise RuntimeError(f"Failed to load diarization pipeline on GPU: {e}")
```

### 2. Diarization Processing with GPU Validation

```python
def diarize_audio_gpu_only(pipeline: Pipeline, audio_path: str) -> dict:
    """Process speaker diarization with GPU enforcement"""
    
    # Verify pipeline is on GPU
    if not any(param.is_cuda for param in pipeline.parameters()):
        raise RuntimeError("Diarization pipeline is not on GPU")
    
    try:
        # Process with GPU
        with torch.cuda.amp.autocast():  # Automatic mixed precision for efficiency
            diarization = pipeline(audio_path)
        
        # Convert to segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return {"segments": segments}
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()  # Clear cache
        raise RuntimeError(
            "GPU out of memory during diarization.\n"
            "Audio file may be too long or complex.\n"
            "Consider processing in chunks or upgrading GPU."
        )
```

## Service-Level GPU Enforcement

### 1. Application Startup with GPU Validation

```python
import os
import sys
import torch
import logging

class WhisperAPIService:
    def __init__(self, require_gpu: bool = True):
        self.require_gpu = require_gpu
        self.logger = logging.getLogger(__name__)
        
        if self.require_gpu:
            self._enforce_gpu_requirements()
    
    def _enforce_gpu_requirements(self):
        """Enforce GPU requirements at service startup"""
        
        # Check environment override (for debugging only)
        if os.environ.get('WHISPER_ALLOW_CPU', '').lower() == 'true':
            self.logger.warning(
                "⚠️  WHISPER_ALLOW_CPU=true is set (debugging mode).\n"
                "   Production deployments should not use this flag."
            )
            self.require_gpu = False
            return
        
        # Validate CUDA availability
        if not torch.cuda.is_available():
            error_msg = (
                "🚫 GPU REQUIRED: This service is configured for GPU-only operation.\n"
                "\n"
                "Reasons:\n"
                "  • CPU transcription is ~10x slower than GPU\n"
                "  • Production SLAs require GPU performance\n"
                "  • Resource efficiency demands GPU acceleration\n"
                "\n"
                "Current Status:\n"
                f"  • CUDA Available: {torch.cuda.is_available()}\n"
                f"  • PyTorch Version: {torch.__version__}\n"
                f"  • CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}\n"
                "\n"
                "Resolution Options:\n"
                "  1. Ensure NVIDIA GPU is present: nvidia-smi\n"
                "  2. Install CUDA drivers: https://developer.nvidia.com/cuda-downloads\n"
                "  3. Install PyTorch with CUDA:\n"
                "     pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                "  4. For RTX 5060 Ti (Blackwell):\n"
                "     pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128\n"
                "  5. Use Docker with GPU support:\n"
                "     docker run --gpus all -p 8765:8765 whisper-api\n"
                "  6. Use NGC Container (includes Blackwell support):\n"
                "     docker run --gpus all nvcr.io/nvidia/pytorch:25.02-py3\n"
            )
            
            self.logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
        # Log GPU information
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        memory_gb = device_props.total_memory / (1024**3)
        
        self.logger.info(f"✅ GPU Validation Passed")
        self.logger.info(f"   Device: {device_name}")
        self.logger.info(f"   Memory: {memory_gb:.1f} GB")
        self.logger.info(f"   Compute Capability: sm_{device_props.major}{device_props.minor}")
```

### 2. Health Check with GPU Status

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check with GPU enforcement status"""
    
    gpu_available = torch.cuda.is_available()
    
    if not gpu_available:
        # Service is unhealthy without GPU
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "status": "unhealthy",
                "error": "GPU required but not available",
                "gpu_required": True,
                "gpu_available": False,
                "message": "Service requires GPU acceleration for operation"
            }
        )
    
    # Healthy with GPU
    return {
        "status": "healthy",
        "gpu_required": True,
        "gpu_available": True,
        "gpu_details": {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        }
    }
```

## Environment and Configuration Patterns

### 1. Environment Variable Handling

```python
import os

def configure_gpu_environment():
    """Configure environment for GPU-only operation"""
    
    # Remove any CPU-forcing environment variables
    cpu_forcing_vars = ['CUDA_VISIBLE_DEVICES', 'FORCE_CPU', 'USE_CPU']
    for var in cpu_forcing_vars:
        if var in os.environ:
            if var == 'CUDA_VISIBLE_DEVICES' and os.environ[var] == '':
                raise RuntimeError(
                    f"{var} is set to empty string, hiding all GPUs.\n"
                    f"Remove this environment variable or set to valid GPU ID (e.g., '0')"
                )
            elif var in ['FORCE_CPU', 'USE_CPU'] and os.environ[var].lower() in ['true', '1', 'yes']:
                raise RuntimeError(
                    f"{var}={os.environ[var]} forces CPU mode.\n"
                    f"This service requires GPU. Remove {var} environment variable."
                )
    
    # Set GPU-optimizing environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent GPU ordering
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.0;8.6;8.9;9.0;12.0'  # Support various architectures
    
    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
```

### 2. Startup Script with GPU Validation

```bash
#!/bin/bash
# start_whisper_gpu_only.sh

echo "🚀 Starting Whisper API (GPU-Only Mode)"

# Check for GPU presence
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "   This service requires GPU acceleration."
    exit 1
fi

# Verify GPU is accessible
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Cannot access GPU. Check NVIDIA driver installation."
    exit 1
fi

# Display GPU information
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Check CUDA availability in Python
python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ PyTorch cannot access CUDA. GPU is required.')
    print('   Install PyTorch with CUDA support:')
    print('   pip install torch --index-url https://download.pytorch.org/whl/cu118')
    sys.exit(1)

print(f'✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU validation failed. Cannot start service."
    exit 1
fi

# Start the service
echo "✅ GPU validation passed. Starting service..."
python main.py
```

## Testing GPU Enforcement

### Unit Test Example

```python
import unittest
from unittest.mock import patch, MagicMock
import torch

class TestGPUEnforcement(unittest.TestCase):
    
    def test_gpu_required_when_not_available(self):
        """Test service fails when GPU not available"""
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(RuntimeError) as context:
                enforce_gpu_availability()
            
            self.assertIn("GPU", str(context.exception))
            self.assertIn("required", str(context.exception).lower())
    
    def test_gpu_validation_passes_with_gpu(self):
        """Test validation passes when GPU available"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5060 Ti'):
                    # Should not raise exception
                    device_name, _, _ = validate_gpu_capabilities()
                    self.assertEqual(device_name, 'NVIDIA RTX 5060 Ti')
    
    def test_insufficient_memory_fails(self):
        """Test validation fails with insufficient GPU memory"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                # Mock device properties with low memory
                mock_props = MagicMock()
                mock_props.total_memory = 2 * 1024**3  # 2GB
                mock_props.major = 8
                mock_props.minor = 6
                
                with patch('torch.cuda.get_device_properties', return_value=mock_props):
                    with self.assertRaises(RuntimeError) as context:
                        validate_gpu_capabilities()
                    
                    self.assertIn("memory", str(context.exception).lower())
```

## Summary

These patterns ensure strict GPU-only operation by:

1. **Failing fast** - Check GPU availability before any model loading
2. **Clear errors** - Provide detailed error messages with remediation steps
3. **No fallbacks** - Remove all code paths that could fall back to CPU
4. **Health monitoring** - Report GPU status in health checks
5. **Environment validation** - Check for environment variables that might force CPU
6. **Comprehensive testing** - Unit tests verify GPU enforcement works correctly

The key principle is: **Better to fail with clear instructions than degrade to poor performance**.