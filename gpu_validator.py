#!/usr/bin/env python3
"""
GPU Validation and Enforcement Utility for Whisper API
Ensures GPU-only operation with no CPU fallback
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch


@dataclass
class GPURequirements:
    """GPU requirements specification for Whisper API"""
    min_memory_gb: float = 4.0
    required_compute_capability: tuple = (7, 5)  # sm_75 minimum
    required_device_count: int = 1


@dataclass
class GPUValidationResult:
    """GPU validation result with detailed information"""
    is_valid: bool
    device_name: Optional[str] = None
    memory_gb: Optional[float] = None
    compute_capability: Optional[tuple] = None
    error_message: Optional[str] = None
    remediation_steps: Optional[List[str]] = None


class GPUEnforcementError(Exception):
    """Custom exception for GPU requirement failures"""
    def __init__(self, message: str, remediation: List[str] = None):
        super().__init__(message)
        self.remediation = remediation or []
    
    def print_remediation(self):
        """Print remediation steps to console"""
        if self.remediation:
            print("\n📋 Remediation Steps:", file=sys.stderr)
            for i, step in enumerate(self.remediation, 1):
                print(f"   {i}. {step}", file=sys.stderr)


class GPUValidator:
    """Comprehensive GPU validation for Whisper API"""
    
    def __init__(self, requirements: Optional[GPURequirements] = None):
        self.requirements = requirements or GPURequirements()
    
    def validate_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        return torch.cuda.is_available()
    
    def validate_device_count(self) -> int:
        """Validate GPU device count"""
        if not self.validate_cuda_available():
            return 0
        return torch.cuda.device_count()
    
    def validate_memory(self, device: int = 0) -> float:
        """Validate GPU memory in GB"""
        if not self.validate_cuda_available():
            return 0.0
        props = torch.cuda.get_device_properties(device)
        return props.total_memory / (1024**3)
    
    def validate_compute_capability(self, device: int = 0) -> tuple:
        """Validate GPU compute capability"""
        if not self.validate_cuda_available():
            return (0, 0)
        props = torch.cuda.get_device_properties(device)
        return (props.major, props.minor)
    
    def enforce_gpu_requirements(self) -> GPUValidationResult:
        """Enforce GPU requirements - fail if not met"""
        
        # Check CUDA availability
        if not self.validate_cuda_available():
            remediation = [
                "Verify GPU presence: nvidia-smi",
                "Install CUDA-enabled PyTorch:",
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "For RTX 5060 Ti Blackwell (sm_120), use:",
                "  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
                "Or use NGC container: nvcr.io/nvidia/pytorch:25.02-py3",
                "Check CUDA_VISIBLE_DEVICES environment variable is not empty"
            ]
            raise GPUEnforcementError(
                "GPU is required but CUDA is not available. CPU operation is not supported.",
                remediation
            )
        
        # Validate device count
        device_count = self.validate_device_count()
        if device_count < self.requirements.required_device_count:
            raise GPUEnforcementError(
                f"Insufficient GPUs: {device_count} < {self.requirements.required_device_count} required. "
                f"This service requires GPU acceleration for acceptable performance.",
                ["Ensure at least one NVIDIA GPU is installed and accessible"]
            )
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        memory_gb = device_props.total_memory / (1024**3)
        compute_capability = (device_props.major, device_props.minor)
        
        # Validate memory
        if memory_gb < self.requirements.min_memory_gb:
            raise GPUEnforcementError(
                f"Insufficient GPU memory: {memory_gb:.1f}GB < {self.requirements.min_memory_gb}GB required. "
                f"Device: {device_name}",
                ["Use a GPU with at least 4GB VRAM", "Reduce model size", "Close other GPU applications"]
            )
        
        # Check for Blackwell architecture
        if compute_capability == (12, 0):
            print(f"⚠️  Blackwell GPU detected (sm_{compute_capability[0]}{compute_capability[1]})")
            print("   Ensure PyTorch nightly or NGC container is used for full support")
        
        # Check minimum compute capability
        min_major, min_minor = self.requirements.required_compute_capability
        if (compute_capability[0] < min_major or 
            (compute_capability[0] == min_major and compute_capability[1] < min_minor)):
            raise GPUEnforcementError(
                f"GPU compute capability {compute_capability[0]}.{compute_capability[1]} is below minimum "
                f"{min_major}.{min_minor} required. Device: {device_name}",
                ["Upgrade to a newer GPU (RTX 2000 series or newer recommended)"]
            )
        
        # Return successful validation
        return GPUValidationResult(
            is_valid=True,
            device_name=device_name,
            memory_gb=memory_gb,
            compute_capability=compute_capability
        )
    
    def check_environment_conflicts(self):
        """Check for environment variables that might force CPU mode"""
        conflicts = []
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible == '':
            conflicts.append(
                "CUDA_VISIBLE_DEVICES is set to empty string, hiding all GPUs. "
                "Remove this variable or set to valid GPU ID (e.g., '0')"
            )
        
        # Check for CPU-forcing variables
        cpu_vars = ['FORCE_CPU', 'USE_CPU', 'WHISPER_ALLOW_CPU', 'FORCE_CPU_ONLY']
        for var in cpu_vars:
            if var in os.environ and os.environ[var].lower() in ['true', '1', 'yes']:
                conflicts.append(
                    f"{var}={os.environ[var]} forces CPU mode. "
                    f"This service requires GPU. Remove {var} environment variable."
                )
        
        if conflicts:
            # Include variable names in error message for clarity
            conflict_msg = "Environment configuration conflicts with GPU-only requirement: " + "; ".join(conflicts)
            raise GPUEnforcementError(conflict_msg, conflicts)


def main():
    """Standalone GPU validation utility"""
    print("🔍 Whisper API GPU Validation")
    print("=" * 50)
    
    validator = GPUValidator()
    
    try:
        # Check environment first
        validator.check_environment_conflicts()
        
        # Validate GPU requirements
        result = validator.enforce_gpu_requirements()
        
        print(f"✅ GPU Validation PASSED")
        print(f"\n📊 GPU Details:")
        print(f"   Device: {result.device_name}")
        print(f"   Memory: {result.memory_gb:.1f} GB")
        print(f"   Compute Capability: sm_{result.compute_capability[0]}{result.compute_capability[1]}")
        print(f"\n✨ GPU is ready for Whisper API operation")
        
        return 0
        
    except GPUEnforcementError as e:
        print(f"\n❌ GPU Validation FAILED: {e}", file=sys.stderr)
        e.print_remediation()
        print(f"\n⚠️  This service requires GPU acceleration. CPU fallback is not supported.", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error during GPU validation: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())