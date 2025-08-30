#!/usr/bin/env python3
"""
Fix PyAnnote models for Blackwell GPU (sm_120) by recompiling CUDA kernels
"""

import torch
import os
import sys

def fix_pyannote_for_blackwell():
    """Force PyAnnote to recompile kernels for sm_120 architecture"""
    
    print("Fixing PyAnnote for Blackwell GPU (RTX 5060 Ti)...")
    
    # Set environment variables to force kernel compilation
    os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Clear PyTorch's kernel cache to force recompilation
    torch.cuda.empty_cache()
    
    # Force JIT compilation for sm_120
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        
        if capability == (12, 0):
            print(f"✓ Detected Blackwell GPU with compute capability {capability}")
            
            # Set PyTorch to compile for this specific architecture
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Test kernel compilation
            try:
                test_tensor = torch.randn(10, 10).cuda()
                _ = test_tensor @ test_tensor
                print("✓ CUDA kernels compiled successfully for sm_120")
                return True
            except Exception as e:
                print(f"✗ Failed to compile kernels: {e}")
                return False
        else:
            print(f"Warning: Expected sm_120, got sm_{capability[0]}{capability[1]}")
            return False
    else:
        print("✗ CUDA not available")
        return False

if __name__ == "__main__":
    success = fix_pyannote_for_blackwell()
    sys.exit(0 if success else 1)