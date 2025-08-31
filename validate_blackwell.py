#!/usr/bin/env python3
"""
Blackwell GPU Validation Script
Tests PyTorch nightly, torchvision NMS, and PyAnnote diarization
"""

import sys
import os
from pathlib import Path


def validate_blackwell():
    """Comprehensive validation for Blackwell GPU setup"""
    
    print("🔍 Blackwell GPU Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: PyTorch and CUDA
    print("\n📊 Test 1: PyTorch and CUDA")
    print("-" * 30)
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check for nightly build
        if "dev" in torch.__version__ or "nightly" in torch.__version__:
            print("✅ Using PyTorch nightly build")
        else:
            print("⚠️  Not using nightly build - may not support Blackwell")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA is not available!")
            print("   Check your GPU drivers and CUDA installation")
            all_passed = False
        else:
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ Memory: {memory_gb:.1f} GB")
            print(f"✅ Compute Capability: sm_{capability[0]}{capability[1]}")
            
            # Check for Blackwell
            if capability == (12, 0):
                print("🎯 Blackwell GPU (sm_120) detected!")
            else:
                print(f"⚠️  Not a Blackwell GPU - sm_{capability[0]}{capability[1]}")
                
            # Test tensor operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                print("✅ Basic CUDA operations working")
            except Exception as e:
                print(f"❌ CUDA operations failed: {e}")
                all_passed = False
                
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        all_passed = False
        return False
    
    # Test 2: Torchvision NMS operator (critical for PyAnnote)
    print("\n📊 Test 2: Torchvision NMS Operator")
    print("-" * 30)
    try:
        import torchvision
        import torchvision.ops
        
        print(f"✅ Torchvision version: {torchvision.__version__}")
        
        if torch.cuda.is_available():
            # This is the exact operation that fails with NGC containers
            boxes = torch.tensor([[0, 0, 10, 10], 
                                 [5, 5, 15, 15], 
                                 [20, 20, 30, 30]], dtype=torch.float32).cuda()
            scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32).cuda()
            iou_threshold = 0.5
            
            try:
                keep = torchvision.ops.nms(boxes, scores, iou_threshold)
                print(f"✅ NMS operation successful! Kept indices: {keep.cpu().tolist()}")
                print("✅ This confirms torchvision is compatible")
            except RuntimeError as e:
                if "no kernel image" in str(e).lower() or "nms does not exist" in str(e).lower():
                    print(f"❌ CRITICAL: Torchvision NMS not working: {e}")
                    print("   This is the Blackwell compatibility issue!")
                    print("   Make sure you're using PyTorch nightly builds")
                    all_passed = False
                else:
                    print(f"❌ NMS operation failed: {e}")
                    all_passed = False
        else:
            print("⚠️  Cannot test NMS without CUDA")
            
    except ImportError as e:
        print(f"❌ Failed to import torchvision: {e}")
        all_passed = False
    
    # Test 3: PyAnnote imports and pipeline
    print("\n📊 Test 3: PyAnnote Audio")
    print("-" * 30)
    try:
        from pyannote.audio import Pipeline
        print("✅ PyAnnote.audio imported successfully")
        
        # Check for token
        token_file = Path.home() / ".config" / "whisper" / "token"
        hf_token = None
        
        if token_file.exists():
            with open(token_file) as f:
                content = f.read().strip()
                if content.startswith("HF_TOKEN="):
                    hf_token = content.split("=", 1)[1].strip()
                    print("✅ HuggingFace token found")
        
        # Try to load pipeline (requires token)
        if hf_token and torch.cuda.is_available():
            try:
                print("Loading PyAnnote pipeline (this may take a moment)...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                
                # Move to GPU
                pipeline.to(torch.device("cuda"))
                print("✅ PyAnnote pipeline loaded on GPU successfully!")
                print("✅ Diarization is ready to use")
                
            except Exception as e:
                print(f"⚠️  Could not load pipeline: {e}")
                if "401" in str(e) or "unauthorized" in str(e):
                    print("   Token may be invalid or model terms not accepted")
                    print("   Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            if not hf_token:
                print("⚠️  No HuggingFace token found")
                print("   Diarization requires a token. To set one:")
                print("   echo 'HF_TOKEN=your_token' > ~/.config/whisper/token")
            if not torch.cuda.is_available():
                print("⚠️  Cannot test pipeline without CUDA")
                
    except ImportError as e:
        print(f"❌ Failed to import pyannote.audio: {e}")
        all_passed = False
    
    # Test 4: Faster-whisper
    print("\n📊 Test 4: Faster-whisper")
    print("-" * 30)
    try:
        import faster_whisper
        print("✅ Faster-whisper imported successfully")
        
        # Check if we can create a model (don't download, just check)
        from faster_whisper import WhisperModel
        print("✅ WhisperModel class available")
        
        if torch.cuda.is_available():
            print("✅ Ready for GPU-accelerated transcription")
        
    except ImportError as e:
        print(f"❌ Failed to import faster-whisper: {e}")
        all_passed = False
    
    # Test 5: FastAPI service dependencies
    print("\n📊 Test 5: FastAPI Dependencies")
    print("-" * 30)
    try:
        import fastapi
        import uvicorn
        import httpx
        import pydantic
        print("✅ FastAPI imported successfully")
        print("✅ Uvicorn imported successfully")
        print("✅ All API dependencies available")
        
    except ImportError as e:
        print(f"❌ Failed to import API dependencies: {e}")
        all_passed = False
    
    # Final Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED!")
        print("\nYour Blackwell GPU setup is ready for:")
        print("  • GPU-accelerated Whisper transcription")
        print("  • PyAnnote speaker diarization")
        print("  • FastAPI service deployment")
        print("\nNext steps:")
        print("  1. Ensure HuggingFace token is set (if using diarization)")
        print("  2. Run: python main.py")
        print("  3. Access API at: http://127.0.0.1:8767")
        return True
    else:
        print("❌ VALIDATION FAILED")
        print("\nPlease address the issues above.")
        print("Common fixes:")
        print("  • Re-run setup_blackwell_venv.sh")
        print("  • Ensure you're using the whisper-blackwell venv")
        print("  • Check GPU drivers with: nvidia-smi")
        return False


if __name__ == "__main__":
    success = validate_blackwell()
    sys.exit(0 if success else 1)