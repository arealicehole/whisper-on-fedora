#!/usr/bin/env python3
"""Test script to debug pyannote diarization issues"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    modules = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("pyannote.audio", "Pyannote Audio"),
        ("speechbrain", "SpeechBrain"),
        ("transformers", "Transformers"),
    ]
    
    all_good = True
    for module_name, display_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name}: {version}")
            
            # Special check for torch CUDA
            if module_name == "torch":
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    print(f"  CUDA: {torch.version.cuda}")
                    print(f"  Device: {torch.cuda.get_device_name(0)}")
                else:
                    print("  CUDA: Not available")
                    
        except ImportError as e:
            print(f"✗ {display_name}: {e}")
            all_good = False
    
    return all_good

def test_token():
    """Test HuggingFace token availability"""
    print("\nTesting HF token...")
    
    token_file = os.path.expanduser("~/.config/whisper/token")
    token = None
    
    if os.path.exists(token_file):
        with open(token_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    
    if token and token != "hf_PUT_YOUR_VALID_TOKEN_HERE":
        print(f"✓ Token found: {token[:10]}...{token[-4:]}")
        os.environ["HF_TOKEN"] = token
        return token
    else:
        print("✗ No valid token found")
        print(f"  Please add your token to: {token_file}")
        print("  Format: HF_TOKEN=hf_your_actual_token_here")
        return None

def test_pipeline_download(token):
    """Test if the diarization pipeline can be downloaded"""
    print("\nTesting pipeline download...")
    
    if not token:
        print("✗ Skipping - no token available")
        return None
    
    try:
        from pyannote.audio import Pipeline
        print("Loading pipeline...")
        
        # Try to load the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        print("✓ Pipeline loaded successfully!")
        
        # Try to move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                print("✓ Pipeline moved to GPU")
        except Exception as e:
            print(f"  GPU move failed: {e}")
        
        return pipeline
        
    except Exception as e:
        print(f"✗ Pipeline loading failed: {e}")
        
        # Common fixes
        if "401" in str(e):
            print("\n  Fix: Your token might be invalid or expired")
            print("  1. Go to https://huggingface.co/settings/tokens")
            print("  2. Create a new token with 'read' access")
            print("  3. Update ~/.config/whisper/token")
            
        elif "404" in str(e):
            print("\n  Fix: Model not found. You might need to:")
            print("  1. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  2. Make sure you're logged in to HuggingFace")
            
        elif "Connection" in str(e):
            print("\n  Fix: Network issue. Check your internet connection")
            
        return None

def test_on_audio(pipeline, audio_file):
    """Test diarization on an actual audio file"""
    print(f"\nTesting on audio file: {audio_file}")
    
    if not pipeline:
        print("✗ No pipeline available")
        return
        
    if not os.path.exists(audio_file):
        print(f"✗ Audio file not found: {audio_file}")
        return
    
    try:
        print("Running diarization...")
        diarization = pipeline(audio_file)
        
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            print(f"  {turn.start:.1f}s - {turn.end:.1f}s: {speaker}")
            if len(speakers) >= 3:  # Just show first few for demo
                print("  ...")
                break
        
        print(f"✓ Found {len(speakers)} speakers")
        
    except Exception as e:
        print(f"✗ Diarization failed: {e}")

def suggest_fixes():
    """Suggest fixes based on common issues"""
    print("\n" + "="*50)
    print("RECOMMENDED SETUP")
    print("="*50)
    
    print("""
1. Use Python 3.11 (best compatibility):
   python3.11 -m venv ~/.venvs/whisper-diarize
   source ~/.venvs/whisper-diarize/bin/activate

2. Install with specific versions:
   pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
   pip install pyannote.audio==3.1.1
   
3. Get HuggingFace token:
   - Go to https://huggingface.co/settings/tokens
   - Create token with 'read' access
   - Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Save token to ~/.config/whisper/token

4. Common version combinations that work:
   - Python 3.11 + torch 2.2.0 + pyannote.audio 3.1.1
   - Python 3.10 + torch 2.1.0 + pyannote.audio 3.0.1
   - Python 3.9 + torch 2.0.0 + pyannote.audio 2.1.1
""")

if __name__ == "__main__":
    print("Pyannote Diarization Diagnostic Tool")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test token
    token = test_token()
    
    # Test pipeline
    pipeline = None
    if imports_ok:
        pipeline = test_pipeline_download(token)
    
    # Test on audio if provided
    if len(sys.argv) > 1 and pipeline:
        test_on_audio(pipeline, sys.argv[1])
    
    # Show recommendations
    if not imports_ok or not token or not pipeline:
        suggest_fixes()