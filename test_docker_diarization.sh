#!/bin/bash
echo "🧪 Testing Diarization Compatibility in Docker CUDA 12.4..."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set. Trying to read from ~/.config/whisper/token"
    if [ -f ~/.config/whisper/token ]; then
        export HF_TOKEN=$(grep "HF_TOKEN=" ~/.config/whisper/token | cut -d'=' -f2)
        echo "  Found token: ${HF_TOKEN:0:10}..."
    else
        echo "❌ No HF token found. Diarization won't work."
        echo "  Set HF_TOKEN environment variable or add to ~/.config/whisper/token"
        exit 1
    fi
fi

docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /home/ice/whisper-api:/app \
  -w /app \
  nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  bash -c '
    echo "📦 Installing dependencies..."
    apt-get update -qq > /dev/null 2>&1
    apt-get install -y -qq python3 python3-pip wget > /dev/null 2>&1
    
    echo "🔧 Installing PyTorch 2.5.1 with CUDA 12.4..."
    pip3 install -q torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    
    echo "📚 Installing PyAnnote dependencies..."
    pip3 install -q numpy scipy scikit-learn
    pip3 install -q pyannote.audio
    
    echo "🧪 Testing PyAnnote import..."
    python3 -c "
import os
import torch
import torchaudio

print(f\"PyTorch: {torch.__version__}\")
print(f\"Torchaudio: {torchaudio.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")

try:
    from pyannote.audio import Pipeline
    print(\"✅ PyAnnote imported successfully!\")
    
    # Try to load pipeline
    token = os.environ.get(\"HF_TOKEN\")
    if token:
        print(f\"Token found: {token[:10]}...\")
        try:
            pipeline = Pipeline.from_pretrained(
                \"pyannote/speaker-diarization-3.1\",
                use_auth_token=token
            )
            if torch.cuda.is_available():
                pipeline.to(torch.device(\"cuda\"))
            print(\"✅ Diarization pipeline loaded successfully!\")
            print(\"\\n🎉 DIARIZATION WILL WORK IN DOCKER!\")
        except Exception as e:
            print(f\"⚠️  Pipeline loading failed: {e}\")
            if \"401\" in str(e):
                print(\"  Token might be invalid\")
            elif \"403\" in str(e):
                print(\"  Need to accept model license at HuggingFace\")
    else:
        print(\"⚠️  No HF_TOKEN in environment\")
        
except ImportError as e:
    print(f\"❌ PyAnnote import failed: {e}\")
    print(\"\\nDiarization will NOT work - dependency issue\")
except Exception as e:
    print(f\"❌ Unexpected error: {e}\")
"
  '