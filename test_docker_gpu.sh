#!/bin/bash
echo "🚀 Testing Whisper with GPU in Docker CUDA 12.4..."

docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /home/ice/whisper-api:/app \
  -w /app \
  nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  bash -c '
    echo "📦 Installing dependencies..."
    apt-get update > /dev/null 2>&1
    apt-get install -y python3 python3-pip wget > /dev/null 2>&1
    
    echo "🔧 Installing PyTorch with CUDA 12.4..."
    pip3 install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    
    echo "📚 Installing faster-whisper..."
    pip3 install -q faster-whisper==1.0.3 ctranslate2==4.4.0 numpy soundfile
    
    echo "🧪 Testing GPU inference..."
    python3 -c "
import torch
import numpy as np
from faster_whisper import WhisperModel

print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU device: {torch.cuda.get_device_name(0)}\")
    print(f\"cuDNN version: {torch.backends.cudnn.version()}\")
    
    print(\"\nLoading Whisper model on GPU...\")
    model = WhisperModel(\"tiny\", device=\"cuda\", compute_type=\"float16\")
    print(\"✅ SUCCESS: Model loaded on GPU!\")
    print(\"\nThis configuration works for GPU acceleration.\")
else:
    print(\"❌ No GPU detected\")
"
  '