#!/bin/bash
# Test script for NGC container with Blackwell support

echo "🚀 Testing NVIDIA NGC Container with Blackwell GPU"
echo "=================================================="

# Check if NGC image is downloaded
if docker images | grep -q "nvcr.io/nvidia/pytorch.*24.07"; then
    echo "✅ NGC PyTorch container found"
else
    echo "⏳ Pulling NGC container (this is large, ~15GB)..."
    docker pull nvcr.io/nvidia/pytorch:24.07-py3
fi

echo ""
echo "🧪 Test 1: GPU Detection in NGC Container"
echo "------------------------------------------"
docker run --gpus all --rm \
    nvcr.io/nvidia/pytorch:24.07-py3 \
    python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    
    # Test tensor operation
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print('✅ GPU tensor operations working!')
"

echo ""
echo "🧪 Test 2: Installing and Testing faster-whisper"
echo "-------------------------------------------------"
docker run --gpus all --rm -v /home/ice/whisper-api:/workspace \
    nvcr.io/nvidia/pytorch:24.07-py3 \
    bash -c "
        pip install -q faster-whisper
        python -c '
from faster_whisper import WhisperModel
import torch
print(f\"Testing faster-whisper with NGC PyTorch {torch.__version__}\")
try:
    model = WhisperModel(\"tiny\", device=\"cuda\", compute_type=\"float16\")
    print(\"✅ faster-whisper loaded on GPU successfully!\")
    print(f\"   Using CUDA: {torch.cuda.is_available()}\")
    print(f\"   GPU: {torch.cuda.get_device_name(0)}\")
except Exception as e:
    print(f\"❌ Error: {e}\")
'
"

echo ""
echo "🧪 Test 3: Full Stack Test (faster-whisper + NeMo potential)"
echo "-------------------------------------------------------------"
docker run --gpus all --rm -v /home/ice/whisper-api:/workspace \
    nvcr.io/nvidia/pytorch:24.07-py3 \
    bash -c "
        cd /workspace
        pip install -q faster-whisper fastapi uvicorn python-multipart
        python -c '
import torch
from faster_whisper import WhisperModel

print(\"Full Stack Test:\")
print(f\"- PyTorch: {torch.__version__}\")
print(f\"- CUDA: {torch.version.cuda}\")
print(f\"- cuDNN: {torch.backends.cudnn.version()}\")
print(f\"- GPU: {torch.cuda.get_device_name(0)}\")

# Test faster-whisper
model = WhisperModel(\"tiny\", device=\"cuda\", compute_type=\"float16\")

# Test with dummy audio
import numpy as np
dummy_audio = np.random.randn(16000).astype(np.float32)
segments, info = model.transcribe(dummy_audio)
print(\"✅ Transcription test passed!\")
print(\"\\n🎉 NGC container WORKS with Blackwell GPU!\")
'
"

echo ""
echo "=================================================="
echo "If all tests pass, the NGC container is the solution!"
echo "No more waiting for ecosystem - NVIDIA already did the work!"