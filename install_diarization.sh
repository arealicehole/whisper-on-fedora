#!/bin/bash

# Quick installer for pyannote diarization with known working versions
# This script installs in the current Python environment

echo "Installing pyannote.audio with compatible versions..."
echo "This will install in your current Python environment."
echo ""

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Choose versions based on Python version
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "Warning: Python 3.13 may have compatibility issues."
    echo "Trying with latest versions..."
    TORCH_VERSION="2.5.0"
    PYANNOTE_VERSION="3.3.1"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "Using Python 3.12 compatible versions..."
    TORCH_VERSION="2.3.0"
    PYANNOTE_VERSION="3.1.1"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    echo "Using Python 3.11 compatible versions (recommended)..."
    TORCH_VERSION="2.2.0"
    PYANNOTE_VERSION="3.1.1"
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    echo "Using Python 3.10 compatible versions..."
    TORCH_VERSION="2.1.0"
    PYANNOTE_VERSION="3.0.1"
else
    echo "Using fallback versions..."
    TORCH_VERSION="2.0.0"
    PYANNOTE_VERSION="2.1.1"
fi

echo ""
echo "Installing PyTorch $TORCH_VERSION..."
pip install torch==$TORCH_VERSION torchaudio==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing pyannote.audio $PYANNOTE_VERSION..."
pip install pyannote.audio==$PYANNOTE_VERSION

echo ""
echo "Installing other required packages..."
pip install \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    pydantic \
    faster-whisper \
    numpy \
    soundfile \
    librosa

echo ""
echo "Testing installation..."
python -c "
import torch
from pyannote.audio import Pipeline
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print('✓ Pyannote.audio imported successfully')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation successful!"
    echo ""
    echo "Next steps:"
    echo "1. Make sure you have a HuggingFace token in ~/.config/whisper/token"
    echo "2. Accept the model license at: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "3. Run: python main.py"
else
    echo ""
    echo "⚠️  Installation may have issues. Try running ./setup_venv.sh for a clean virtual environment."
fi