#!/bin/bash

# Setup script for Whisper API with working pyannote diarization
# This uses Python 3.11 which is known to work well with pyannote.audio 3.x

set -e

echo "Setting up Whisper API with diarization support..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is required but not found."
    echo "On Fedora/RHEL: sudo dnf install python3.11"
    echo "On Ubuntu/Debian: sudo apt install python3.11"
    exit 1
fi

# Create virtual environment with Python 3.11
VENV_DIR="$HOME/.venvs/whisper-diarize"
echo "Creating virtual environment at $VENV_DIR..."

python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch first with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install pyannote.audio and dependencies with specific versions
echo "Installing pyannote.audio..."
pip install pyannote.audio==3.1.1

# Install other dependencies
echo "Installing other dependencies..."
pip install \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.9 \
    httpx==0.27.0 \
    pydantic==2.6.0 \
    faster-whisper==1.0.0 \
    numpy==1.24.3 \
    soundfile==0.12.1 \
    librosa==0.10.1

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To test diarization, create a token file:"
echo "  mkdir -p ~/.config/whisper"
echo "  echo 'HF_TOKEN=your_huggingface_token_here' > ~/.config/whisper/token"
echo ""
echo "Then run the service:"
echo "  python main.py"