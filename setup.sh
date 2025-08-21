#!/bin/bash

# One-command setup for GitHub users
# This script sets up everything needed to run Whisper API

echo "================================"
echo "Whisper API Setup"
echo "================================"
echo ""

# Check system requirements
echo "Checking system requirements..."

# Check Python
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found"
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "⚠ Python $PYTHON_VERSION found (3.11 recommended)"
    PYTHON_CMD="python3"
else
    echo "✗ Python not found. Please install Python 3.11"
    exit 1
fi

# Check CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
else
    echo "ℹ No NVIDIA GPU detected (CPU mode will be used)"
fi

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg found"
else
    echo "⚠ FFmpeg not found (recommended for audio processing)"
    echo "  Install with: sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (Mac)"
fi

echo ""
echo "Setting up environment..."

# Create virtual environment
if [ ! -d "$HOME/.venvs/whisper-diarize" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$HOME/.venvs/whisper-diarize"
else
    echo "Virtual environment already exists"
fi

# Activate and install
echo "Installing dependencies..."
source "$HOME/.venvs/whisper-diarize/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install dependencies
echo "Installing PyTorch..."
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -q

echo "Installing requirements..."
if [ -f "requirements_diarization.txt" ]; then
    pip install -r requirements_diarization.txt -q
else
    pip install fastapi uvicorn[standard] faster-whisper pyannote.audio numpy httpx pydantic -q
fi

echo ""
echo "Checking HuggingFace token..."

TOKEN_FILE="$HOME/.config/whisper/token"
if [ -f "$TOKEN_FILE" ]; then
    echo "✓ Token file found"
else
    echo "ℹ No HuggingFace token found"
    echo ""
    echo "To enable speaker diarization:"
    echo "1. Get a token from: https://huggingface.co/settings/tokens"
    echo "2. Accept the license: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "3. Save your token:"
    echo "   mkdir -p ~/.config/whisper"
    echo "   echo 'HF_TOKEN=hf_your_token_here' > ~/.config/whisper/token"
fi

echo ""
echo "Testing installation..."

# Test imports
python -c "
import sys
try:
    import fastapi
    import faster_whisper
    print('✓ Core dependencies installed')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)

try:
    import torch
    import pyannote.audio
    print('✓ Diarization dependencies installed')
except ImportError:
    print('ℹ Diarization dependencies not available')
" || exit 1

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To start the service:"
echo "  ./start_whisper.sh start"
echo ""
echo "Or manually:"
echo "  source ~/.venvs/whisper-diarize/bin/activate"
echo "  python main.py"
echo ""
echo "Test the API:"
echo "  curl http://localhost:8765/health"
echo ""
echo "For usage examples, see:"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo "  - examples/"