#!/bin/bash

# Setup script for Whisper API with Blackwell GPU (RTX 5060 Ti) support
# Uses PyTorch nightly builds for sm_120 compute capability
# This fixes the "operator torchvision::nms does not exist" error

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Blackwell GPU Setup for Whisper API${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Error: Python 3.11 is required but not found.${NC}"
    echo "On Fedora/RHEL: sudo dnf install python3.11 python3.11-devel"
    echo "On Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 3.11 found"

# Create virtual environment with Python 3.11
VENV_DIR="$HOME/.venvs/whisper-blackwell"
echo -e "${YELLOW}Creating virtual environment at $VENV_DIR...${NC}"

# Remove old venv if it exists (make it idempotent)
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

# Create parent directory if needed
mkdir -p "$HOME/.venvs"

python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo -e "${GREEN}✓${NC} Virtual environment created and activated"

# Upgrade pip to latest
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# CRITICAL: Install PyTorch nightly with CUDA 12.8 support for Blackwell
echo ""
echo -e "${YELLOW}Installing PyTorch nightly for Blackwell GPU (sm_120)...${NC}"
echo "This may take a few minutes..."

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
echo ""
echo -e "${YELLOW}Verifying PyTorch installation...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f'GPU detected: {gpu_name}')
    print(f'Compute capability: sm_{capability[0]}{capability[1]}')
    if capability == (12, 0):
        print('✅ Blackwell GPU detected!')
else:
    print('⚠️  CUDA not available - check GPU drivers')
"

# Install core dependencies (without PyTorch)
echo ""
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip install \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.9 \
    httpx==0.27.0 \
    pydantic==2.6.0 \
    faster-whisper==1.0.3 \
    numpy==1.26.4 \
    soundfile==0.12.1 \
    librosa==0.10.1

# Install PyAnnote WITHOUT allowing it to downgrade PyTorch
echo ""
echo -e "${YELLOW}Installing PyAnnote (without downgrading PyTorch)...${NC}"

# First install pyannote.audio without dependencies
pip install --no-deps pyannote.audio==3.3.1

# Then install its dependencies individually (excluding torch/torchvision)
pip install \
    asteroid-filterbanks>=0.4.0 \
    einops>=0.6.0 \
    hbreader>=0.9.1 \
    hyperpyyaml>=1.2.0 \
    julius>=0.2.7 \
    omegaconf>=2.3.0 \
    pyannote.core>=5.0.0 \
    pyannote.database>=5.0.0 \
    pyannote.metrics>=3.2.0 \
    pyannote.pipeline>=3.0.0 \
    pytorch-lightning>=2.0.0 \
    pytorch-metric-learning>=2.3.0 \
    rich>=13.0.0 \
    semver>=3.0.0 \
    sentencepiece>=0.1.97 \
    speechbrain>=1.0.0 \
    torchmetrics>=0.11.0 \
    transformers>=4.44.0 \
    torch-audiomentations>=0.11.0

# Test torchvision NMS operator (critical for PyAnnote)
echo ""
echo -e "${YELLOW}Testing torchvision NMS operator...${NC}"
python -c "
import torch
import torchvision.ops
if torch.cuda.is_available():
    # Test NMS operator that fails with NGC containers
    boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32).cuda()
    scores = torch.tensor([0.9], dtype=torch.float32).cuda()
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print('✅ Torchvision NMS operator working!')
else:
    print('⚠️  Cannot test NMS without CUDA')
"

# Setup token location
echo ""
echo -e "${YELLOW}Setting up token configuration...${NC}"
TOKEN_DIR="$HOME/.config/whisper"
mkdir -p "$TOKEN_DIR"

if [ -f "$TOKEN_DIR/token" ]; then
    echo -e "${GREEN}✓${NC} Token file already exists at $TOKEN_DIR/token"
else
    echo -e "${YELLOW}Note: You need to set your HuggingFace token for diarization${NC}"
    echo "  1. Get token from: https://huggingface.co/settings/tokens"
    echo "  2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "  3. Create token file:"
    echo "     echo 'HF_TOKEN=your_token_here' > $TOKEN_DIR/token"
fi

# Final summary
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${GREEN}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "To test the installation, run:"
echo -e "  ${GREEN}python validate_blackwell.py${NC}"
echo ""
echo "To start the service:"
echo -e "  ${GREEN}python main.py${NC}"
echo ""
echo -e "${YELLOW}Important: This venv uses PyTorch nightly builds${NC}"
echo "If you encounter issues, re-run this script to get the latest nightly."