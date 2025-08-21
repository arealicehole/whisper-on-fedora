#!/bin/bash

# Setup isolated Python 3.11 environment for Whisper with diarization
# This keeps everything contained without affecting system Python

set -e

echo "==================================="
echo "Isolated Python 3.11 Setup"
echo "==================================="
echo ""
echo "This will create a completely isolated Python 3.11 environment"
echo "without affecting your system Python 3.13"
echo ""

# Option 1: Check if conda/mamba is available
if command -v conda &> /dev/null; then
    echo "Found conda! Creating isolated environment..."
    conda create -n whisper-diarize python=3.11 -y
    echo ""
    echo "✓ Created conda environment 'whisper-diarize'"
    echo ""
    echo "To use it:"
    echo "  conda activate whisper-diarize"
    echo "  pip install -r requirements_diarization.txt"
    echo "  python main.py"
    exit 0
fi

# Option 2: Use pyenv (recommended for isolation)
PYENV_ROOT="$HOME/.pyenv"

if [ ! -d "$PYENV_ROOT" ]; then
    echo "Installing pyenv for isolated Python management..."
    echo "This won't affect your system Python at all."
    echo ""
    
    # Install pyenv
    curl https://pyenv.run | bash
    
    # Add to shell config
    echo "" >> ~/.bashrc
    echo '# Pyenv configuration' >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    
    # Load pyenv for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python 3.11.9 with pyenv
echo "Installing Python 3.11.9 (isolated)..."
pyenv install -s 3.11.9

# Create virtual environment with the isolated Python
echo "Creating virtual environment..."
~/.pyenv/versions/3.11.9/bin/python -m venv ~/.venvs/whisper-diarize

echo ""
echo "✅ Setup complete!"
echo ""
echo "The isolated environment is ready. To use it:"
echo ""
echo "  source ~/.venvs/whisper-diarize/bin/activate"
echo "  pip install -r requirements_diarization.txt"
echo "  python main.py"
echo ""
echo "This Python 3.11 environment is completely separate from your system Python 3.13"