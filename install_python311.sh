#!/bin/bash

# Script to install Python 3.11 alongside existing Python installation

echo "Installing Python 3.11 for pyannote compatibility..."
echo ""

# Detect the OS
if [ -f /etc/fedora-release ]; then
    echo "Detected Fedora/RHEL system"
    echo "Running: sudo dnf install python3.11 python3.11-devel"
    sudo dnf install -y python3.11 python3.11-devel python3.11-pip
    
elif [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu system"
    echo "Adding deadsnakes PPA for Python 3.11..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
    
elif [ -f /etc/arch-release ]; then
    echo "Detected Arch Linux"
    echo "Python 3.11 might be in AUR"
    echo "Try: yay -S python311"
    
else
    echo "Unknown OS. Manual installation required."
    echo ""
    echo "Option 1: Build from source"
    echo "  wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz"
    echo "  tar -xf Python-3.11.9.tgz"
    echo "  cd Python-3.11.9"
    echo "  ./configure --enable-optimizations"
    echo "  make -j$(nproc)"
    echo "  sudo make altinstall"
    echo ""
    echo "Option 2: Use pyenv"
    echo "  curl https://pyenv.run | bash"
    echo "  pyenv install 3.11.9"
    exit 1
fi

echo ""
echo "Checking installation..."
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 installed successfully!"
    python3.11 --version
    echo ""
    echo "Now run: ./setup_venv.sh"
else
    echo "⚠ Python 3.11 not found. Installation may have failed."
fi