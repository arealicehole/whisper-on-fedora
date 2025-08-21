#!/bin/bash

# Install Whisper API as a system service
# This will make it start automatically on boot

echo "Installing Whisper API as system service..."

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Copy service file
cp whisper-api.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service (start on boot)
systemctl enable whisper-api

echo ""
echo "✓ Service installed successfully!"
echo ""
echo "Commands:"
echo "  Start now:    sudo systemctl start whisper-api"
echo "  Stop:         sudo systemctl stop whisper-api"
echo "  Status:       sudo systemctl status whisper-api"
echo "  Logs:         sudo journalctl -u whisper-api -f"
echo "  Disable boot: sudo systemctl disable whisper-api"
echo ""
echo "The service will start automatically on boot."