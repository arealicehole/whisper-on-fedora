#!/bin/bash

# Initialize as a GitHub repository

echo "Initializing Whisper API GitHub repository..."

# Initialize git if not already
if [ ! -d .git ]; then
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Whisper API with optional speaker diarization

- FastAPI-based REST API for audio transcription
- GPU-accelerated using faster-whisper
- Optional speaker diarization with pyannote
- Multiple output formats (JSON, SRT, VTT)
- Docker and systemd support
- Comprehensive documentation and examples"

echo ""
echo "Repository initialized!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote:"
echo "   git remote add origin https://github.com/yourusername/whisper-api.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Don't forget to:"
echo "- Update 'yourusername' in README.md with your GitHub username"
echo "- Update the LICENSE file with your name"
echo "- Add topics to your GitHub repo: whisper, speech-to-text, fastapi, diarization, python"