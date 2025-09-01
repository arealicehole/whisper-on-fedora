#!/bin/bash
# Run whisper-api in Docker with CUDA 12.4 support

# Kill any existing whisper services on port 8767
echo "Stopping existing services..."
docker stop whisper-api-cuda 2>/dev/null || true
docker rm whisper-api-cuda 2>/dev/null || true
lsof -ti:8767 | xargs kill -9 2>/dev/null || true

# Create directories if they don't exist
mkdir -p models config audio_files

# Use NVIDIA's NGC PyTorch container which has CUDA 12.4 and cuDNN 8.9
# This should work with Blackwell GPUs
echo "Starting Whisper API with CUDA 12.4 in Docker..."

docker run -d \
  --name whisper-api-cuda \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8767:8767 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.03-py3 \
  bash -c "
    # Install dependencies
    pip install --no-cache-dir \
      faster-whisper==1.0.3 \
      ctranslate2==4.4.0 \
      fastapi==0.110.0 \
      uvicorn[standard]==0.27.0 \
      python-multipart==0.0.9 \
      httpx==0.27.0 \
      pydantic==2.6.0 \
      soundfile \
      librosa \
      scipy && \
    # Run the application
    python main_docker.py
  "

echo "Waiting for service to start..."
sleep 10

# Check if it's running
if docker ps | grep whisper-api-cuda > /dev/null; then
  echo "✅ Whisper API is running in Docker"
  echo ""
  echo "Testing health endpoint..."
  curl -s http://localhost:8767/health | python3 -m json.tool | head -20
  echo ""
  echo "View logs with: docker logs -f whisper-api-cuda"
else
  echo "❌ Failed to start Whisper API"
  echo "Check logs with: docker logs whisper-api-cuda"
  docker logs whisper-api-cuda 2>&1 | tail -20
fi