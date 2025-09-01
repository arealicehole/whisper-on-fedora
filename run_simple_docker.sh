#!/bin/bash
# Simple Docker run with the CUDA 12.4 image we already downloaded

echo "Stopping any existing containers..."
docker stop whisper-cuda-simple 2>/dev/null || true
docker rm whisper-cuda-simple 2>/dev/null || true

echo "Starting whisper with CUDA 12.4..."
docker run -d \
  --name whisper-cuda-simple \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -p 8769:8767 \
  -v $(pwd):/app \
  -w /app \
  nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  bash -c "
    apt-get update && apt-get install -y python3 python3-pip ffmpeg && \
    pip3 install torch==2.5.1+cu124 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install \
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
    python3 main_docker.py
  "

echo "Container started. Checking status in 30 seconds..."
sleep 30

if docker ps | grep whisper-cuda-simple > /dev/null; then
  echo "✅ Container is running"
  echo "Checking logs..."
  docker logs whisper-cuda-simple 2>&1 | tail -20
else
  echo "❌ Container failed to start"
  docker logs whisper-cuda-simple 2>&1 | tail -30
fi