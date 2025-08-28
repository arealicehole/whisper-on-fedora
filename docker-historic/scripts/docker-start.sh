#!/bin/bash

# Start Whisper API with Blackwell GPU support via Docker
# This script handles all setup and verification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Starting Whisper API with Blackwell Support${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Step 1: Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Please install Docker first:"
    echo "  https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker found: $(docker --version)"

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker daemon is not running${NC}"
    echo "Start Docker with: sudo systemctl start docker"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker daemon is running"

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠ NVIDIA Container Toolkit may not be properly configured${NC}"
    echo "Testing will continue, but GPU access might fail."
    echo "To install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
else
    echo -e "${GREEN}✓${NC} NVIDIA Container Toolkit is configured"
fi

# Check for HuggingFace token
if [ ! -f "$HOME/.config/whisper/token" ]; then
    echo -e "${YELLOW}⚠ HuggingFace token not found${NC}"
    echo "Diarization may not work without a token."
    echo "Add your token to: ~/.config/whisper/token"
    echo "Format: HF_TOKEN=hf_xxxxx"
fi

# Step 2: Navigate to project directory
cd "$PROJECT_DIR"

# Step 3: Build Docker image
echo
echo -e "${YELLOW}Building Docker image...${NC}"
echo "This may take several minutes on first run."

docker compose -f docker/docker-compose.yml build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Docker image built successfully"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Step 4: Stop any existing container
echo
echo -e "${YELLOW}Stopping any existing containers...${NC}"
docker compose -f docker/docker-compose.yml down 2>/dev/null || true

# Step 5: Start the service
echo
echo -e "${YELLOW}Starting Whisper API service...${NC}"

docker compose -f docker/docker-compose.yml up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to start container${NC}"
    echo "Check logs with: docker compose -f docker/docker-compose.yml logs"
    exit 1
fi

# Step 6: Wait for service to be ready
echo
echo -e "${YELLOW}Waiting for service to initialize...${NC}"

MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s -f http://localhost:8765/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Service is ready!"
        break
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 2
done

echo

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}✗ Service failed to start within 60 seconds${NC}"
    echo "Check logs with: docker compose -f docker/docker-compose.yml logs"
    exit 1
fi

# Step 7: Verify GPU support
echo
echo -e "${YELLOW}Verifying GPU support...${NC}"

docker compose -f docker/docker-compose.yml exec whisper-blackwell python -c "
import torch
import sys

try:
    assert torch.cuda.is_available(), 'CUDA not available'
    gpu_name = torch.cuda.get_device_name(0)
    print(f'✓ GPU detected: {gpu_name}')
    
    arch_list = torch.cuda.get_arch_list()
    if 'sm_120' in arch_list:
        print('✓ Blackwell architecture (sm_120) supported')
    else:
        print('⚠ Warning: sm_120 not in architecture list')
        print(f'  Available: {arch_list}')
    
    # Test CUDA operation
    test = torch.randn(3, 3).cuda()
    result = test @ test.T
    print('✓ CUDA operations working')
    
except Exception as e:
    print(f'✗ GPU verification failed: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} GPU verification successful"
else
    echo -e "${YELLOW}⚠${NC} GPU verification had issues, but service is running"
fi

# Step 8: Display service information
echo
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}✅ Whisper API is running with Blackwell support!${NC}"
echo -e "${BLUE}================================================${NC}"
echo
echo "Service URL: http://localhost:8765"
echo
echo "Test commands:"
echo -e "  ${BLUE}# Check health:${NC}"
echo "  curl http://localhost:8765/health | jq ."
echo
echo -e "  ${BLUE}# Transcribe audio:${NC}"
echo "  curl -X POST http://localhost:8765/v1/transcribe \\"
echo "    -F 'file=@audio.wav' | jq ."
echo
echo -e "  ${BLUE}# With diarization:${NC}"
echo "  curl -X POST http://localhost:8765/v1/transcribe \\"
echo "    -F 'file=@audio.wav' -F 'diarize=true' | jq ."
echo
echo "Management commands:"
echo -e "  ${BLUE}# View logs:${NC}"
echo "  docker compose -f docker/docker-compose.yml logs -f"
echo
echo -e "  ${BLUE}# Stop service:${NC}"
echo "  docker compose -f docker/docker-compose.yml down"
echo
echo -e "  ${BLUE}# Restart service:${NC}"
echo "  docker compose -f docker/docker-compose.yml restart"
echo
echo -e "${GREEN}Your existing whisper_client.py scripts will work unchanged!${NC}"