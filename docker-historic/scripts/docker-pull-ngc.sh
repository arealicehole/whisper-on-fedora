#!/bin/bash

# Pull NVIDIA NGC PyTorch container with progress indication
# This container is large (~8-10GB) so first pull takes time

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NGC_IMAGE="nvcr.io/nvidia/pytorch:25.02-py3"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Pulling NVIDIA NGC PyTorch Container${NC}"
echo -e "${BLUE}================================================${NC}"
echo
echo "Container: $NGC_IMAGE"
echo "Size: ~8-10GB (first download only)"
echo
echo -e "${YELLOW}This will take several minutes on first pull...${NC}"
echo

# Pull with progress
docker pull $NGC_IMAGE

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✅ Container pulled successfully!${NC}"
    echo
    
    # Show image info
    echo "Image details:"
    docker images | grep pytorch | head -1
    
    echo
    echo "Testing PyTorch version..."
    docker run --rm $NGC_IMAGE python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"
else
    echo -e "${RED}✗ Failed to pull container${NC}"
    exit 1
fi