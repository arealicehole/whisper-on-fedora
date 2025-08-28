#!/bin/bash

# Quick Docker GPU test without downloading large containers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Quick Docker GPU Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Test 1: Check Docker
echo -e "${YELLOW}Test 1: Docker Status${NC}"
docker --version
echo -e "${GREEN}✓${NC} Docker is installed"
echo

# Test 2: Check NVIDIA runtime with small CUDA image
echo -e "${YELLOW}Test 2: NVIDIA Container Runtime${NC}"
echo "Using smaller CUDA base image for quick test..."

if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi; then
    echo -e "${GREEN}✓${NC} GPU is accessible in Docker"
else
    echo -e "${RED}✗${NC} GPU not accessible. Check nvidia-container-toolkit"
    exit 1
fi

echo
echo -e "${YELLOW}Test 3: GPU Details${NC}"
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi \
    --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv

echo
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Basic GPU support confirmed!${NC}"
echo -e "${BLUE}================================================${NC}"
echo
echo "Your RTX 5060 Ti (Compute 12.0) is accessible in Docker."
echo
echo "To download the full NGC PyTorch container (~8-10GB), run:"
echo -e "  ${BLUE}./scripts/docker-pull-ngc.sh${NC}"
echo
echo "Or start the Whisper service directly (will pull if needed):"
echo -e "  ${BLUE}./scripts/docker-start.sh${NC}"
echo
echo -e "${YELLOW}Note: First pull of NGC container takes 10-20 minutes${NC}"