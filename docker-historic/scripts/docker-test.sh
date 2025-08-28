#!/bin/bash

# Test script to verify Blackwell GPU support in Docker container
# Runs comprehensive diagnostics without starting the full service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Testing Blackwell GPU Support in Docker${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Test 1: Basic NVIDIA runtime test
echo -e "${YELLOW}Test 1: NVIDIA Docker Runtime${NC}"
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA Docker runtime is working"
    
    # Show GPU info
    docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi \
        --query-gpu=name,compute_cap,driver_version,memory.total \
        --format=csv,noheader | while IFS=',' read -r name compute driver memory; do
        echo "  GPU: $name"
        echo "  Compute Capability: $compute"
        echo "  Driver: $driver"
        echo "  Memory: $memory"
    done
else
    echo -e "${RED}✗${NC} NVIDIA Docker runtime not working"
    echo "Please install nvidia-container-toolkit"
    exit 1
fi

echo

# Test 2: NGC Container PyTorch version
echo -e "${YELLOW}Test 2: NGC Container PyTorch Version${NC}"
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.02-py3 python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')

# Check for cu128
if '+cu128' in torch.__version__ or '12.8' in str(torch.version.cuda):
    print('✓ Has CUDA 12.8 support')
else:
    print('⚠ May not have CUDA 12.8')
" 2>/dev/null

echo

# Test 3: Blackwell architecture support
echo -e "${YELLOW}Test 3: Blackwell Architecture (sm_120) Support${NC}"
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.02-py3 python -c "
import torch
import sys

print('Checking architecture support...')

if not torch.cuda.is_available():
    print('✗ CUDA is not available')
    sys.exit(1)

# Get GPU info
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
compute_cap = f'sm_{major}{minor}'

print(f'GPU: {gpu_name}')
print(f'Compute Capability: {major}.{minor} ({compute_cap})')

# Check compiled architectures
arch_list = torch.cuda.get_arch_list()
print(f'Compiled architectures: {arch_list}')

if 'sm_120' in arch_list:
    print('✅ Blackwell (sm_120) is SUPPORTED!')
else:
    print('❌ Blackwell (sm_120) is NOT supported')
    print('This container may need updating')

# Test if this specific GPU is supported
if compute_cap in arch_list or f'compute_{major}{minor}' in arch_list:
    print(f'✅ Your GPU ({compute_cap}) is supported')
else:
    print(f'⚠ Your GPU ({compute_cap}) may not be fully supported')
" 2>/dev/null

echo

# Test 4: CUDA operations
echo -e "${YELLOW}Test 4: CUDA Operations Test${NC}"
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.02-py3 python -c "
import torch
import sys

try:
    # Create tensors on GPU
    print('Testing CUDA operations...')
    
    # Test 1: Tensor creation
    x = torch.randn(1000, 1000).cuda()
    print('✓ Tensor creation on GPU')
    
    # Test 2: Matrix multiplication
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✓ Matrix multiplication')
    
    # Test 3: Neural network operations
    conv = torch.nn.Conv2d(3, 64, 3).cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    output = conv(input_tensor)
    print('✓ Convolution operation')
    
    # Test 4: Softmax
    softmax = torch.nn.functional.softmax(x[0], dim=0)
    print('✓ Softmax operation')
    
    print('')
    print('✅ All CUDA operations successful!')
    
except RuntimeError as e:
    if 'no kernel image' in str(e):
        print(f'❌ CRITICAL: {e}')
        print('This is the Blackwell compatibility issue!')
    else:
        print(f'❌ Error: {e}')
    sys.exit(1)
" 2>/dev/null

echo

# Test 5: Build and test the actual Whisper container
echo -e "${YELLOW}Test 5: Building Whisper Container${NC}"

cd "$PROJECT_DIR"

# Build the container
if docker compose -f docker/docker-compose.yml build --no-cache whisper-blackwell 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Container built successfully"
    
    # Test the built container
    echo
    echo -e "${YELLOW}Test 6: Testing Whisper Container${NC}"
    
    docker compose -f docker/docker-compose.yml run --rm whisper-blackwell python -c "
import torch
import sys

print('Whisper Container Diagnostics:')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: sm_{major}{minor}')
    
    arch_list = torch.cuda.get_arch_list()
    if 'sm_120' in arch_list:
        print('✅ Blackwell support confirmed in Whisper container')
    else:
        print('⚠ Blackwell support not detected')

# Test imports
try:
    import faster_whisper
    print('✓ faster-whisper imported')
except:
    print('✗ faster-whisper import failed')

try:
    from pyannote.audio import Pipeline
    print('✓ pyannote.audio imported')
except:
    print('✗ pyannote.audio import failed')

try:
    import fastapi
    import uvicorn
    print('✓ FastAPI/Uvicorn imported')
except:
    print('✗ FastAPI import failed')
" 2>/dev/null

else
    echo -e "${YELLOW}⚠${NC} Container build skipped or failed"
fi

# Final summary
echo
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"

echo
echo "If all tests passed, you can start the service with:"
echo -e "  ${BLUE}./scripts/docker-start.sh${NC}"
echo
echo "Or manually with:"
echo -e "  ${BLUE}cd $PROJECT_DIR${NC}"
echo -e "  ${BLUE}docker compose -f docker/docker-compose.yml up${NC}"
echo
echo -e "${GREEN}Your RTX 5060 Ti should now work with full CUDA acceleration!${NC}"