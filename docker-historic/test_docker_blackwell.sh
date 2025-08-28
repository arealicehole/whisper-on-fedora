#!/bin/bash

echo "Testing Blackwell GPU support in container..."

docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.02-py3 python -c "
import torch
import sys

print('=' * 60)
print('NVIDIA NGC Container - Blackwell Test')
print('=' * 60)

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: {major}.{minor} (sm_{major}{minor})')
    
    arch_list = torch.cuda.get_arch_list()
    print(f'Architectures: {arch_list}')
    
    if 'sm_120' in arch_list:
        print('')
        print('✅ Blackwell (sm_120) is SUPPORTED!')
        
        # Test operation
        try:
            test = torch.randn(3, 3).cuda()
            result = test @ test.T
            print('✅ CUDA operations work!')
        except Exception as e:
            print(f'❌ CUDA operation failed: {e}')
    else:
        print('')
        print('❌ Blackwell (sm_120) NOT supported in this container')
        print('Try a newer NGC container version')

print('=' * 60)
"
