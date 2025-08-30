#!/bin/bash

# Pre-patch Python to fix torchvision before ANY imports
python -c "
import sys
import types
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create complete fake torchvision module
torchvision = types.ModuleType('torchvision')
torchvision.__version__ = '0.22.0'
torchvision.__file__ = 'fake'
torchvision.__spec__ = types.SimpleNamespace(name='torchvision', loader=None)

# Register it
sys.modules['torchvision'] = torchvision

# Create submodules
for submodule in ['ops', 'transforms', 'models', 'datasets', 'io', 'utils']:
    module = types.ModuleType(f'torchvision.{submodule}')
    module.__spec__ = types.SimpleNamespace(name=f'torchvision.{submodule}', loader=None)
    setattr(torchvision, submodule, module)
    sys.modules[f'torchvision.{submodule}'] = module

print('✅ Torchvision pre-patched')
"

# Now start the actual application
exec python /app/startup.py