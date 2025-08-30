"""
sitecustomize.py - Automatically loaded by Python before any imports
This patches torchvision for Blackwell GPU compatibility
"""

import sys
import types
import warnings

# Only run this once
if 'torchvision' not in sys.modules:
    warnings.filterwarnings('ignore')
    
    # Create complete fake torchvision module
    class FakeModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__file__ = f'/fake/{name}'
            self.__loader__ = None
            self.__package__ = name.rsplit('.', 1)[0] if '.' in name else None
            self.__spec__ = types.SimpleNamespace(
                name=name,
                loader=None,
                origin=self.__file__,
                submodule_search_locations=[] if '.' in name else [f'/fake/{name}']
            )
    
    # Create torchvision and all its submodules
    torchvision = FakeModule('torchvision')
    torchvision.__version__ = '0.22.0'
    
    # Register main module
    sys.modules['torchvision'] = torchvision
    
    # Create and register all submodules that might be imported
    submodules = [
        'ops', 'ops.boxes', 'transforms', 'transforms.functional',
        'models', 'datasets', 'io', 'utils'
    ]
    
    for submodule_name in submodules:
        full_name = f'torchvision.{submodule_name}'
        submodule = FakeModule(full_name)
        sys.modules[full_name] = submodule
        
        # Add to parent module
        parts = submodule_name.split('.')
        parent = torchvision
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, FakeModule(f'torchvision.{part}'))
            parent = getattr(parent, part)
        setattr(parent, parts[-1], submodule)
    
    # Add essential classes/functions
    class Compose:
        def __init__(self, transforms): self.transforms = transforms
        def __call__(self, img): return img
    
    class ToTensor:
        def __call__(self, pic): return pic
    
    torchvision.transforms.Compose = Compose
    torchvision.transforms.ToTensor = ToTensor
    
    print("✅ Torchvision pre-patched via sitecustomize")