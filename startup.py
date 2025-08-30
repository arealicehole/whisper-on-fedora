#!/usr/bin/env python3
"""
Startup script for Whisper API with Blackwell GPU support
This pre-patches torchvision before any imports to ensure compatibility
"""

# CRITICAL: Patch torchvision BEFORE any imports that might use it
import sys
import types
import warnings

# Suppress all warnings during patching
warnings.filterwarnings("ignore")

# Create fake torchvision module structure BEFORE any imports
torchvision = types.ModuleType('torchvision')
torchvision.__version__ = '0.22.0'
torchvision.__file__ = '/usr/local/lib/python3.12/dist-packages/torchvision/__init__.py'
torchvision.__spec__ = types.SimpleNamespace(
    name='torchvision',
    loader=None,
    origin='/usr/local/lib/python3.12/dist-packages/torchvision/__init__.py',
    submodule_search_locations=['/usr/local/lib/python3.12/dist-packages/torchvision']
)

# Create all submodules that might be imported
torchvision.ops = types.ModuleType('torchvision.ops')
torchvision.ops.boxes = types.ModuleType('torchvision.ops.boxes')
torchvision.transforms = types.ModuleType('torchvision.transforms')
torchvision.models = types.ModuleType('torchvision.models')
torchvision.datasets = types.ModuleType('torchvision.datasets')
torchvision.io = types.ModuleType('torchvision.io')
torchvision.utils = types.ModuleType('torchvision.utils')

# Register all modules in sys.modules BEFORE any imports
sys.modules['torchvision'] = torchvision
sys.modules['torchvision.ops'] = torchvision.ops
sys.modules['torchvision.ops.boxes'] = torchvision.ops.boxes
sys.modules['torchvision.transforms'] = torchvision.transforms
sys.modules['torchvision.models'] = torchvision.models
sys.modules['torchvision.datasets'] = torchvision.datasets
sys.modules['torchvision.io'] = torchvision.io
sys.modules['torchvision.utils'] = torchvision.utils

# Now import torch and define NMS
import torch

def nms_fallback(boxes, scores, iou_threshold):
    """NMS fallback implementation for Blackwell GPU"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    device = boxes.device
    boxes = boxes.cpu()
    scores = scores.cpu()
    
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        remaining = order[1:]
        xx1 = torch.maximum(boxes[i, 0], boxes[remaining, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[remaining, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[remaining, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[remaining, 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        areas_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas = (boxes[remaining, 2] - boxes[remaining, 0]) * (boxes[remaining, 3] - boxes[remaining, 1])
        
        iou = inter / (areas_i + areas - inter)
        mask = iou <= iou_threshold
        order = remaining[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=device)

# Add NMS to torchvision.ops
torchvision.ops.nms = nms_fallback
torchvision.ops.boxes.nms = nms_fallback

# Add essential transforms that torchmetrics might need
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    def __call__(self, pic):
        return torch.from_numpy(pic) if hasattr(pic, 'shape') else pic

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img

# Add transforms to module
torchvision.transforms.Compose = Compose
torchvision.transforms.ToTensor = ToTensor
torchvision.transforms.Normalize = Normalize
torchvision.transforms.Resize = Resize

# Add some dummy functions that might be imported
torchvision.transforms.functional = types.ModuleType('torchvision.transforms.functional')
sys.modules['torchvision.transforms.functional'] = torchvision.transforms.functional

print("✅ Torchvision pre-patched for Blackwell GPU compatibility")

# Now run the main application
if __name__ == "__main__":
    import main  # This will import everything else