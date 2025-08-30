#!/usr/bin/env python3
"""
PyAnnote Blackwell GPU Fix
Provides NMS implementation without requiring torchvision
"""

import torch
import sys
import types
import warnings

def nms_cpu(boxes, scores, iou_threshold):
    """
    CPU implementation of Non-Maximum Suppression
    Args:
        boxes: Tensor of shape (N, 4) with box coordinates [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with box scores
        iou_threshold: Float, IoU threshold for suppression
    Returns:
        Tensor of kept box indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # Convert to CPU if needed
    device = boxes.device
    boxes = boxes.cpu()
    scores = scores.cpu()
    
    # Get indices sorted by score
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
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
        
        # Keep boxes with IoU less than threshold
        mask = iou <= iou_threshold
        order = remaining[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=device)

def patch_pyannote_for_blackwell():
    """
    Patch PyAnnote to work without torchvision on Blackwell GPUs
    """
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create fake torchvision module with NMS
    if 'torchvision' not in sys.modules:
        torchvision = types.ModuleType('torchvision')
        sys.modules['torchvision'] = torchvision
    else:
        torchvision = sys.modules['torchvision']
    
    # Create ops submodule
    if not hasattr(torchvision, 'ops'):
        torchvision.ops = types.ModuleType('torchvision.ops')
        sys.modules['torchvision.ops'] = torchvision.ops
    
    # Add NMS function
    torchvision.ops.nms = nms_cpu
    
    # Also register in torch.ops for compatibility
    if not hasattr(torch.ops, 'torchvision'):
        # Create namespace
        torch.ops.torchvision = types.SimpleNamespace()
    
    torch.ops.torchvision.nms = nms_cpu
    
    # Patch common torchvision imports that PyAnnote might use
    torchvision.ops.boxes = types.ModuleType('torchvision.ops.boxes')
    torchvision.ops.boxes.nms = nms_cpu
    sys.modules['torchvision.ops.boxes'] = torchvision.ops.boxes
    
    # Add version info to satisfy import checks
    torchvision.__version__ = '0.22.0'
    
    # Add transforms module (minimal implementation for PyAnnote)
    torchvision.transforms = types.ModuleType('torchvision.transforms')
    sys.modules['torchvision.transforms'] = torchvision.transforms
    
    # Add common transform classes PyAnnote might need
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
    
    torchvision.transforms.Compose = Compose
    torchvision.transforms.ToTensor = ToTensor
    
    # Add models module (empty but importable)
    torchvision.models = types.ModuleType('torchvision.models')
    sys.modules['torchvision.models'] = torchvision.models
    
    # Add datasets module (empty but importable)
    torchvision.datasets = types.ModuleType('torchvision.datasets')
    sys.modules['torchvision.datasets'] = torchvision.datasets
    
    print("✅ PyAnnote patched for Blackwell GPU (NMS fallback registered)")
    return True

# Auto-patch on import
patch_pyannote_for_blackwell()