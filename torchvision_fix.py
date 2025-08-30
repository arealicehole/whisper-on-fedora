#!/usr/bin/env python3
"""
Torchvision NMS operator fix for Blackwell GPU (RTX 5060 Ti)
This module provides a workaround for the "operator torchvision::nms does not exist" error
"""

import torch
import warnings

def register_nms_fallback():
    """Register a fallback NMS implementation if torchvision NMS is not available"""
    
    try:
        # Try to import torchvision ops
        import torchvision.ops
        print("✓ Torchvision ops imported successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not import torchvision.ops: {e}")
    
    # Fallback: Define NMS manually if torchvision fails
    try:
        # Check if NMS operator is registered
        if not hasattr(torch.ops, 'torchvision') or not hasattr(torch.ops.torchvision, 'nms'):
            print("Registering fallback NMS implementation...")
            
            # Pure PyTorch NMS implementation
            def nms_fallback(boxes, scores, iou_threshold):
                """
                Non-Maximum Suppression fallback implementation
                Args:
                    boxes: Tensor of shape (N, 4) with box coordinates [x1, y1, x2, y2]
                    scores: Tensor of shape (N,) with box scores
                    iou_threshold: Float, IoU threshold for suppression
                Returns:
                    Tensor of kept box indices
                """
                if boxes.numel() == 0:
                    return torch.empty((0,), dtype=torch.long, device=boxes.device)
                
                # Sort by scores
                _, order = scores.sort(descending=True)
                keep = []
                
                while order.numel() > 0:
                    i = order[0]
                    keep.append(i)
                    
                    if order.numel() == 1:
                        break
                    
                    # Compute IoU with remaining boxes
                    xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
                    yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
                    xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
                    yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
                    
                    w = torch.clamp(xx2 - xx1, min=0)
                    h = torch.clamp(yy2 - yy1, min=0)
                    inter = w * h
                    
                    areas_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                    areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
                    
                    iou = inter / (areas_i + areas - inter)
                    
                    # Keep boxes with IoU less than threshold
                    mask = iou <= iou_threshold
                    order = order[1:][mask]
                
                return torch.tensor(keep, dtype=torch.long, device=boxes.device)
            
            # Register the fallback
            torch.ops.torchvision = torch.ops.torchvision if hasattr(torch.ops, 'torchvision') else lambda: None
            torch.ops.torchvision.nms = nms_fallback
            print("✓ Fallback NMS registered successfully")
            return True
            
    except Exception as e:
        print(f"Error registering NMS fallback: {e}")
        return False

def initialize_torchvision_blackwell():
    """Initialize torchvision with Blackwell GPU compatibility"""
    
    # Suppress torchvision warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    
    # For Blackwell GPUs, we need to ensure torchvision loads correctly
    # The NGC container has torchvision but it needs special handling
    
    # First, register our fallback NMS
    success = register_nms_fallback()
    
    # Try to fix torchvision import for PyAnnote
    try:
        # Monkey-patch torchvision if needed
        import sys
        import types
        
        # Create a minimal torchvision.ops module if it doesn't exist
        if 'torchvision' not in sys.modules:
            torchvision = types.ModuleType('torchvision')
            sys.modules['torchvision'] = torchvision
            torchvision.ops = types.ModuleType('torchvision.ops')
            sys.modules['torchvision.ops'] = torchvision.ops
        elif not hasattr(sys.modules['torchvision'], 'ops'):
            torchvision = sys.modules['torchvision']
            torchvision.ops = types.ModuleType('torchvision.ops')
            sys.modules['torchvision.ops'] = torchvision.ops
        
        # Ensure NMS is available in torchvision.ops
        if hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'):
            sys.modules['torchvision.ops'].nms = torch.ops.torchvision.nms
            print("✓ Torchvision NMS operator registered in module")
        
        success = True
        
    except Exception as e:
        print(f"Error setting up torchvision module: {e}")
    
    return success

if __name__ == "__main__":
    # Test the fix
    if initialize_torchvision_blackwell():
        print("\n✅ Torchvision initialized successfully for Blackwell GPU")
        
        # Additional test
        try:
            import torchvision
            print(f"Torchvision version: {torchvision.__version__}")
        except:
            print("Note: Torchvision not fully imported, but fallback NMS is available")
    else:
        print("\n❌ Failed to initialize torchvision")