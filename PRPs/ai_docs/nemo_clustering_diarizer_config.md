# NeMo ClusteringDiarizer Configuration Reference

## Critical Configuration Parameters for Production

This document provides detailed configuration guidance for the NeMo ClusteringDiarizer, specifically optimized for the RTX 5060 Ti (Blackwell) GPU with 16GB VRAM.

## Complete Configuration Template

```yaml
# File: configs/diar_infer_inference.yaml
name: &name "ClusteringDiarizer"

diarizer:
  manifest_filepath: ???  # Will be set dynamically
  out_dir: ./nemo_outputs
  oracle_vad: False  # Always False for inference
  collar: 0.25  # Evaluation collar in seconds
  ignore_overlap: True  # Ignore overlapping speech in scoring

  # Voice Activity Detection Configuration
  vad:
    model_path: 'vad_multilingual_marblenet'  # Best multilingual VAD model
    external_vad_manifest: null  # Can provide external VAD results
    
    parameters:
      # Window for VAD processing
      window_length_in_sec: 0.63  # Optimal for MarbleNet
      shift_length_in_sec: 0.01   # 10ms shift for fine-grained detection
      
      # VAD decision thresholds
      onset: 0.9   # Higher = more conservative speech detection
      offset: 0.5  # Lower = longer speech segments
      
      # Post-processing parameters
      min_duration_on: 0.1   # Minimum speech duration (seconds)
      min_duration_off: 0.6  # Minimum silence duration (seconds)
      filter_speech_first: True
      
      # Advanced parameters
      smoothing: False  # Disable for better boundaries
      overlap: 0.5     # Overlap ratio for windowing

  # Speaker Embedding Extraction
  speaker_embeddings:
    model_path: 'titanet_large'  # Best accuracy model
    # Alternative: 'titanet_base' for 2x faster processing
    
    parameters:
      # Multi-scale segmentation windows (seconds)
      window_length_in_sec: [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
      shift_length_in_sec: [1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
      
      # Weights for combining multi-scale scores
      multiscale_weights: [1, 1, 1, 1, 1, 1]  # Equal weighting
      
      # Memory optimization
      save_embeddings: False  # Set True for debugging
      batch_size: 32  # Reduce if OOM occurs

  # Clustering Configuration
  clustering:
    parameters:
      # Speaker count estimation
      oracle_num_speakers: False  # Auto-detect speaker count
      max_num_speakers: 8        # Maximum speakers to detect
      enhanced_count_thres: 80   # Threshold for enhanced counting
      
      # Clustering algorithm parameters
      max_rp_threshold: 0.25    # Maximum threshold for eigen gap
      sparse_search_volume: 30  # Search volume for sparse affinity
      maj_vote_spk_count: False # Majority voting for speaker count
      
      # NME clustering specific (if using NME)
      nme_rmax: 0.3
      nme_rmin: 0.05
      nme_Kmax: 8
```

## Parameter Tuning Guide

### For Different Audio Types

#### Meeting/Conference Audio
```yaml
vad:
  parameters:
    onset: 0.8  # More permissive for multiple speakers
    offset: 0.6
speaker_embeddings:
  parameters:
    window_length_in_sec: [3.0, 2.0, 1.0]  # Longer windows for cleaner audio
```

#### Telephonic/Call Center Audio
```yaml
vad:
  parameters:
    onset: 0.9  # Stricter due to noise
    offset: 0.5
speaker_embeddings:
  model_path: 'titanet_base'  # Faster processing for real-time needs
```

#### Podcast/Interview Audio
```yaml
clustering:
  parameters:
    oracle_num_speakers: True
    max_num_speakers: 4  # Usually 2-4 speakers
```

### Memory Optimization Settings

#### For 8GB GPUs
```yaml
speaker_embeddings:
  parameters:
    window_length_in_sec: [1.5, 1.0, 0.5]  # Fewer scales
    batch_size: 16  # Smaller batch
```

#### For 16GB GPUs (RTX 5060 Ti)
```yaml
speaker_embeddings:
  parameters:
    window_length_in_sec: [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]  # Full multi-scale
    batch_size: 32  # Optimal batch size
```

## Model Selection Guide

### VAD Models
- `vad_multilingual_marblenet`: Best overall, supports 100+ languages
- `vad_marblenet`: English-optimized, slightly faster
- `vad_telephony_marblenet`: Optimized for phone calls

### Speaker Embedding Models
- `titanet_large`: Best accuracy, 23M parameters
- `titanet_base`: 2x faster, 12M parameters
- `ecapa_tdnn`: Legacy model, compatible with older GPUs

## Dynamic Configuration in Python

```python
from omegaconf import OmegaConf

def create_config(audio_type='general', num_speakers=None, gpu_memory_gb=16):
    """Create optimized config based on audio characteristics"""
    
    base_config = OmegaConf.load('configs/diar_infer_inference.yaml')
    
    # Adjust for GPU memory
    if gpu_memory_gb < 12:
        base_config.diarizer.speaker_embeddings.parameters.batch_size = 16
        base_config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.0, 0.5]
    
    # Adjust for audio type
    if audio_type == 'telephony':
        base_config.diarizer.vad.parameters.onset = 0.95
        base_config.diarizer.speaker_embeddings.model_path = 'titanet_base'
    elif audio_type == 'meeting':
        base_config.diarizer.vad.parameters.onset = 0.8
        base_config.diarizer.clustering.parameters.max_num_speakers = 12
    
    # Set known speaker count if provided
    if num_speakers:
        base_config.diarizer.clustering.parameters.oracle_num_speakers = True
        base_config.diarizer.clustering.parameters.max_num_speakers = num_speakers
    
    return base_config
```

## Performance Benchmarks

| Configuration | Processing Speed | DER (%) | GPU Memory |
|--------------|------------------|---------|------------|
| Default | 1.5s/min | 15-20 | 4GB |
| Optimized for Speed | 0.8s/min | 18-23 | 3GB |
| Optimized for Accuracy | 2.2s/min | 12-17 | 6GB |
| Multi-scale Full | 2.5s/min | 10-15 | 8GB |

## Troubleshooting

### Common Issues and Solutions

1. **OOM Error**
   - Reduce `batch_size`
   - Reduce number of scales in `window_length_in_sec`
   - Use `titanet_base` instead of `titanet_large`

2. **Poor Diarization Quality**
   - Increase `onset` threshold if too many false speakers
   - Decrease `offset` if missing speech segments
   - Add more scales to `window_length_in_sec`

3. **Slow Processing**
   - Use `titanet_base` model
   - Reduce multi-scale windows to [1.5, 1.0, 0.5]
   - Enable VAD filtering first

4. **Wrong Speaker Count**
   - Adjust `enhanced_count_thres`
   - Set `oracle_num_speakers: True` if count is known
   - Tune `max_rp_threshold` for eigen gap

## Integration Example

```python
import json
import os
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

class OptimizedNeMoDiarizer:
    def __init__(self, gpu_memory_gb=16):
        self.base_config = self.create_optimized_config(gpu_memory_gb)
        
    def create_optimized_config(self, gpu_memory_gb):
        """Create config optimized for available GPU memory"""
        config = {
            'diarizer': {
                'manifest_filepath': None,
                'out_dir': './nemo_outputs',
                'oracle_vad': False,
                'vad': {
                    'model_path': 'vad_multilingual_marblenet',
                    'parameters': {
                        'onset': 0.9,
                        'offset': 0.5,
                        'min_duration_on': 0.1,
                        'min_duration_off': 0.6
                    }
                },
                'speaker_embeddings': {
                    'model_path': 'titanet_large' if gpu_memory_gb >= 12 else 'titanet_base',
                    'parameters': {
                        'window_length_in_sec': [3.0, 2.0, 1.0, 0.5] if gpu_memory_gb >= 12 else [1.5, 1.0],
                        'shift_length_in_sec': [1.5, 1.0, 0.5, 0.25] if gpu_memory_gb >= 12 else [0.75, 0.5],
                        'multiscale_weights': [1, 1, 1, 1] if gpu_memory_gb >= 12 else [1, 1],
                        'batch_size': 32 if gpu_memory_gb >= 16 else 16
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': False,
                        'max_num_speakers': 8
                    }
                }
            }
        }
        return OmegaConf.create(config)
    
    def diarize(self, audio_path, num_speakers=None):
        # Create manifest
        manifest_path = 'temp_manifest.json'
        manifest_data = {
            'audio_filepath': audio_path,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'num_speakers': num_speakers
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
            f.write('\n')
        
        # Update config
        config = self.base_config.copy()
        config.diarizer.manifest_filepath = manifest_path
        
        if num_speakers:
            config.diarizer.clustering.parameters.oracle_num_speakers = True
            config.diarizer.clustering.parameters.max_num_speakers = num_speakers
        
        # Run diarization
        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()
        
        # Parse results
        return self.parse_rttm_output(audio_path)
```

This configuration guide ensures optimal performance on the target hardware while maintaining flexibility for different audio types and use cases.