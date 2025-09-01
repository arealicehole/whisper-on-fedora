# PRP: GPU-Based Whisper API with NeMo Diarization

## Goal
**Feature Goal**: Replace PyAnnote diarization with NVIDIA NeMo for full GPU acceleration and improved speaker diarization accuracy while maintaining API compatibility
**Deliverable**: Production-ready whisper API with NeMo ClusteringDiarizer that outputs the same format as current implementation
**Success Definition**: GPU-accelerated transcription + diarization working on Blackwell GPU (RTX 5060 Ti) with <2s processing per minute of audio

## Context
```yaml
references:
  - documentation:
      - url: https://github.com/SYSTRAN/faster-whisper#gpu
        purpose: GPU setup requirements for faster-whisper with CUDA 12
      - url: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
        purpose: NeMo diarization implementation tutorial
      - url: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
        purpose: NeMo speaker diarization architecture and configuration
      - url: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L1-L100
        purpose: faster-whisper transcribe API structure
  - examples:
      - url: https://github.com/MahmoudAshraf97/whisper-diarization
        patterns: Parallel processing of transcription and diarization
      - url: https://huggingface.co/spaces/ml6team/Speaker-Diarization/blob/main/diarizers/nemo_diarizer.py
        patterns: Production NeMo diarization wrapper
  - internal:
      - file: /home/ice/whisper-api/main.py:126-150
        convention: GPU enforcement and initialization patterns
      - file: /home/ice/whisper-api/main.py:474-521
        convention: API response format with segments and speaker labels
      - file: /home/ice/whisper-api/gpu_validator.py:80-124
        convention: Blackwell GPU detection and optimization

gotchas:
  - issue: Blackwell GPU (sm_120) not supported by stable PyTorch
    solution: Use NGC container with PyTorch 2.5.1 which has partial sm_120 support
  - issue: cuDNN version incompatibility (9.x vs 8.x)
    solution: Use Docker with CUDA 12.4 and cuDNN 8.x for compatibility
  - issue: NeMo outputs RTTM format, not Python objects
    solution: Parse RTTM file to extract speaker segments
  - issue: High memory usage with parallel processing
    solution: Implement sequential processing with GPU memory cleanup
  - issue: HuggingFace token required for model downloads
    solution: Read from ~/.config/whisper/token or environment

dependencies:
  - package: nemo_toolkit[all]==2.0.0
    reason: Latest stable with ClusteringDiarizer support
  - package: faster-whisper==1.0.3
    reason: Compatible with CUDA 12.4 environment
  - package: torch==2.5.1+cu124
    reason: Best compatibility with Docker CUDA 12.4
  - package: omegaconf==2.3.0
    reason: Required for NeMo configuration management
  - system: NVIDIA Docker runtime
    check: docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  - system: HuggingFace token with pyannote access
    check: test -f ~/.config/whisper/token && echo "Token exists"

patterns:
  - naming: main_nemo.py following main_docker.py pattern
  - structure: Docker-based deployment in docker-nemo/ directory
  - error_handling: Graceful degradation if diarization fails
  - gpu_init: Blackwell-specific optimizations from gpu_validator.py
  - api_format: Maintain exact segment structure with speaker field
```

## Implementation Tasks
```markdown
### Task 1: Docker Environment Setup
- Location: /home/ice/whisper-api/Dockerfile.nemo
- Pattern: Follow /home/ice/whisper-api/Dockerfile.with_diarization structure
- Dependencies: None
- Content:
  ```dockerfile
  FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
  # Install PyTorch 2.5.1 with CUDA 12.4
  RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  # Install NeMo and faster-whisper
  RUN pip install nemo_toolkit[asr]==2.0.0 faster-whisper==1.0.3
  ```
- Validation: docker build -f Dockerfile.nemo -t whisper-nemo . && docker run --rm whisper-nemo python -c "import nemo; import faster_whisper"

### Task 2: NeMo Configuration File
- Location: /home/ice/whisper-api/configs/diar_infer_inference.yaml
- Pattern: Based on NeMo tutorial configuration
- Dependencies: Task 1 completion
- Content structure:
  ```yaml
  diarizer:
    manifest_filepath: ???
    out_dir: ./nemo_outputs
    oracle_vad: False
    vad:
      model_path: vad_multilingual_marblenet
      parameters:
        onset: 0.9
        offset: 0.5
    speaker_embeddings:
      model_path: titanet_large
      parameters:
        window_length_in_sec: [1.5, 1.25, 1.0, 0.75, 0.5]
        shift_length_in_sec: [0.75, 0.625, 0.5, 0.375, 0.25]
    clustering:
      parameters:
        oracle_num_speakers: False
        max_num_speakers: 8
  ```
- Validation: python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/diar_infer_inference.yaml')"

### Task 3: NeMo Diarization Module
- Location: /home/ice/whisper-api/nemo_diarizer.py
- Key Classes/Functions:
  - NeMoDiarizer class wrapping ClusteringDiarizer
  - run_diarization(audio_path, num_speakers=None) -> List[Dict]
  - parse_rttm(rttm_path) -> List[Dict] with {start, end, speaker} format
- Integration Points: Replace pyannote Pipeline in main.py
- Pattern:
  ```python
  from omegaconf import OmegaConf
  from nemo.collections.asr.models import ClusteringDiarizer
  
  class NeMoDiarizer:
      def __init__(self, config_path, device="cuda"):
          self.config = OmegaConf.load(config_path)
          self.device = device
      
      def diarize(self, audio_path, num_speakers=None):
          # Create manifest
          # Run ClusteringDiarizer
          # Parse RTTM output
          # Return segments list
  ```
- Testing: python -c "from nemo_diarizer import NeMoDiarizer; d = NeMoDiarizer('configs/diar_infer_inference.yaml')"

### Task 4: Main Service Integration
- Location: /home/ice/whisper-api/main_nemo.py
- API Changes: None - maintain compatibility
- Data Flow:
  1. Audio upload -> temp file
  2. Parallel: faster-whisper transcription + NeMo diarization
  3. Align transcription segments with speaker segments
  4. Return combined result in original format
- Error Cases:
  - NeMo fails -> continue without speaker labels
  - GPU OOM -> clear cache and retry sequentially
  - No HF token -> skip diarization
- Key Integration:
  ```python
  # Replace pyannote import
  from nemo_diarizer import NeMoDiarizer
  
  # Initialize on startup
  diarization_pipeline = NeMoDiarizer('configs/diar_infer_inference.yaml')
  
  # In transcribe_audio function
  if diarize and diarization_pipeline:
      speaker_segments = diarization_pipeline.diarize(audio_path, num_speakers)
      # Align with transcription segments using midpoint algorithm
  ```
- Performance: Add torch.cuda.empty_cache() between models

### Task 5: Docker Compose Configuration
- Location: /home/ice/whisper-api/docker-compose.nemo.yml
- Pattern: Follow docker-compose.diarization.yml
- Content:
  ```yaml
  services:
    whisper-nemo:
      build:
        dockerfile: Dockerfile.nemo
      runtime: nvidia
      environment:
        - WHISPER_MODEL=small
        - WHISPER_DEVICE=cuda
        - HF_TOKEN=${HF_TOKEN}
      ports:
        - "8767:8767"
      volumes:
        - ./models:/app/models
        - ~/.cache/huggingface:/app/.cache/huggingface
  ```
- Validation: docker-compose -f docker-compose.nemo.yml config

### Task 6: Alignment Algorithm Optimization
- Location: Update alignment logic in main_nemo.py
- Current Pattern: Midpoint-based assignment
- Improved Pattern: Weighted intersection algorithm
  ```python
  def align_segments_weighted(transcription_segments, speaker_segments):
      for t_seg in transcription_segments:
          overlaps = {}
          for s_seg in speaker_segments:
              overlap = calculate_overlap(t_seg, s_seg)
              if overlap > 0:
                  speaker = s_seg['speaker']
                  overlaps[speaker] = overlaps.get(speaker, 0) + overlap
          if overlaps:
              t_seg['speaker'] = max(overlaps, key=overlaps.get)
  ```
- Testing: Unit tests with known segment overlaps

### Task 7: Memory Management
- Location: main_nemo.py GPU memory management
- Implementation:
  ```python
  # After transcription
  torch.cuda.empty_cache()
  
  # Before diarization
  if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
      torch.cuda.empty_cache()
      gc.collect()
  ```
- Long audio handling: Chunk into 10-minute segments with 30s overlap

### Task 8: Production Testing Script
- Location: /home/ice/whisper-api/test_nemo_integration.sh
- Content:
  ```bash
  #!/bin/bash
  # Test GPU detection
  docker-compose -f docker-compose.nemo.yml run whisper-nemo python -c "import torch; print(torch.cuda.is_available())"
  
  # Test transcription
  curl -X POST http://localhost:8767/v1/transcribe \
    -F "file=@test_audio.wav" \
    -F "diarize=true"
  
  # Check health endpoint
  curl http://localhost:8767/health | jq .diarization
  ```
```

## Validation Gates
```bash
# Component Test - NeMo loads correctly
docker run --rm --gpus all whisper-nemo:latest python -c "from nemo.collections.asr.models import ClusteringDiarizer; print('NeMo OK')"

# Integration Test - Both models work together
docker-compose -f docker-compose.nemo.yml run whisper-nemo python -c "
from faster_whisper import WhisperModel
from nemo_diarizer import NeMoDiarizer
model = WhisperModel('tiny', device='cuda')
diarizer = NeMoDiarizer('configs/diar_infer_inference.yaml')
print('Integration OK')
"

# Performance Check - Process test file under 2s/minute
time curl -X POST http://localhost:8767/v1/transcribe \
  -F "file=@test_sample.wav" \
  -F "diarize=true" \
  -F "format=json" | jq '.duration'

# System Validation - Full pipeline test
./test_nemo_integration.sh && echo "All tests passed"

# GPU Memory Check - Ensure no OOM
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

## Final Validation Checklist
- [ ] All existing tests pass: `python -m pytest tests/`
- [ ] New functionality works: Diarization returns speaker labels in segments
- [ ] Performance acceptable: <2s processing per minute of audio
- [ ] GPU memory stable: Peak usage <14GB on RTX 5060 Ti (16GB)
- [ ] API compatibility: Response format identical to current implementation
- [ ] Docker deployment: Container starts and runs without errors
- [ ] Health endpoint: Reports diarization available with NeMo backend
- [ ] Long audio: 1-hour file processes without OOM
- [ ] Error handling: Gracefully degrades when diarization fails
- [ ] Production ready: No memory leaks after 100 requests

## Implementation Notes

### Critical Path
1. Docker environment is foundation - must work before anything else
2. NeMo configuration must be tuned for available GPU memory
3. Alignment algorithm determines quality of speaker attribution
4. Memory management prevents production crashes

### Blackwell GPU Considerations
- Use CUDA 12.4 Docker image for best compatibility
- Enable TF32 for better performance: `torch.backends.cuda.matmul.allow_tf32 = True`
- Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
- Monitor for cuDNN warnings but continue if functional

### Migration Strategy
1. Deploy as separate service first (port 8768)
2. Run A/B testing comparing PyAnnote vs NeMo results
3. Switch primary service after validation
4. Keep PyAnnote code as fallback option

### Known Limitations
- Overlapping speech: NeMo assigns to dominant speaker only
- Real-time processing: Not supported, batch mode only
- Maximum speakers: Limited to 8 by default configuration
- Audio format: Requires 16kHz mono WAV internally

### Performance Optimization Tips
- Use titanet_large for best accuracy vs titanet_base for speed
- Reduce multiscale windows for faster processing
- Enable VAD filtering to skip silence
- Cache NeMo models after first load
- Use batch processing for multiple files

### Debugging Commands
```bash
# Check NeMo model downloads
ls -la ~/.cache/torch/hub/

# Monitor GPU usage during processing
watch -n 0.5 nvidia-smi

# Check RTTM output format
cat nemo_outputs/pred_rttms/*.rttm

# Verify manifest creation
cat nemo_outputs/input_manifest.json
```

## Confidence Score: 8/10

The implementation is well-researched with clear patterns from production systems. The main uncertainty is Blackwell GPU compatibility, which is mitigated by using Docker with CUDA 12.4. The architecture maintains API compatibility while upgrading the diarization backend, minimizing risk to existing clients.