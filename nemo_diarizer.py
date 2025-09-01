#\!/usr/bin/env python3
"""
Sortformer v1 Diarization Module for Whisper API
Uses NVIDIA NeMo's Sortformer v1 model which is stable and well-tested
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    NEMO_AVAILABLE = True
except ImportError as e:
    logger.error(f"NeMo not available: {e}")
    NEMO_AVAILABLE = False

class NeMoDiarizer:
    def __init__(self, device: str = "cuda", gpu_memory_gb: int = 16):
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo toolkit is not installed")
        
        self.device = device
        self.gpu_memory_gb = gpu_memory_gb
        self.model = None
        
        # Try to load Sortformer v1 model (stable version)
        try:
            logger.info("Loading Sortformer v1 diarization model...")
            
            # Set HF token from environment
            hf_token = os.environ.get('HF_TOKEN', '')
            if hf_token and hf_token.startswith('hf_'):
                os.environ['HF_HOME'] = '/workspace/models'
                os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
                logger.info(f"Using HF token: {hf_token[:10]}...")
            
            model_loaded = False
            
            # Option 1: Try local cached model
            model_path = "/workspace/models/diar_sortformer_4spk-v1.nemo"
            if os.path.exists(model_path):
                try:
                    self.model = SortformerEncLabelModel.restore_from(
                        restore_path=model_path,
                        map_location=torch.device(device),
                        strict=False
                    )
                    logger.info(f"Loaded Sortformer v1 from cache: {model_path}")
                    model_loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")
            
            # Option 2: Download from HuggingFace with token
            if not model_loaded and hf_token:
                try:
                    logger.info("Downloading Sortformer v1 from HuggingFace...")
                    # Use the standard v1 model which is more stable
                    self.model = SortformerEncLabelModel.from_pretrained(
                        "nvidia/diar_sortformer_4spk-v1"
                    )
                    logger.info("Successfully loaded Sortformer v1 from HuggingFace")
                    model_loaded = True
                    
                    # Save to cache for next time
                    try:
                        os.makedirs("/workspace/models", exist_ok=True)
                        self.model.save_to(model_path)
                        logger.info(f"Cached model to {model_path}")
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Failed to download from HF: {e}")
            
            if model_loaded and self.model is not None:
                # Move model to GPU if available
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("Sortformer v1 model moved to GPU")
                
                # Switch to evaluation mode
                self.model.eval()
                logger.info("Sortformer v1 ready for inference")
            else:
                logger.error("Failed to load Sortformer v1 model")
                self.model = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Sortformer v1: {e}")
            self.model = None
        
        logger.info(f"NeMoDiarizer initialized on {device}")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """
        Perform speaker diarization using Sortformer v1 model.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (auto-detect if None)
        
        Returns:
            List of speaker segments with format:
            [{'start': float, 'end': float, 'speaker': str}, ...]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            if self.model is None:
                logger.warning("No diarization model available, returning empty segments")
                return []
            
            logger.info(f"Starting Sortformer v1 diarization for {audio_path}")
            
            # Ensure model is on GPU
            if self.device == "cuda" and torch.cuda.is_available():
                if not next(self.model.parameters()).is_cuda:
                    self.model = self.model.cuda()
                    logger.info("Model moved to GPU for inference")
            
            # Run diarization with Sortformer v1
            try:
                # Use batch_size=1 for stability
                pred_list = self.model.diarize(audio=audio_path, batch_size=1)
                logger.info(f"Diarization completed, got {len(pred_list) if pred_list else 0} results")
            except Exception as e:
                logger.error(f"Diarization inference failed: {e}")
                return []
            
            if not pred_list or len(pred_list) == 0:
                logger.warning("Sortformer returned no results")
                return []
            
            # Parse the output format from Sortformer
            segments = []
            result = pred_list[0] if isinstance(pred_list, list) else pred_list
            
            # Handle list of segments
            if isinstance(result, list):
                for segment_str in result:
                    try:
                        if isinstance(segment_str, str):
                            # Parse string format: "start end speaker"
                            parts = segment_str.strip().split()
                            if len(parts) >= 3:
                                start = float(parts[0])
                                end = float(parts[1])
                                # Use speaker label from model (e.g., "speaker_0", "speaker_1")
                                speaker = parts[2] if len(parts) > 2 else "SPEAKER_00"
                                
                                # Convert to consistent format
                                if speaker.startswith("speaker_"):
                                    speaker_num = speaker.split("_")[1]
                                    speaker = f"SPEAKER_{int(speaker_num):02d}"
                                elif not speaker.startswith("SPEAKER_"):
                                    speaker = f"SPEAKER_00"
                                
                                segments.append({
                                    "start": start,
                                    "end": end,
                                    "speaker": speaker
                                })
                        elif isinstance(segment_str, dict):
                            segments.append({
                                "start": segment_str.get("start", 0),
                                "end": segment_str.get("end", 0),
                                "speaker": segment_str.get("speaker", "SPEAKER_00")
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse segment: {e}")
                        continue
            
            logger.info(f"Diarization complete: {len(segments)} speaker segments found")
            
            # Log first few segments for debugging
            if segments:
                logger.info(f"Sample segments: {segments[:3]}")
            
            # Log GPU memory usage
            if self.device == "cuda" and torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**2
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.info(f"GPU Memory: Current={mem_used:.1f}MB, Peak={peak_mem:.1f}MB")
            
            return segments
            
        except Exception as e:
            logger.error(f"Sortformer v1 diarization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

def align_transcription_with_speakers(transcription_segments, speaker_segments):
    """Align whisper transcription with NeMo speaker segments."""
    if not speaker_segments:
        # No diarization available
        logger.warning("No speaker segments available for alignment")
        return [{"id": i+1, **seg, "speaker": "Unknown"} for i, seg in enumerate(transcription_segments)]
    
    logger.info(f"Aligning {len(transcription_segments)} transcription segments with {len(speaker_segments)} speaker segments")
    
    aligned = []
    for i, trans_seg in enumerate(transcription_segments):
        start = trans_seg.get("start", 0)
        end = trans_seg.get("end", start + 1)
        
        # Find overlapping speaker with maximum overlap
        best_speaker = "Unknown"
        max_overlap = 0
        
        for spk_seg in speaker_segments:
            overlap_start = max(start, spk_seg["start"])
            overlap_end = min(end, spk_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = spk_seg["speaker"]
        
        # If we found ANY overlap, use that speaker
        # Don't require a minimum threshold - any overlap counts
        if max_overlap > 0:
            logger.debug(f"Segment {i}: {start:.2f}-{end:.2f} assigned to {best_speaker} (overlap: {max_overlap:.2f}s)")
        else:
            logger.debug(f"Segment {i}: {start:.2f}-{end:.2f} has no speaker overlap")
            best_speaker = "Unknown"
        
        aligned.append({
            "id": i + 1,
            **trans_seg,
            "speaker": best_speaker
        })
    
    # Log summary
    speakers_found = set(seg["speaker"] for seg in aligned if seg["speaker"] != "Unknown")
    unknown_count = sum(1 for seg in aligned if seg["speaker"] == "Unknown")
    logger.info(f"Alignment complete: {len(speakers_found)} speakers found, {unknown_count} segments unknown")
    
    return aligned
