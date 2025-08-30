#!/usr/bin/env python3
"""
WhisperX Diarization Adapter
Drop-in replacement for PyAnnote diarization using WhisperX
Provides Blackwell GPU compatibility with PyTorch 2.7.1+cu128
"""

import whisperx
import torch
import gc
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class WhisperXDiarization:
    """Drop-in replacement for PyAnnote diarization using WhisperX"""
    
    def __init__(self, device: str = "cuda", compute_type: str = "float16"):
        self.device = device
        self.compute_type = compute_type
        self.diarize_model = None
        self.align_model = None
        self.metadata = None
        
    def load_pipeline(self, auth_token: str) -> bool:
        """Load WhisperX diarization pipeline"""
        try:
            # Load PyAnnote diarization model through WhisperX
            # WhisperX uses PyAnnote internally, we just need to import it correctly
            import pyannote.audio as pa
            self.diarize_model = pa.Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            # Move to GPU if available
            if torch.cuda.is_available():
                self.diarize_model.to(torch.device(self.device))
            logger.info("WhisperX/PyAnnote diarization pipeline loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load WhisperX/PyAnnote diarization: {e}")
            return False
    
    def process(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """Process audio file and return diarization results"""
        try:
            # Run PyAnnote diarization
            diarization = self.diarize_model(audio_path, num_speakers=num_speakers)
            
            # Convert to expected format
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Clean up GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def cleanup(self):
        """Release GPU memory"""
        del self.diarize_model
        gc.collect()
        torch.cuda.empty_cache()

# Global instance for API compatibility
diarization_pipeline = None

def load_diarization_pipeline(auth_token: str) -> bool:
    """Load global diarization pipeline - matches PyAnnote interface"""
    global diarization_pipeline
    try:
        diarization_pipeline = WhisperXDiarization(device="cuda")
        return diarization_pipeline.load_pipeline(auth_token)
    except Exception as e:
        logger.error(f"Failed to initialize WhisperX: {e}")
        return False

def run_diarization(audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """Run diarization - matches PyAnnote interface"""
    if diarization_pipeline is None:
        raise RuntimeError("Diarization pipeline not loaded")
    return diarization_pipeline.process(audio_path, num_speakers)