#!/usr/bin/env python3
"""
Enhanced Diarization Handler with Error Recovery
Part of the Diarization Testing & Hardening Initiative
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from functools import lru_cache
from contextlib import contextmanager

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiarizationError(Exception):
    """Base exception for diarization errors"""
    pass


class ModelLoadError(DiarizationError):
    """Failed to load pyannote model"""
    pass


class AudioProcessingError(DiarizationError):
    """Failed to process audio for diarization"""
    pass


class SpeakerDetectionError(DiarizationError):
    """Failed to detect speakers"""
    pass


class CUDAError(DiarizationError):
    """CUDA-related errors"""
    pass


class DiarizationStatus(Enum):
    """Status of diarization pipeline"""
    NOT_INITIALIZED = "not_initialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    PROCESSING = "processing"


@dataclass
class DiarizationConfig:
    """Configuration for diarization"""
    model_name: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    embedding_cache_size: int = 1000
    max_speakers: int = 10
    min_speakers: int = 1
    speaker_overlap_threshold: float = 0.5
    vad_threshold: float = 0.5
    min_segment_duration: float = 0.5
    merge_segments_gap: float = 0.5
    enable_cache: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0


class SpeakerEmbeddingCache:
    """LRU cache for speaker embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, embedding: np.ndarray):
        """Store embedding in cache"""
        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = embedding
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


class EnhancedDiarizationHandler:
    """Enhanced diarization handler with error recovery and optimization"""
    
    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self.status = DiarizationStatus.NOT_INITIALIZED
        self.pipeline = None
        self.error_message = None
        self.embedding_cache = SpeakerEmbeddingCache(self.config.embedding_cache_size)
        self.performance_metrics = {
            'total_processed': 0,
            'total_errors': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def initialize(self, token: Optional[str] = None) -> bool:
        """Initialize diarization pipeline with error recovery"""
        self.status = DiarizationStatus.LOADING
        
        # Try to import required modules
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            self.error_message = f"Failed to import pyannote.audio: {e}"
            self.status = DiarizationStatus.ERROR
            logger.error(self.error_message)
            return False
        
        # Check CUDA compatibility
        if self.config.device == "cuda" and not self._check_cuda_compatibility():
            logger.warning("CUDA compatibility check failed, falling back to CPU")
            self.config.device = "cpu"
        
        # Try loading different model versions
        models_to_try = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-3.0",
            "pyannote/speaker-diarization@2.1"
        ]
        
        for attempt in range(self.config.retry_attempts):
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to load model: {model_name} (attempt {attempt + 1})")
                    
                    self.pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=token
                    )
                    
                    # Move to specified device
                    if self.config.device == "cuda":
                        self.pipeline.to(torch.device("cuda"))
                    
                    self.status = DiarizationStatus.READY
                    logger.info(f"Successfully loaded model: {model_name}")
                    return True
                    
                except Exception as e:
                    self.error_message = f"Failed to load {model_name}: {str(e)}"
                    logger.warning(self.error_message)
                    
                    # If CUDA error, try CPU
                    if "CUDA" in str(e) and self.config.device == "cuda":
                        logger.info("Retrying with CPU due to CUDA error")
                        self.config.device = "cpu"
                        continue
                    
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        self.status = DiarizationStatus.ERROR
        return False
    
    def _check_cuda_compatibility(self) -> bool:
        """Check if CUDA is properly configured"""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Test basic CUDA operations
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor * 2
            result.cpu()  # Ensure we can move back to CPU
            
            # Check for specific CUDA errors
            torch.cuda.synchronize()
            
            return True
        except RuntimeError as e:
            if "no kernel image" in str(e):
                logger.error("CUDA architecture mismatch detected")
                logger.info("Your GPU requires a different PyTorch build")
                logger.info("Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                logger.error(f"CUDA compatibility check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected CUDA error: {e}")
            return False
    
    @contextmanager
    def error_handler(self, operation: str):
        """Context manager for error handling"""
        try:
            yield
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise CUDAError(f"GPU out of memory during {operation}. Try reducing batch size.")
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise CUDAError(f"CUDA error during {operation}: {e}")
            raise AudioProcessingError(f"Runtime error during {operation}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during {operation}: {e}")
            logger.debug(traceback.format_exc())
            raise DiarizationError(f"Failed during {operation}: {e}")
    
    def process_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process audio with diarization"""
        
        if self.status != DiarizationStatus.READY:
            raise DiarizationError(f"Pipeline not ready. Status: {self.status}")
        
        start_time = time.time()
        self.status = DiarizationStatus.PROCESSING
        
        try:
            # Generate cache key
            cache_key = None
            if self.config.enable_cache:
                cache_key = self._generate_cache_key(audio_path, num_speakers)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.performance_metrics['cache_hits'] += 1
                    logger.info(f"Cache hit for {audio_path}")
                    return cached_result
                self.performance_metrics['cache_misses'] += 1
            
            # Process with error recovery
            with self.error_handler("audio processing"):
                result = self._process_with_retry(
                    audio_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers or self.config.min_speakers,
                    max_speakers=max_speakers or self.config.max_speakers
                )
            
            # Post-process results
            processed_result = self._post_process_results(result)
            
            # Cache results
            if cache_key and self.config.enable_cache:
                self._cache_result(cache_key, processed_result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=True)
            
            return processed_result
            
        except Exception as e:
            self._update_metrics(0, success=False)
            raise
        finally:
            self.status = DiarizationStatus.READY
    
    def _process_with_retry(
        self,
        audio_path: str,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int
    ) -> Any:
        """Process audio with retry logic"""
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Run diarization
                if num_speakers:
                    result = self.pipeline(
                        audio_path,
                        num_speakers=num_speakers
                    )
                else:
                    result = self.pipeline(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                
                return result
                
            except torch.cuda.OutOfMemoryError:
                # Clear cache and retry with smaller batch
                torch.cuda.empty_cache()
                self.config.batch_size = max(1, self.config.batch_size // 2)
                logger.warning(f"GPU OOM, reducing batch size to {self.config.batch_size}")
                last_error = "GPU out of memory"
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise DiarizationError(f"Failed after {self.config.retry_attempts} attempts: {last_error}")
    
    def _post_process_results(self, diarization) -> Dict[str, Any]:
        """Post-process diarization results"""
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                'start': round(turn.start, 2),
                'end': round(turn.end, 2),
                'speaker': speaker,
                'duration': round(turn.end - turn.start, 2)
            }
            
            # Filter short segments
            if segment['duration'] >= self.config.min_segment_duration:
                segments.append(segment)
        
        # Merge close segments from same speaker
        merged_segments = self._merge_segments(segments)
        
        # Add confidence scores (mock for now)
        for segment in merged_segments:
            segment['confidence'] = 0.85  # Would calculate real confidence
        
        return {
            'segments': merged_segments,
            'num_speakers': len(set(s['speaker'] for s in merged_segments)),
            'total_duration': max(s['end'] for s in merged_segments) if merged_segments else 0
        }
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge close segments from the same speaker"""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last = merged[-1]
            
            # Check if should merge
            if (segment['speaker'] == last['speaker'] and
                segment['start'] - last['end'] <= self.config.merge_segments_gap):
                # Merge segments
                last['end'] = segment['end']
                last['duration'] = round(last['end'] - last['start'], 2)
            else:
                merged.append(segment)
        
        return merged
    
    def _generate_cache_key(self, audio_path: str, num_speakers: Optional[int]) -> str:
        """Generate cache key for audio file"""
        key_parts = [
            audio_path,
            str(num_speakers) if num_speakers else "auto",
            str(os.path.getmtime(audio_path))
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached diarization result"""
        cache_dir = Path(".diarization_cache")
        cache_file = cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cache diarization result"""
        cache_dir = Path(".diarization_cache")
        cache_dir.mkdir(exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics['total_processed'] += 1
        
        if not success:
            self.performance_metrics['total_errors'] += 1
        else:
            # Update average processing time
            current_avg = self.performance_metrics['average_processing_time']
            total = self.performance_metrics['total_processed'] - self.performance_metrics['total_errors']
            
            if total > 0:
                self.performance_metrics['average_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and metrics"""
        return {
            'status': self.status.value,
            'device': self.config.device,
            'error_message': self.error_message,
            'metrics': self.performance_metrics,
            'cache_size': len(self.embedding_cache.cache),
            'config': {
                'model': self.config.model_name,
                'batch_size': self.config.batch_size,
                'max_speakers': self.config.max_speakers
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        
        self.embedding_cache.clear()
        self.status = DiarizationStatus.NOT_INITIALIZED


# Singleton instance
_diarization_handler: Optional[EnhancedDiarizationHandler] = None


def get_diarization_handler() -> EnhancedDiarizationHandler:
    """Get or create diarization handler instance"""
    global _diarization_handler
    
    if _diarization_handler is None:
        _diarization_handler = EnhancedDiarizationHandler()
    
    return _diarization_handler


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced handler
    handler = get_diarization_handler()
    
    # Try to initialize
    token = os.environ.get("HF_TOKEN")
    if handler.initialize(token):
        print("✓ Diarization handler initialized successfully")
        print(f"  Device: {handler.config.device}")
        print(f"  Status: {handler.status.value}")
    else:
        print("✗ Failed to initialize diarization handler")
        print(f"  Error: {handler.error_message}")
    
    # Get status
    status = handler.get_status()
    print("\nHandler Status:")
    print(json.dumps(status, indent=2))