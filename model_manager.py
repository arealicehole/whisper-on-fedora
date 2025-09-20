#!/usr/bin/env python3
"""
Whisper Model Manager for Dynamic Local Model Loading
Manages multiple local Whisper models with memory optimization
"""

import os
import gc
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import torch

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str
    device: str = "cuda"
    compute_type: str = "float16"
    max_loaded_models: int = 2
    models_directory: str = "/workspace/models"


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    path: str
    loaded_at: float
    last_used: float
    memory_usage_mb: Optional[float] = None
    model_instance: Optional[WhisperModel] = None


class ModelNotFoundError(Exception):
    """Raised when requested model is not available locally"""
    def __init__(self, model_name: str, available_models: List[str]):
        self.model_name = model_name
        self.available_models = available_models
        super().__init__(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )


class ModelValidationError(Exception):
    """Raised when model validation fails"""
    pass


class ModelLoadingError(Exception):
    """Raised when model fails to load"""
    pass


class WhisperModelManager:
    """
    Manages multiple local Whisper models with dynamic loading and memory optimization.
    
    Features:
    - Local model discovery and validation
    - Dynamic model loading/unloading
    - LRU-based memory management
    - GPU memory monitoring
    - Thread-safe model access
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model manager"""
        self.config = config or ModelConfig(
            name="default",
            models_directory=os.environ.get("MODELS_DIRECTORY", "/workspace/models"),
            max_loaded_models=int(os.environ.get("MAX_LOADED_MODELS", "2"))
        )
        
        self.models_directory = Path(self.config.models_directory)
        self.loaded_models: OrderedDict[str, ModelInfo] = OrderedDict()
        self.max_loaded_models = self.config.max_loaded_models
        
        # Validate faster-whisper availability
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper is not installed")
        
        # Create models directory if it doesn't exist
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"WhisperModelManager initialized: {self.models_directory}")
        logger.info(f"Max loaded models: {self.max_loaded_models}")
    
    def discover_available_models(self) -> List[str]:
        """
        Discover available local models by scanning the models directory.
        Supports both HuggingFace hub format and CT2 format.
        
        Returns:
            List of model names
        """
        available_models = []
        
        if not self.models_directory.exists():
            logger.warning(f"Models directory does not exist: {self.models_directory}")
            return available_models
        
        # Scan for both formats:
        # 1. HuggingFace hub format: models--Systran--faster-whisper-{name}
        # 2. CT2 format: {name}-ct2
        for item in self.models_directory.iterdir():
            if not item.is_dir():
                continue
                
            model_name = None
            model_path = None
            
            # Check HuggingFace hub format
            if item.name.startswith('models--Systran--faster-whisper-'):
                model_name = item.name.replace('models--Systran--faster-whisper-', '')
                # Find the actual model files in snapshots subdirectory
                snapshots_dir = item / "snapshots"
                if snapshots_dir.exists():
                    for snapshot in snapshots_dir.iterdir():
                        if snapshot.is_dir():
                            model_path = snapshot
                            break
            
            # Check CT2 format
            elif item.name.endswith('-ct2'):
                model_name = item.name[:-4]  # Remove -ct2 suffix
                model_path = item
            
            # Validate and add model if found
            if model_name and model_path and self._validate_model_directory(model_path):
                available_models.append(model_name)
            elif model_name:
                logger.warning(f"Invalid model directory: {item}")
        
        logger.info(f"Discovered {len(available_models)} models: {available_models}")
        return sorted(available_models)
    
    def _validate_model_directory(self, model_path: Path) -> bool:
        """
        Validate that a model directory has required files.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if valid, False otherwise
        """
        required_files = ["config.json"]  # Minimum required file
        optional_files = ["model.bin", "model.ctranslate2", "vocabulary.txt", "tokenizer.json"]
        
        # Check for required files
        for file in required_files:
            if not (model_path / file).exists():
                return False
        
        # Check for at least one model file
        has_model_file = any((model_path / file).exists() for file in optional_files)
        return has_model_file
    
    def validate_model_availability(self, model_name: str) -> None:
        """
        Validate that a model is available and loadable.
        
        Args:
            model_name: Name of the model to validate
            
        Raises:
            ModelNotFoundError: If model is not found
            ModelValidationError: If model validation fails
        """
        available_models = self.discover_available_models()
        
        if model_name not in available_models:
            raise ModelNotFoundError(model_name, available_models)
        
        # Get actual model path using the updated method
        model_path_str = self.get_model_path(model_name)
        model_path = Path(model_path_str)
        
        if not self._validate_model_directory(model_path):
            raise ModelValidationError(
                f"Model '{model_name}' directory is invalid or corrupted: {model_path}"
            )
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get the filesystem path for a model.
        Supports both HuggingFace hub format and CT2 format.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Absolute path to model directory
        """
        # Check HuggingFace hub format first
        hf_path = self.models_directory / f"models--Systran--faster-whisper-{model_name}"
        if hf_path.exists():
            snapshots_dir = hf_path / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        return str(snapshot)
        
        # Fallback to CT2 format
        ct2_path = self.models_directory / f"{model_name}-ct2"
        return str(ct2_path)
    
    def _cleanup_unused_models(self) -> None:
        """
        Remove least recently used models to make room for new ones.
        Uses LRU eviction strategy.
        """
        while len(self.loaded_models) >= self.max_loaded_models:
            # Get least recently used model (first in OrderedDict)
            lru_model_name = next(iter(self.loaded_models))
            lru_model_info = self.loaded_models[lru_model_name]
            
            logger.info(f"Evicting LRU model: {lru_model_name}")
            
            # Remove model instance
            if lru_model_info.model_instance:
                del lru_model_info.model_instance
            
            # Remove from loaded models
            del self.loaded_models[lru_model_name]
            
            # Force garbage collection and GPU cache cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {lru_model_name} evicted from memory")
    
    def load_model(
        self, 
        model_name: str, 
        device: Optional[str] = None, 
        compute_type: Optional[str] = None
    ) -> WhisperModel:
        """
        Load a model into memory or return existing loaded model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on (default: cuda)
            compute_type: Compute type for model (default: float16)
            
        Returns:
            Loaded WhisperModel instance
            
        Raises:
            ModelNotFoundError: If model is not found
            ModelLoadingError: If model fails to load
        """
        # Validate model exists
        self.validate_model_availability(model_name)
        
        # Check if model is already loaded
        if model_name in self.loaded_models:
            model_info = self.loaded_models[model_name]
            model_info.last_used = time.time()
            
            # Move to end (most recently used)
            self.loaded_models.move_to_end(model_name)
            
            logger.debug(f"Using cached model: {model_name}")
            return model_info.model_instance
        
        # Cleanup unused models if needed
        self._cleanup_unused_models()
        
        # Load new model
        device = device or self.config.device
        compute_type = compute_type or self.config.compute_type
        model_path = self.get_model_path(model_name)
        
        logger.info(f"Loading model: {model_name} from {model_path}")
        logger.info(f"Device: {device}, Compute type: {compute_type}")
        
        try:
            # Monitor memory before loading
            mem_before = self.monitor_memory_usage()
            
            # Load model with local_files_only to prevent HF downloads
            model_instance = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                local_files_only=True  # Prevent HuggingFace downloads
            )
            
            # Monitor memory after loading
            mem_after = self.monitor_memory_usage()
            memory_used = mem_after.get("allocated_gb", 0) - mem_before.get("allocated_gb", 0)
            
            # Create model info
            current_time = time.time()
            model_info = ModelInfo(
                name=model_name,
                path=model_path,
                loaded_at=current_time,
                last_used=current_time,
                memory_usage_mb=memory_used * 1024,  # Convert GB to MB
                model_instance=model_instance
            )
            
            # Add to loaded models (at end - most recently used)
            self.loaded_models[model_name] = model_info
            
            logger.info(f"Model {model_name} loaded successfully")
            logger.info(f"Memory usage: {memory_used*1024:.1f} MB")
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Failed to load model '{model_name}': {str(e)}")
    
    def get_model(self, model_name: str) -> WhisperModel:
        """
        Get a model instance, loading it if necessary.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            WhisperModel instance
        """
        return self.load_model(model_name)
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not loaded
        """
        if model_name not in self.loaded_models:
            return False
        
        model_info = self.loaded_models[model_name]
        
        # Remove model instance
        if model_info.model_instance:
            del model_info.model_instance
        
        # Remove from loaded models
        del self.loaded_models[model_name]
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model {model_name} unloaded from memory")
        return True
    
    def unload_all_models(self) -> None:
        """Unload all models from memory"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        logger.info("All models unloaded from memory")
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor GPU memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "usage_percent": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_percent = (allocated / total * 100) if total > 0 else 0
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "usage_percent": usage_percent
        }
    
    def get_loaded_models_info(self) -> Dict[str, Dict]:
        """
        Get information about currently loaded models.
        
        Returns:
            Dictionary mapping model names to their info
        """
        result = {}
        for name, info in self.loaded_models.items():
            result[name] = {
                "name": info.name,
                "path": info.path,
                "loaded_at": info.loaded_at,
                "last_used": info.last_used,
                "memory_usage_mb": info.memory_usage_mb,
                "is_loaded": info.model_instance is not None
            }
        return result
    
    def get_status(self) -> Dict:
        """
        Get comprehensive status of the model manager.
        
        Returns:
            Status dictionary with all relevant information
        """
        return {
            "models_directory": str(self.models_directory),
            "max_loaded_models": self.max_loaded_models,
            "available_models": self.discover_available_models(),
            "loaded_models": list(self.loaded_models.keys()),
            "loaded_models_info": self.get_loaded_models_info(),
            "memory_usage": self.monitor_memory_usage(),
            "faster_whisper_available": FASTER_WHISPER_AVAILABLE
        }


def main():
    """Standalone model manager testing utility"""
    print("🔍 Whisper Model Manager Test")
    print("=" * 50)
    
    try:
        # Initialize model manager
        manager = WhisperModelManager()
        
        # Discover models
        available = manager.discover_available_models()
        print(f"Available models: {available}")
        
        # Show status
        status = manager.get_status()
        print(f"Memory usage: {status['memory_usage']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())