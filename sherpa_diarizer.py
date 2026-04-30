#!/usr/bin/env python3
import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import sherpa_onnx
from scipy.io import wavfile
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SherpaDiarizer:
    def __init__(self, device: str = "cuda", models_dir: str = "/workspace/models-onnx"):
        self.device = device
        self.models_dir = Path(models_dir)
        self.segmentation_model = self.models_dir / "pyannote-segmentation-3-0.onnx"
        self.embedding_model = self.models_dir / "3d-speaker-campplus.onnx"
        
        # Hyperparameters for noisy audio
        self.min_duration_on = 0.6
        self.min_duration_off = 1.0
        self.threshold = 0.8
        
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model)
                ),
                num_threads=2,
                debug=False,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model),
                num_threads=2,
                debug=False,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=self.threshold),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        
        if device == "cuda":
            config.segmentation.provider = "cuda"
            config.embedding.provider = "cuda"
            
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        logger.info(f"Sherpa-ONNX Diarizer ready (Thresh={self.threshold}, MinOn={self.min_duration_on})")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        wav_path = audio_path + ".sherpa.wav"
        try:
            logger.info(f"Preprocessing {audio_path} with Highpass + Loudnorm...")
            # Battle-tested FFmpeg chain for telephony
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path, 
                "-af", "highpass=f=100, loudnorm=I=-16", 
                "-ar", "16000", "-ac", "1", wav_path
            ], check=True, capture_output=True)
            
            sample_rate, samples = wavfile.read(wav_path)
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32) / 32768.0
            
            logger.info("Running Sherpa-ONNX diarization...")
            result = self.sd.process(samples)
            segments = result.sort_by_start_time()
            logger.info(f"Sherpa segments found: {len(segments)}")
            
            output = []
            for seg in segments:
                output.append({
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": f"SPEAKER_{seg.speaker:02d}"
                })
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return output
        except Exception as e:
            logger.error(f"Sherpa diarization failed: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return []

def align_transcription_with_speakers(transcription_segments, speaker_segments):
    if not speaker_segments:
        return [{"id": i+1, **seg, "speaker": "Unknown"} for i, seg in enumerate(transcription_segments)]
    
    aligned = []
    for i, trans_seg in enumerate(transcription_segments):
        start, end = trans_seg.get("start", 0), trans_seg.get("end", trans_seg.get("start", 0) + 1)
        best_speaker, max_overlap = "Unknown", 0
        for spk_seg in speaker_segments:
            overlap = max(0, min(end, spk_seg["end"]) - max(start, spk_seg["start"]))
            if overlap > max_overlap:
                max_overlap, best_speaker = overlap, spk_seg["speaker"]
        aligned.append({"id": i + 1, **trans_seg, "speaker": best_speaker})
    return aligned
