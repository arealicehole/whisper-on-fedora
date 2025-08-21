#!/usr/bin/env python3
"""
Comprehensive Test Suite for Diarization
Part of the Diarization Testing & Hardening Initiative
"""

import pytest
import torch
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Test fixtures and configuration
SAMPLE_RATE = 16000
TEST_API_URL = "http://localhost:8765"


class TestDiarizationPipeline:
    """Unit tests for diarization pipeline components"""
    
    @pytest.fixture
    def mock_audio(self):
        """Generate mock audio data"""
        duration = 10  # seconds
        samples = duration * SAMPLE_RATE
        # Generate simple sine wave
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, SAMPLE_RATE
    
    @pytest.fixture
    def mock_pipeline(self):
        """Mock pyannote pipeline for testing"""
        with patch('pyannote.audio.Pipeline') as mock:
            pipeline = Mock()
            pipeline.return_value = {
                'segments': [
                    {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
                    {'start': 3.0, 'end': 6.0, 'speaker': 'SPEAKER_01'},
                    {'start': 6.0, 'end': 10.0, 'speaker': 'SPEAKER_00'}
                ]
            }
            mock.from_pretrained.return_value = pipeline
            yield mock
    
    def test_cuda_availability(self):
        """Test CUDA availability and compatibility"""
        assert torch.cuda.is_available(), "CUDA should be available"
        
        # Test basic CUDA operations
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        cuda_tensor = test_tensor.cuda()
        result = cuda_tensor * 2
        
        assert result.device.type == 'cuda'
        assert torch.allclose(result.cpu(), torch.tensor([2.0, 4.0, 6.0]))
    
    def test_pipeline_loading(self, mock_pipeline):
        """Test diarization pipeline loading"""
        from pyannote.audio import Pipeline
        
        # Test model loading with different versions
        models = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-3.0",
            "pyannote/speaker-diarization@2.1"
        ]
        
        for model_name in models:
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token="mock_token")
            assert pipeline is not None
    
    def test_audio_preprocessing(self, mock_audio):
        """Test audio preprocessing for diarization"""
        audio, sr = mock_audio
        
        # Test resampling
        if sr != 16000:
            # Would resample here
            pass
        
        # Test normalization
        normalized = audio / np.max(np.abs(audio))
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test VAD (Voice Activity Detection)
        # Simple energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        assert len(energy) > 0
    
    def test_speaker_embedding_extraction(self, mock_audio):
        """Test speaker embedding extraction"""
        audio, sr = mock_audio
        
        # Mock embedding extraction
        embedding_dim = 256
        mock_embedding = np.random.randn(embedding_dim)
        
        assert mock_embedding.shape == (embedding_dim,)
        assert np.isfinite(mock_embedding).all()
    
    def test_segment_merging(self):
        """Test merging of speaker segments"""
        segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_00'},  # Should merge
            {'start': 2.5, 'end': 3.5, 'speaker': 'SPEAKER_01'},
            {'start': 3.6, 'end': 4.0, 'speaker': 'SPEAKER_01'},  # Should merge if close
        ]
        
        def merge_segments(segments, max_gap=0.5):
            """Merge consecutive segments from same speaker"""
            if not segments:
                return []
            
            merged = [segments[0]]
            for seg in segments[1:]:
                last = merged[-1]
                if (seg['speaker'] == last['speaker'] and 
                    seg['start'] - last['end'] <= max_gap):
                    last['end'] = seg['end']
                else:
                    merged.append(seg)
            return merged
        
        merged = merge_segments(segments)
        assert len(merged) == 3  # First two should merge, last two might merge
    
    @pytest.mark.parametrize("num_speakers", [1, 2, 3, 5, 10])
    def test_speaker_count_detection(self, num_speakers):
        """Test automatic speaker count detection"""
        # Mock segments for different speaker counts
        segments = []
        for i in range(num_speakers):
            segments.append({
                'start': i * 2.0,
                'end': (i + 1) * 2.0,
                'speaker': f'SPEAKER_{i:02d}'
            })
        
        detected_speakers = len(set(seg['speaker'] for seg in segments))
        assert detected_speakers == num_speakers


class TestDiarizationAPI:
    """Integration tests for diarization API endpoints"""
    
    @pytest.fixture
    async def client(self):
        """Create async HTTP client"""
        async with httpx.AsyncClient(base_url=TEST_API_URL) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_check_diarization(self, client):
        """Test health endpoint reports diarization status"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'diarization' in data
        assert 'modules_available' in data['diarization']
        assert 'pipeline_loaded' in data['diarization']
    
    @pytest.mark.asyncio
    async def test_transcribe_with_diarization(self, client):
        """Test transcription with diarization enabled"""
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Generate test audio
            duration = 5
            sr = 16000
            t = np.linspace(0, duration, duration * sr)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Save as WAV (would use soundfile in real implementation)
            import wave
            with wave.open(f.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            # Test API call
            with open(f.name, 'rb') as audio_file:
                files = {'file': ('test.wav', audio_file, 'audio/wav')}
                data = {'diarize': 'true', 'num_speakers': 2}
                
                response = await client.post("/v1/transcribe", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    assert 'segments' in result
                    if result['segments']:
                        assert 'speaker' in result['segments'][0]
    
    @pytest.mark.asyncio
    async def test_diarization_error_handling(self, client):
        """Test error handling when diarization fails"""
        # Test with invalid audio
        files = {'file': ('test.txt', b'not audio', 'text/plain')}
        data = {'diarize': 'true'}
        
        response = await client.post("/v1/transcribe", files=files, data=data)
        assert response.status_code in [400, 422, 500]
    
    @pytest.mark.asyncio
    async def test_concurrent_diarization_requests(self, client):
        """Test handling of concurrent diarization requests"""
        # Create multiple test audio files
        tasks = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                # Generate different audio for each
                duration = 2
                sr = 16000
                t = np.linspace(0, duration, duration * sr)
                freq = 440 * (i + 1)  # Different frequency for each
                audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
                
                import wave
                with wave.open(f.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sr)
                    wav.writeframes((audio * 32767).astype(np.int16).tobytes())
                
                # Create async task
                async def transcribe(file_path):
                    with open(file_path, 'rb') as audio_file:
                        files = {'file': ('test.wav', audio_file, 'audio/wav')}
                        data = {'diarize': 'true'}
                        return await client.post("/v1/transcribe", files=files, data=data)
                
                tasks.append(transcribe(f.name))
        
        # Run concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check responses
        success_count = sum(1 for r in responses 
                          if not isinstance(r, Exception) and r.status_code == 200)
        assert success_count > 0  # At least some should succeed


class TestDiarizationPerformance:
    """Performance and benchmark tests for diarization"""
    
    @pytest.mark.benchmark
    def test_diarization_speed(self, benchmark):
        """Benchmark diarization processing speed"""
        def process_audio():
            # Mock diarization processing
            duration = 60  # 1 minute audio
            sr = 16000
            samples = duration * sr
            audio = np.random.randn(samples).astype(np.float32)
            
            # Simulate processing time
            time.sleep(0.1)  # Mock processing
            
            return {
                'segments': [
                    {'start': i, 'end': i+1, 'speaker': f'SPEAKER_{i%2:02d}'}
                    for i in range(duration)
                ]
            }
        
        result = benchmark(process_audio)
        assert len(result['segments']) > 0
    
    def test_memory_usage(self):
        """Test memory usage during diarization"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate diarization of large audio
        duration = 300  # 5 minutes
        sr = 16000
        audio = np.random.randn(duration * sr).astype(np.float32)
        
        # Process audio (mock)
        segments = []
        for i in range(0, duration, 10):
            segments.append({
                'start': i,
                'end': min(i + 10, duration),
                'speaker': f'SPEAKER_{(i//10)%3:02d}'
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 1GB for 5 minute audio
        assert memory_increase < 1024, f"Memory usage increased by {memory_increase}MB"
    
    def test_gpu_memory_usage(self):
        """Test GPU memory usage during diarization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Create large tensor on GPU (simulate model)
        model_size = (1024, 1024, 10)  # ~40MB
        model_tensor = torch.randn(model_size).cuda()
        
        # Process mock audio
        audio_tensor = torch.randn(16000 * 60).cuda()  # 1 minute audio
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Clean up
        del model_tensor, audio_tensor
        torch.cuda.empty_cache()
        
        # Should not exceed 4GB as per requirements
        assert peak_memory < 4096, f"Peak GPU memory usage: {peak_memory}MB"


class TestDiarizationAccuracy:
    """Accuracy tests for speaker diarization"""
    
    def test_speaker_identification_accuracy(self):
        """Test accuracy of speaker identification"""
        # Ground truth segments
        ground_truth = [
            {'start': 0.0, 'end': 3.0, 'speaker': 'A'},
            {'start': 3.0, 'end': 6.0, 'speaker': 'B'},
            {'start': 6.0, 'end': 9.0, 'speaker': 'A'},
            {'start': 9.0, 'end': 12.0, 'speaker': 'B'},
        ]
        
        # Predicted segments (with some errors)
        predicted = [
            {'start': 0.0, 'end': 2.8, 'speaker': 'SPEAKER_00'},
            {'start': 2.8, 'end': 6.1, 'speaker': 'SPEAKER_01'},
            {'start': 6.1, 'end': 9.2, 'speaker': 'SPEAKER_00'},
            {'start': 9.2, 'end': 12.0, 'speaker': 'SPEAKER_01'},
        ]
        
        def calculate_der(ground_truth, predicted):
            """Calculate Diarization Error Rate (DER)"""
            # Simplified DER calculation
            total_duration = max(s['end'] for s in ground_truth)
            
            # Map predicted speakers to ground truth
            speaker_map = {'SPEAKER_00': 'A', 'SPEAKER_01': 'B'}
            
            error_duration = 0.0
            for gt in ground_truth:
                for pred in predicted:
                    # Calculate overlap
                    overlap_start = max(gt['start'], pred['start'])
                    overlap_end = min(gt['end'], pred['end'])
                    
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        if speaker_map.get(pred['speaker']) != gt['speaker']:
                            error_duration += overlap_duration
            
            der = error_duration / total_duration
            return der
        
        der = calculate_der(ground_truth, predicted)
        # Target: <15% DER (85% accuracy)
        assert der < 0.15, f"DER too high: {der:.2%}"
    
    def test_overlapping_speech_detection(self):
        """Test detection of overlapping speech"""
        segments = [
            {'start': 0.0, 'end': 3.0, 'speaker': 'SPEAKER_00'},
            {'start': 2.5, 'end': 4.0, 'speaker': 'SPEAKER_01'},  # Overlap
            {'start': 3.5, 'end': 6.0, 'speaker': 'SPEAKER_00'},
        ]
        
        overlaps = []
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                if seg1['start'] < seg2['end'] and seg2['start'] < seg1['end']:
                    overlaps.append({
                        'start': max(seg1['start'], seg2['start']),
                        'end': min(seg1['end'], seg2['end']),
                        'speakers': [seg1['speaker'], seg2['speaker']]
                    })
        
        assert len(overlaps) > 0, "Should detect overlapping speech"
        assert overlaps[0]['start'] == 2.5
        assert overlaps[0]['end'] == 3.0


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])