# Speaker-to-Transcript Alignment Algorithm

## Overview

This document describes the algorithm for aligning NeMo's speaker diarization output (RTTM format) with faster-whisper's transcription segments to produce speaker-attributed transcripts.

## Current Implementation (Midpoint Algorithm)

The existing codebase uses a simple midpoint-based assignment:

```python
# From main.py
for segment in segments_list:
    mid_time = (segment["start"] + segment["end"]) / 2
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= mid_time <= turn.end:
            segment["speaker"] = speaker
            break
```

**Limitations:**
- Assigns entire segment to one speaker
- Poor accuracy at speaker boundaries
- Cannot handle overlapping speech

## Improved Algorithm: Weighted Intersection

### Core Algorithm

```python
def align_transcription_with_speakers(transcription_segments, speaker_segments):
    """
    Align transcription segments with speaker segments using weighted intersection.
    
    Args:
        transcription_segments: List of dicts with 'start', 'end', 'text' keys
        speaker_segments: List of dicts with 'start', 'end', 'speaker' keys from RTTM
    
    Returns:
        List of transcription segments with added 'speaker' field
    """
    
    for t_seg in transcription_segments:
        t_start, t_end = t_seg['start'], t_seg['end']
        
        # Calculate overlap with each speaker segment
        speaker_overlaps = {}
        
        for s_seg in speaker_segments:
            s_start, s_end = s_seg['start'], s_seg['end']
            
            # Calculate intersection
            overlap_start = max(t_start, s_start)
            overlap_end = min(t_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                speaker = s_seg['speaker']
                if speaker not in speaker_overlaps:
                    speaker_overlaps[speaker] = 0
                speaker_overlaps[speaker] += overlap_duration
        
        # Assign speaker with maximum overlap
        if speaker_overlaps:
            # Get speaker with maximum overlap duration
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            confidence = speaker_overlaps[best_speaker] / (t_end - t_start)
            
            t_seg['speaker'] = best_speaker
            t_seg['speaker_confidence'] = confidence
        else:
            t_seg['speaker'] = 'UNKNOWN'
            t_seg['speaker_confidence'] = 0.0
    
    return transcription_segments
```

### Word-Level Attribution

For more precise speaker attribution at word level:

```python
def align_words_with_speakers(words, speaker_segments):
    """
    Align word-level timestamps with speaker segments.
    
    Args:
        words: List of dicts with 'start', 'end', 'word' keys
        speaker_segments: List of speaker segments from RTTM
    
    Returns:
        Words with added 'speaker' field
    """
    
    # Build interval tree for efficient lookup (optional optimization)
    speaker_intervals = build_interval_tree(speaker_segments)
    
    for word in words:
        # Use word midpoint for faster processing
        word_midpoint = (word['start'] + word['end']) / 2
        
        # Find speaker at midpoint
        speaker = find_speaker_at_time(word_midpoint, speaker_segments)
        word['speaker'] = speaker if speaker else 'UNKNOWN'
    
    return words

def find_speaker_at_time(timestamp, speaker_segments):
    """Find speaker active at given timestamp."""
    for segment in speaker_segments:
        if segment['start'] <= timestamp <= segment['end']:
            return segment['speaker']
    return None
```

## Advanced: Multi-Speaker Segment Handling

For segments with multiple speakers:

```python
def handle_multi_speaker_segments(transcription_segments, speaker_segments, 
                                  overlap_threshold=0.3):
    """
    Split segments that have multiple speakers above threshold.
    
    Args:
        transcription_segments: Original transcription segments
        speaker_segments: Speaker diarization results
        overlap_threshold: Minimum overlap ratio to consider multiple speakers
    
    Returns:
        Potentially split segments with speaker attribution
    """
    
    result_segments = []
    
    for t_seg in transcription_segments:
        t_start, t_end = t_seg['start'], t_seg['end']
        t_duration = t_end - t_start
        
        # Find all overlapping speakers
        speaker_overlaps = calculate_all_overlaps(t_seg, speaker_segments)
        
        # Check if multiple speakers have significant overlap
        significant_speakers = {
            speaker: overlap 
            for speaker, overlap in speaker_overlaps.items() 
            if overlap / t_duration >= overlap_threshold
        }
        
        if len(significant_speakers) <= 1:
            # Single speaker or no significant overlap
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get) if speaker_overlaps else 'UNKNOWN'
            t_seg['speaker'] = best_speaker
            result_segments.append(t_seg)
        else:
            # Multiple speakers - split segment
            split_segments = split_by_speakers(t_seg, speaker_segments)
            result_segments.extend(split_segments)
    
    return result_segments

def split_by_speakers(segment, speaker_segments):
    """Split a segment based on speaker boundaries."""
    
    seg_start, seg_end = segment['start'], segment['end']
    text_words = segment['text'].split()
    
    # Find speaker change points within segment
    change_points = [seg_start]
    
    for s_seg in speaker_segments:
        if seg_start < s_seg['start'] < seg_end:
            change_points.append(s_seg['start'])
        if seg_start < s_seg['end'] < seg_end:
            change_points.append(s_seg['end'])
    
    change_points.append(seg_end)
    change_points = sorted(set(change_points))
    
    # Create sub-segments
    sub_segments = []
    for i in range(len(change_points) - 1):
        sub_start = change_points[i]
        sub_end = change_points[i + 1]
        
        # Find speaker for this sub-segment
        speaker = find_speaker_at_time((sub_start + sub_end) / 2, speaker_segments)
        
        # Estimate text portion (simple word distribution)
        progress_ratio = (sub_end - seg_start) / (seg_end - seg_start)
        word_index = int(progress_ratio * len(text_words))
        
        sub_segments.append({
            'start': sub_start,
            'end': sub_end,
            'speaker': speaker if speaker else 'UNKNOWN',
            'text': ' '.join(text_words[:word_index]) if i == 0 else ' '.join(text_words[word_index:])
        })
    
    return sub_segments
```

## RTTM Parser Implementation

```python
def parse_rttm_file(rttm_path):
    """
    Parse RTTM format file from NeMo output.
    
    RTTM format:
    SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
    
    Returns:
        List of speaker segments with start, end, speaker fields
    """
    
    speaker_segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                
                if parts[0] == 'SPEAKER':
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    end_time = start_time + duration
                    speaker_id = parts[7]
                    
                    speaker_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'speaker': speaker_id,
                        'confidence': 1.0  # NeMo doesn't provide confidence
                    })
    
    # Sort by start time
    speaker_segments.sort(key=lambda x: x['start'])
    
    return speaker_segments
```

## Complete Integration Example

```python
class SpeakerAligner:
    """Complete speaker alignment system."""
    
    def __init__(self, algorithm='weighted', enable_splitting=False):
        self.algorithm = algorithm
        self.enable_splitting = enable_splitting
    
    def align(self, transcription_result, rttm_path):
        """
        Main alignment method.
        
        Args:
            transcription_result: faster-whisper transcription output
            rttm_path: Path to NeMo RTTM output file
        
        Returns:
            Transcription segments with speaker attribution
        """
        
        # Parse RTTM file
        speaker_segments = parse_rttm_file(rttm_path)
        
        # Extract segments from transcription
        transcription_segments = self.extract_segments(transcription_result)
        
        # Apply alignment algorithm
        if self.algorithm == 'weighted':
            aligned_segments = align_transcription_with_speakers(
                transcription_segments, speaker_segments
            )
        elif self.algorithm == 'midpoint':
            aligned_segments = self.align_midpoint(
                transcription_segments, speaker_segments
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Optional: Handle multi-speaker segments
        if self.enable_splitting:
            aligned_segments = handle_multi_speaker_segments(
                aligned_segments, speaker_segments
            )
        
        # Post-process to clean up speaker labels
        aligned_segments = self.post_process(aligned_segments)
        
        return aligned_segments
    
    def extract_segments(self, transcription_result):
        """Extract segments from faster-whisper output."""
        segments = []
        for segment in transcription_result.segments:
            segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': segment.words if hasattr(segment, 'words') else None
            })
        return segments
    
    def align_midpoint(self, transcription_segments, speaker_segments):
        """Legacy midpoint algorithm for compatibility."""
        for t_seg in transcription_segments:
            midpoint = (t_seg['start'] + t_seg['end']) / 2
            t_seg['speaker'] = find_speaker_at_time(midpoint, speaker_segments) or 'UNKNOWN'
        return transcription_segments
    
    def post_process(self, segments):
        """Clean up speaker labels and merge adjacent same-speaker segments."""
        
        # Rename generic speaker IDs to readable format
        speaker_map = {}
        speaker_counter = 0
        
        for segment in segments:
            if segment['speaker'] != 'UNKNOWN':
                if segment['speaker'] not in speaker_map:
                    speaker_map[segment['speaker']] = f"SPEAKER_{speaker_counter:02d}"
                    speaker_counter += 1
                segment['speaker'] = speaker_map[segment['speaker']]
        
        # Optional: Merge adjacent segments with same speaker
        merged_segments = []
        for segment in segments:
            if merged_segments and merged_segments[-1]['speaker'] == segment['speaker']:
                # Check if segments are close enough to merge (within 0.5 seconds)
                if segment['start'] - merged_segments[-1]['end'] < 0.5:
                    merged_segments[-1]['end'] = segment['end']
                    merged_segments[-1]['text'] += ' ' + segment['text']
                    continue
            merged_segments.append(segment)
        
        return merged_segments
```

## Performance Optimization

### For Large Files

```python
def build_interval_tree(speaker_segments):
    """Build interval tree for O(log n) speaker lookup."""
    from intervaltree import IntervalTree
    
    tree = IntervalTree()
    for segment in speaker_segments:
        tree.addi(segment['start'], segment['end'], segment['speaker'])
    return tree

def find_speaker_optimized(timestamp, interval_tree):
    """Fast speaker lookup using interval tree."""
    intervals = interval_tree.at(timestamp)
    if intervals:
        return intervals.pop().data
    return None
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def align_parallel(transcription_segments, speaker_segments, num_workers=4):
    """Parallel alignment for large transcriptions."""
    
    def align_chunk(chunk):
        return align_transcription_with_speakers(chunk, speaker_segments)
    
    # Split into chunks
    chunk_size = len(transcription_segments) // num_workers
    chunks = [
        transcription_segments[i:i+chunk_size] 
        for i in range(0, len(transcription_segments), chunk_size)
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(align_chunk, chunks)
    
    # Combine results
    aligned_segments = []
    for chunk_result in results:
        aligned_segments.extend(chunk_result)
    
    return aligned_segments
```

## Testing and Validation

```python
def calculate_alignment_accuracy(aligned_segments, ground_truth):
    """Calculate accuracy metrics for speaker alignment."""
    
    total_duration = 0
    correct_duration = 0
    
    for aligned, truth in zip(aligned_segments, ground_truth):
        duration = aligned['end'] - aligned['start']
        total_duration += duration
        
        if aligned['speaker'] == truth['speaker']:
            correct_duration += duration
    
    accuracy = correct_duration / total_duration if total_duration > 0 else 0
    return accuracy

# Unit test example
def test_weighted_intersection():
    transcription = [{'start': 0, 'end': 5, 'text': 'Hello world'}]
    speakers = [
        {'start': 0, 'end': 3, 'speaker': 'A'},
        {'start': 3, 'end': 5, 'speaker': 'B'}
    ]
    
    result = align_transcription_with_speakers(transcription, speakers)
    assert result[0]['speaker'] == 'A'  # A has more overlap (3s vs 2s)
    assert result[0]['speaker_confidence'] == 0.6  # 3s out of 5s
```

This comprehensive alignment algorithm ensures accurate speaker attribution while maintaining compatibility with the existing API format.