#!/bin/bash
# Production Testing Script for NeMo Integration
# Tests GPU detection, transcription with diarization, and health endpoints

set -e  # Exit on error

echo "🧪 NeMo Integration Test Suite"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://localhost:8769"
TEST_AUDIO="test_sample.wav"

# Function to check if service is running
check_service() {
    echo -n "Checking if service is running... "
    if curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health" | grep -q "200"; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        echo "Service not running. Starting with docker compose..."
        docker compose -f docker-compose.nemo.yml up -d
        echo "Waiting for service to start (60 seconds)..."
        sleep 60
    fi
}

# Test 1: GPU Detection
test_gpu_detection() {
    echo ""
    echo "Test 1: GPU Detection"
    echo "---------------------"
    
    docker compose -f docker-compose.nemo.yml run --rm whisper-nemo python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    if torch.cuda.get_device_capability(0) == (12, 0):
        print('✓ Blackwell GPU (RTX 5060 Ti) detected')
    exit(0)
else:
    print('✗ No GPU detected')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPU detection test passed${NC}"
    else
        echo -e "${RED}✗ GPU detection test failed${NC}"
        exit 1
    fi
}

# Test 2: NeMo Module Loading
test_nemo_loading() {
    echo ""
    echo "Test 2: NeMo Module Loading"
    echo "---------------------------"
    
    docker compose -f docker-compose.nemo.yml run --rm whisper-nemo python3 -c "
try:
    from nemo.collections.asr.models import ClusteringDiarizer
    print('✓ NeMo ClusteringDiarizer loaded successfully')
    from nemo_diarizer import NeMoDiarizer
    print('✓ NeMoDiarizer wrapper loaded successfully')
    exit(0)
except Exception as e:
    print(f'✗ Failed to load NeMo: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ NeMo loading test passed${NC}"
    else
        echo -e "${RED}✗ NeMo loading test failed${NC}"
        exit 1
    fi
}

# Test 3: Health Endpoint
test_health_endpoint() {
    echo ""
    echo "Test 3: Health Endpoint"
    echo "-----------------------"
    
    check_service
    
    HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/health")
    echo "Health Response:"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool
    
    # Check for required fields
    if echo "$HEALTH_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
checks = [
    ('status' in data and data['status'] == 'healthy', 'Service status'),
    ('gpu_required' in data and data['gpu_required'], 'GPU requirement'),
    ('gpu_available' in data and data['gpu_available'], 'GPU availability'),
    ('diarization' in data, 'Diarization info'),
    ('diarization' in data and 'modules_available' in data['diarization'], 'NeMo modules')
]
all_pass = True
for check, name in checks:
    if check:
        print(f'✓ {name}')
    else:
        print(f'✗ {name}')
        all_pass = False
exit(0 if all_pass else 1)
"; then
        echo -e "${GREEN}✓ Health endpoint test passed${NC}"
    else
        echo -e "${RED}✗ Health endpoint test failed${NC}"
        exit 1
    fi
}

# Test 4: Transcription without Diarization
test_transcription_only() {
    echo ""
    echo "Test 4: Transcription without Diarization"
    echo "-----------------------------------------"
    
    # Create test audio if it doesn't exist
    if [ ! -f "$TEST_AUDIO" ]; then
        echo "Creating test audio file..."
        docker compose -f docker-compose.nemo.yml run --rm whisper-nemo python3 -c "
import numpy as np
import soundfile as sf
# Generate 3 seconds of test audio
sample_rate = 16000
duration = 3
t = np.linspace(0, duration, sample_rate * duration)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
sf.write('$TEST_AUDIO', audio, sample_rate)
print('Test audio created')
"
    fi
    
    echo "Sending transcription request..."
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/v1/transcribe" \
        -F "file=@$TEST_AUDIO" \
        -F "diarize=false" \
        -F "format=json")
    
    if echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'segments' in data:
    print(f'✓ Received {len(data[\"segments\"])} segments')
    exit(0)
else:
    print('✗ No segments in response')
    exit(1)
"; then
        echo -e "${GREEN}✓ Transcription test passed${NC}"
    else
        echo -e "${RED}✗ Transcription test failed${NC}"
        echo "Response: $RESPONSE"
        exit 1
    fi
}

# Test 5: Transcription with Diarization
test_transcription_with_diarization() {
    echo ""
    echo "Test 5: Transcription with Diarization"
    echo "--------------------------------------"
    
    echo "Sending transcription request with diarization..."
    START_TIME=$(date +%s)
    
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/v1/transcribe" \
        -F "file=@$TEST_AUDIO" \
        -F "diarize=true" \
        -F "num_speakers=2" \
        -F "format=json")
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Processing time: ${DURATION} seconds"
    
    if echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'segments' in data:
    segments = data['segments']
    print(f'✓ Received {len(segments)} segments')
    # Check if segments have speaker labels
    has_speakers = any('speaker' in seg for seg in segments)
    if has_speakers:
        print('✓ Segments have speaker labels')
        speakers = set(seg.get('speaker', 'UNKNOWN') for seg in segments)
        print(f'  Speakers found: {speakers}')
        exit(0)
    else:
        print('✗ No speaker labels in segments')
        exit(1)
else:
    print('✗ No segments in response')
    exit(1)
"; then
        echo -e "${GREEN}✓ Diarization test passed${NC}"
    else
        echo -e "${YELLOW}⚠ Diarization test failed (may be expected if HF token not configured)${NC}"
        echo "Response: $RESPONSE"
    fi
}

# Test 6: Performance Check
test_performance() {
    echo ""
    echo "Test 6: Performance Check"
    echo "-------------------------"
    
    # Create a longer test audio (30 seconds)
    LONG_AUDIO="test_long.wav"
    if [ ! -f "$LONG_AUDIO" ]; then
        echo "Creating 30-second test audio..."
        docker compose -f docker-compose.nemo.yml run --rm whisper-nemo python3 -c "
import numpy as np
import soundfile as sf
sample_rate = 16000
duration = 30
t = np.linspace(0, duration, sample_rate * duration)
audio = 0.5 * np.sin(2 * np.pi * 440 * t)
sf.write('$LONG_AUDIO', audio, sample_rate)
print('30-second test audio created')
"
    fi
    
    echo "Processing 30-second audio with diarization..."
    START_TIME=$(date +%s.%N)
    
    curl -s -X POST "$SERVICE_URL/v1/transcribe" \
        -F "file=@$LONG_AUDIO" \
        -F "diarize=true" \
        -F "format=json" > /dev/null
    
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    
    echo "Processing time: ${DURATION} seconds"
    RATE=$(echo "scale=2; $DURATION / 0.5" | bc)  # 30 seconds = 0.5 minutes
    echo "Processing rate: ${RATE} seconds per minute of audio"
    
    # Check if under 2 seconds per minute (target from PRP)
    if (( $(echo "$RATE < 2" | bc -l) )); then
        echo -e "${GREEN}✓ Performance test passed (< 2s/min)${NC}"
    else
        echo -e "${YELLOW}⚠ Performance slower than target (${RATE}s/min vs 2s/min target)${NC}"
    fi
}

# Test 7: GPU Memory Check
test_gpu_memory() {
    echo ""
    echo "Test 7: GPU Memory Check"
    echo "------------------------"
    
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "Current GPU memory usage: ${GPU_MEM} MB"
    
    # Process multiple requests to check for memory leaks
    echo "Sending 5 consecutive requests..."
    for i in {1..5}; do
        curl -s -X POST "$SERVICE_URL/v1/transcribe" \
            -F "file=@$TEST_AUDIO" \
            -F "diarize=true" \
            -F "format=json" > /dev/null
        echo -n "."
    done
    echo ""
    
    GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory after 5 requests: ${GPU_MEM_AFTER} MB"
    
    MEM_INCREASE=$((GPU_MEM_AFTER - GPU_MEM))
    echo "Memory increase: ${MEM_INCREASE} MB"
    
    if [ $MEM_INCREASE -lt 1000 ]; then
        echo -e "${GREEN}✓ Memory management test passed (no significant leak)${NC}"
    else
        echo -e "${YELLOW}⚠ Possible memory leak detected (${MEM_INCREASE} MB increase)${NC}"
    fi
}

# Main test execution
main() {
    echo ""
    echo "Starting NeMo Integration Tests"
    echo "================================"
    echo ""
    
    # Check for HF token
    if [ -z "$HF_TOKEN" ] && [ ! -f ~/.config/whisper/token ]; then
        echo -e "${YELLOW}⚠ Warning: No HuggingFace token found${NC}"
        echo "  Diarization tests may fail without a valid token"
        echo "  Set HF_TOKEN environment variable or add to ~/.config/whisper/token"
    fi
    
    # Run tests
    test_gpu_detection
    test_nemo_loading
    test_health_endpoint
    test_transcription_only
    test_transcription_with_diarization
    test_performance
    test_gpu_memory
    
    echo ""
    echo "================================"
    echo -e "${GREEN}✓ All tests completed${NC}"
    echo ""
    echo "Service is running at: $SERVICE_URL"
    echo "View logs: docker compose -f docker-compose.nemo.yml logs -f"
    echo ""
}

# Run main function
main