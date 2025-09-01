#!/bin/bash
# Start script for Whisper with Diarization in Docker

echo "🚀 Starting Whisper API with Diarization Support..."

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.config/whisper/token ]; then
        export HF_TOKEN=$(grep "HF_TOKEN=" ~/.config/whisper/token | cut -d'=' -f2)
        echo "✅ HF Token loaded: ${HF_TOKEN:0:10}..."
    else
        echo "⚠️  No HF_TOKEN found - diarization will not work"
    fi
fi

# Stop any existing containers
echo "Stopping existing containers..."
docker stop whisper-api-diarization 2>/dev/null
docker rm whisper-api-diarization 2>/dev/null

# Check if image exists
if docker images | grep -q "whisper-diarization"; then
    echo "✅ Image exists, starting container..."
else
    echo "⚠️  Image not found. Building..."
    docker compose -f docker-compose.diarization.yml build
fi

# Start the service
echo "Starting service..."
docker compose -f docker-compose.diarization.yml up -d

# Wait for service to be ready
echo "Waiting for service to start..."
sleep 15

# Check health
echo "Checking service health..."
if curl -s http://localhost:8767/health > /dev/null 2>&1; then
    echo "✅ Service is running!"
    echo ""
    echo "Health status:"
    curl -s http://localhost:8767/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Status: {data.get('status', 'unknown')}\"
print(f\"  Device: {data.get('device', 'unknown')}\"
print(f\"  GPU: {data.get('gpu', {}).get('device_name', 'None')}\"
print(f\"  Diarization available: {data.get('diarization', {}).get('available', False)}\")
"
    echo ""
    echo "View logs: docker logs -f whisper-api-diarization"
else
    echo "⚠️  Service not responding. Check logs:"
    docker logs whisper-api-diarization 2>&1 | tail -20
fi