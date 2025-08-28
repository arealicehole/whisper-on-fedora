#!/bin/bash

# View logs from the Whisper Docker container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if container is running
if docker-compose -f docker/docker-compose.yml ps | grep -q "whisper-blackwell.*Up"; then
    echo "Showing logs from whisper-blackwell container (Ctrl+C to exit)..."
    echo
    docker-compose -f docker/docker-compose.yml logs -f --tail=100 whisper-blackwell
else
    echo "Container is not running. Showing last 100 lines of logs..."
    echo
    docker-compose -f docker/docker-compose.yml logs --tail=100 whisper-blackwell
fi