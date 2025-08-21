#!/bin/bash

# CLI wrapper for Whisper API with diarization
# Usage: ./whisper-cli.sh <audio-file> [options]

API_URL="http://localhost:8765"

# Check if service is running
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo "Error: Whisper API is not running on $API_URL"
    echo "Start it with: source ~/.venvs/whisper-diarize/bin/activate && python main.py"
    exit 1
fi

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio-file> [--diarize] [--speakers N] [--format json|text|srt|vtt]"
    echo ""
    echo "Examples:"
    echo "  $0 meeting.wav --diarize"
    echo "  $0 interview.mp3 --diarize --speakers 2"
    echo "  $0 podcast.wav --diarize --format srt > podcast.srt"
    exit 1
fi

AUDIO_FILE="$1"
shift

# Default options
DIARIZE="false"
SPEAKERS=""
FORMAT="json"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --diarize)
            DIARIZE="true"
            shift
            ;;
        --speakers)
            SPEAKERS="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: File not found: $AUDIO_FILE"
    exit 1
fi

# Build curl command
CURL_CMD="curl -s -X POST $API_URL/v1/transcribe"
CURL_CMD="$CURL_CMD -F file=@$AUDIO_FILE"
CURL_CMD="$CURL_CMD -F diarize=$DIARIZE"
CURL_CMD="$CURL_CMD -F format=$FORMAT"

if [ -n "$SPEAKERS" ]; then
    CURL_CMD="$CURL_CMD -F num_speakers=$SPEAKERS"
fi

# Execute
if [ "$FORMAT" = "json" ]; then
    # Pretty print JSON
    $CURL_CMD | jq '.'
else
    # Raw output for text/srt/vtt
    $CURL_CMD
fi