#!/bin/bash

# Unified Whisper API Launcher
# Automatically uses the right Python environment and starts the service

set -e

# Configuration
SERVICE_NAME="Whisper API with Optional Diarization"
VENV_PATH="$HOME/.venvs/whisper-diarize"
SERVICE_PORT="8765"
LOG_FILE="$HOME/.whisper-api.log"
PID_FILE="$HOME/.whisper-api.pid"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if service is already running
check_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm "$PID_FILE"  # Clean up stale PID file
        fi
    fi
    
    # Also check if port is in use
    if lsof -i:$SERVICE_PORT > /dev/null 2>&1; then
        return 0  # Port in use
    fi
    
    return 1  # Not running
}

# Function to stop the service
stop_service() {
    echo -e "${YELLOW}Stopping $SERVICE_NAME...${NC}"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID"  # Force kill if needed
            fi
            rm "$PID_FILE"
            echo -e "${GREEN}✓ Service stopped${NC}"
        else
            echo "Service not running (stale PID file)"
            rm "$PID_FILE"
        fi
    else
        echo "Service not running (no PID file)"
    fi
}

# Function to start the service
start_service() {
    # Check if already running
    if check_running; then
        echo -e "${YELLOW}Service is already running${NC}"
        echo "Use '$0 restart' to restart it"
        return 1
    fi
    
    echo -e "${GREEN}Starting $SERVICE_NAME...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
        echo ""
        echo "Please run one of these first:"
        echo "  ./setup_isolated_python.sh  (for pyenv setup)"
        echo "  ./setup_venv.sh             (if Python 3.11 is installed)"
        exit 1
    fi
    
    # Activate virtual environment and start service
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    
    # Check if dependencies are installed
    if ! python -c "import fastapi, pyannote.audio" 2>/dev/null; then
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -r requirements_diarization.txt
    fi
    
    # Set environment variables
    export WHISPER_MODEL="${WHISPER_MODEL:-small}"
    export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
    export WHISPER_COMPUTE="${WHISPER_COMPUTE:-float16}"
    export WHISPER_DIARIZE="${WHISPER_DIARIZE:-true}"  # Available but not default
    
    # Start the service in background
    echo "Starting service on port $SERVICE_PORT..."
    nohup python main.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Wait a moment and check if it started
    sleep 3
    if ps -p "$PID" > /dev/null; then
        echo -e "${GREEN}✓ Service started successfully (PID: $PID)${NC}"
        echo ""
        echo "Service endpoints:"
        echo "  Health: http://localhost:$SERVICE_PORT/health"
        echo "  API:    http://localhost:$SERVICE_PORT/v1/transcribe"
        echo ""
        echo "Diarization status:"
        curl -s "http://localhost:$SERVICE_PORT/health" | python -m json.tool | grep -A4 '"diarization"' || true
        echo ""
        echo "Logs: tail -f $LOG_FILE"
        echo "Stop: $0 stop"
    else
        echo -e "${RED}✗ Failed to start service${NC}"
        echo "Check logs: tail $LOG_FILE"
        rm "$PID_FILE"
        exit 1
    fi
}

# Function to show service status
show_status() {
    echo -e "${GREEN}=== $SERVICE_NAME Status ===${NC}"
    
    if check_running; then
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            echo -e "Status: ${GREEN}Running${NC} (PID: $PID)"
        else
            echo -e "Status: ${GREEN}Running${NC} (external process)"
        fi
        
        # Get health status
        echo ""
        echo "Health check:"
        curl -s "http://localhost:$SERVICE_PORT/health" 2>/dev/null | python -m json.tool || echo "  Unable to connect"
        
        # Show recent logs
        echo ""
        echo "Recent logs:"
        if [ -f "$LOG_FILE" ]; then
            tail -5 "$LOG_FILE"
        fi
    else
        echo -e "Status: ${RED}Not running${NC}"
    fi
    
    echo ""
    echo "Configuration:"
    echo "  Port: $SERVICE_PORT"
    echo "  Venv: $VENV_PATH"
    echo "  Logs: $LOG_FILE"
    echo "  PID:  $PID_FILE"
}

# Function to show usage examples
show_usage() {
    echo -e "${GREEN}=== How to Use Whisper API ===${NC}"
    echo ""
    echo "1. Basic transcription (no diarization):"
    echo '   curl -X POST http://localhost:8765/v1/transcribe \'
    echo '     -F "file=@audio.wav"'
    echo ""
    echo "2. With speaker diarization:"
    echo '   curl -X POST http://localhost:8765/v1/transcribe \'
    echo '     -F "file=@audio.wav" \'
    echo '     -F "diarize=true" \'
    echo '     -F "num_speakers=2"'
    echo ""
    echo "3. Using the CLI wrapper:"
    echo '   ./whisper-cli.sh audio.wav --diarize --speakers 3'
    echo ""
    echo "4. Using Python client:"
    echo '   from whisper_client import WhisperClient'
    echo '   client = WhisperClient()'
    echo '   result = client.transcribe("audio.wav", diarize=True)'
    echo ""
    echo "Diarization is OPTIONAL per request:"
    echo "  - Default: diarize=false (fast, no speaker detection)"
    echo "  - With diarize=true: slower but identifies speakers"
}

# Main command handling
case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        show_status
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found"
        fi
        ;;
    usage|help)
        show_usage
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|usage}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the Whisper API service"
        echo "  stop    - Stop the service"
        echo "  restart - Restart the service"
        echo "  status  - Show service status and health"
        echo "  logs    - Tail the service logs"
        echo "  usage   - Show API usage examples"
        exit 1
        ;;
esac