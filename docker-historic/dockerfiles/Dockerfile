# Dockerfile for Whisper API with Diarization
# Uses Python 3.11 for best compatibility

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_diarization.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_diarization.txt

# Copy application code
COPY main.py .
COPY test_transcribe.py .
COPY test_diarization.py .

# Create config directory
RUN mkdir -p /root/.config/whisper

# Expose port
EXPOSE 8765

# Set environment variables
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=float16
ENV WHISPER_DIARIZE=true

# Run the service
CMD ["python", "main.py"]