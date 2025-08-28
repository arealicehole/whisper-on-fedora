# Post-Reboot Instructions for Whisper API with GPU Enforcement

## Context
The Whisper API has been modified to enforce GPU-only operation (no CPU fallback). The RTX 5060 Ti Blackwell GPU requires specific setup to work with CUDA.

## Step 1: Verify GPU is Detected
```bash
nvidia-smi
```
Expected: Should show RTX 5060 Ti with driver version

## Step 2: Test CUDA Access
```bash
# Activate the correct virtual environment
source ~/.venvs/whisper-blackwell/bin/activate

# Test CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Step 3A: If CUDA Works - Start the Service
```bash
# Navigate to the project directory
cd /home/ice/whisper-api

# Activate virtual environment
source ~/.venvs/whisper-blackwell/bin/activate

# Start the service
python main.py
```

The service will run on `http://localhost:8765`

### Test the service:
```bash
# Health check
curl http://localhost:8765/health

# Basic transcription test (if you have an audio file)
curl -X POST http://localhost:8765/v1/transcribe -F "file=@test.wav"
```

## Step 3B: If CUDA Still Doesn't Work

### Option 1: Fix Device Permissions
```bash
# Check device permissions
ls -la /dev/nvidia*

# If not 666, fix them:
sudo chmod 666 /dev/nvidia*
sudo chmod 666 /dev/nvidiactl
sudo chmod 666 /dev/nvidia-modeset
sudo chmod 666 /dev/nvidia-uvm
sudo chmod 666 /dev/nvidia-uvm-tools
```

### Option 2: Reload NVIDIA Modules
```bash
# Remove modules
sudo rmmod nvidia_uvm
sudo rmmod nvidia_modeset 
sudo rmmod nvidia

# Reload modules
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset

# Set persistence mode
sudo nvidia-smi -pm 1

# Test CUDA again
source ~/.venvs/whisper-blackwell/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Option 3: Reinstall PyTorch (if needed)
```bash
# Activate environment
source ~/.venvs/whisper-blackwell/bin/activate

# Uninstall existing PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch nightly with CUDA 12.8 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Test
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Step 4: Alternative - Use Docker
If native CUDA still doesn't work:

```bash
# Pull the NGC container with Blackwell support
docker pull nvcr.io/nvidia/pytorch:25.02-py3

# Run the service in Docker
docker run --gpus all -it --rm \
  -v /home/ice/whisper-api:/app \
  -w /app \
  -p 8765:8765 \
  nvcr.io/nvidia/pytorch:25.02-py3 \
  python main.py
```

## Important Files Modified for GPU Enforcement

1. **gpu_validator.py** - New GPU validation utility that enforces GPU requirements
2. **main.py** - Modified to remove CPU fallback, requires GPU at startup
3. **diarization_handler.py** - CPU fallback removed (if this file exists)
4. **whisper_config_override.py** - Forces CUDA device for both whisper and diarization
5. **start_whisper.sh** - Added GPU validation before service start
6. **fix_blackwell_cuda.sh** - Removed CPU fallback options
7. **tests/test_gpu_enforcement.py** - Test suite for GPU enforcement

## Key Changes Made
- Service will **EXIT** if GPU is not available (no CPU fallback)
- Both Whisper and Diarization require GPU
- Clear error messages guide users to fix GPU issues
- GPU validation happens at startup

## Troubleshooting Commands

### Check GPU status:
```bash
nvidia-smi
```

### Check CUDA in Python:
```bash
source ~/.venvs/whisper-blackwell/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Run GPU validator:
```bash
cd /home/ice/whisper-api
source ~/.venvs/whisper-blackwell/bin/activate
python gpu_validator.py
```

### Check service logs (if running):
```bash
tail -f ~/.whisper-api.log
```

## Expected Behavior
- Service refuses to start without GPU
- Clear error messages about GPU requirements
- No performance degradation from CPU fallback
- Full GPU acceleration for both transcription and diarization

## If All Else Fails
The CUDA unknown error on Fedora with the RTX 5060 Ti might require:
1. Different NVIDIA driver version
2. Kernel module updates
3. Docker container approach (most reliable)

Remember: The service now **requires** GPU and will not fall back to CPU under any circumstances.