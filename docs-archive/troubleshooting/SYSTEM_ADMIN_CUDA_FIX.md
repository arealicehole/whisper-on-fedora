# System Administrator Guide: Fix CUDA Initialization for RTX 5060 Ti Blackwell

## Executive Summary

A user-space application cannot initialize CUDA despite the GPU being properly detected by system tools. This requires system-level intervention to fix device permissions and driver configuration.

## Problem Description

- **GPU**: NVIDIA GeForce RTX 5060 Ti (Blackwell architecture)
- **Issue**: CUDA initialization fails with "CUDA unknown error"
- **Impact**: PyTorch and other CUDA applications cannot use GPU acceleration

## System Diagnostics

### Current System State

```bash
# What Works:
nvidia-smi -L                    # ✅ Shows: GPU 0: NVIDIA GeForce RTX 5060 Ti
nvidia-ml-py (NVML)             # ✅ Can access GPU properties
Driver Version                   # ✅ 580.65.06

# What Fails:
PyTorch torch.cuda.init()       # ❌ CUDA unknown error
Direct CUDA API cuInit()        # ❌ Error code 999
User-space CUDA operations      # ❌ Cannot access GPU
```

### Hardware Configuration

```bash
# Hybrid GPU Setup Detected
$ lspci | grep -i vga
23:00.0 VGA compatible controller: NVIDIA Corporation GP106 [GeForce GTX 1060 6GB] (rev a1)
2d:00.0 VGA compatible controller: NVIDIA Corporation GB206 [GeForce RTX 5060 Ti] (rev a1)

# Loaded Kernel Modules
$ lsmod | grep nvidia
nvidia_uvm           4214784  4    # Note: 4 processes using it (may be stuck)
nvidia_drm            155648  12
nvidia_modeset       2170880  5 nvidia_drm
nvidia              15839232  92 nvidia_uvm,nvidia_modeset
```

## Root Cause Analysis

The issue stems from one or more of:

1. **Device File Permissions**: `/dev/nvidia*` files not accessible to user
2. **Stuck nvidia_uvm Processes**: 4 processes holding UVM module
3. **Hybrid GPU Conflict**: Both nouveau (GTX 1060) and nvidia (RTX 5060 Ti) drivers loaded
4. **CUDA Runtime Mismatch**: System has mixed CUDA versions

## Fix Procedures

### Step 1: Check Current Device Permissions

```bash
# Check device files
ls -la /dev/nvidia*

# Expected output (if broken):
crw-rw-rw- 1 root root 195,   0 Aug 28 10:00 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 Aug 28 10:00 /dev/nvidiactl
crw-rw-rw- 1 root root 195, 254 Aug 28 10:00 /dev/nvidia-modeset
crw-rw-rw- 1 root root 236,   0 Aug 28 10:00 /dev/nvidia-uvm
crw-rw-rw- 1 root root 236,   1 Aug 28 10:00 /dev/nvidia-uvm-tools

# If permissions are wrong (e.g., 660 instead of 666), fix them
```

### Step 2: Fix Device Permissions

```bash
# Immediate fix (temporary)
sudo chmod 666 /dev/nvidia*
sudo nvidia-modprobe -u -c=0

# Permanent fix - Create udev rule
sudo tee /etc/udev/rules.d/99-nvidia.rules << 'EOF'
# NVIDIA device nodes
KERNEL=="nvidia", MODE="0666"
KERNEL=="nvidia[0-9]*", MODE="0666"
KERNEL=="nvidiactl", MODE="0666"
KERNEL=="nvidia-modeset", MODE="0666"
KERNEL=="nvidia-uvm", MODE="0666"
KERNEL=="nvidia-uvm-tools", MODE="0666"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Step 3: Fix nvidia-persistenced

```bash
# Check if nvidia-persistenced is running
systemctl status nvidia-persistenced

# If not running or not installed:
sudo dnf install -y nvidia-persistenced  # For Fedora
# OR
sudo apt install -y nvidia-persistenced  # For Ubuntu/Debian

# Enable and start the service
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

# Set persistence mode
sudo nvidia-smi -pm 1
```

### Step 4: Clear Stuck NVIDIA UVM Processes

```bash
# Check what's using nvidia_uvm
sudo lsof | grep nvidia

# If processes are stuck, clean them up
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# If rmmod fails due to "in use", find and kill the processes
sudo fuser -k /dev/nvidia*
```

### Step 5: Disable Nouveau for GTX 1060 (if causing conflicts)

```bash
# Create blacklist for nouveau
sudo tee /etc/modprobe.d/blacklist-nouveau.conf << 'EOF'
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF

# Regenerate initramfs
sudo dracut --regenerate-all --force  # Fedora
# OR
sudo update-initramfs -u              # Ubuntu/Debian

# Reboot required after this step
```

### Step 6: Set CUDA Environment System-Wide

```bash
# Add to /etc/environment
sudo tee -a /etc/environment << 'EOF'
CUDA_HOME=/usr/local/cuda-12.9
PATH=$PATH:/usr/local/cuda-12.9/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.9/lib64
EOF

# Create profile.d script
sudo tee /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
EOF

sudo chmod +x /etc/profile.d/cuda.sh
```

### Step 7: Configure for RTX 5060 Ti Specifically

```bash
# Set compute capability for Blackwell
sudo tee /etc/profile.d/blackwell.sh << 'EOF'
# RTX 5060 Ti Blackwell Support
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_MODULE_LOADING=EAGER
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
EOF

sudo chmod +x /etc/profile.d/blackwell.sh
```

## Verification Steps

After applying fixes, verify as the user (not root):

```bash
# Test 1: Check device access
ls -la /dev/nvidia*  # Should show 666 permissions

# Test 2: Test CUDA initialization
python3 << 'EOF'
import ctypes
try:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    result = cuda.cuInit(0)
    if result == 0:
        print("✅ CUDA initialization: SUCCESS")
    else:
        print(f"❌ CUDA initialization failed: code {result}")
except Exception as e:
    print(f"❌ Could not load CUDA: {e}")
EOF

# Test 3: Test PyTorch
su - ice  # Switch to user
source ~/.venvs/whisper-blackwell/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Alternative: Docker Solution

If the above fixes don't work, deploy via Docker:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## System Information Commands

```bash
# Collect system info for debugging
nvidia-bug-report.sh  # Creates comprehensive report

# Manual checks
cat /proc/driver/nvidia/version
nvidia-smi -q
nvcc --version
ldconfig -p | grep cuda
dmesg | grep -i nvidia
journalctl -xe | grep -i nvidia
```

## Expected Outcome

After successful fix:

1. User can run: `torch.cuda.is_available()` → Returns `True`
2. GPU memory is accessible from user space
3. CUDA operations execute without errors
4. Whisper API uses GPU acceleration

## Troubleshooting

### If fixes don't work:

1. **Check SELinux/AppArmor**: May block device access
   ```bash
   # Fedora
   sudo setenforce 0  # Temporary disable
   # Check if it works, then create proper SELinux policy
   
   # Ubuntu
   sudo aa-complain /usr/bin/python3
   ```

2. **Check cgroups**: Systemd may restrict device access
   ```bash
   # Add user to video group
   sudo usermod -a -G video ice
   ```

3. **Reinstall NVIDIA drivers**: Complete clean reinstall
   ```bash
   sudo dnf remove nvidia* cuda*
   sudo dnf install akmod-nvidia nvidia-driver cuda-toolkit-12-9
   ```

## Contact Information

- **User**: ice
- **Working Directory**: /home/ice/whisper-api
- **Python Environment**: ~/.venvs/whisper-blackwell (Python 3.11)
- **Service Port**: 8765

## Final Validation

```bash
# As root, run the comprehensive test
sudo -u ice bash << 'EOF'
source ~/.venvs/whisper-blackwell/bin/activate
python /home/ice/whisper-api/blackwell_diagnostic.py
EOF
```

Success indicators:
- "CUDA initialization: SUCCESS"
- "PyTorch has Blackwell (sm_120) support!"
- "CUDA tensor operations: SUCCESS"

## Notes

- The RTX 5060 Ti requires PyTorch nightly builds (already installed)
- The system has a hybrid GPU setup which may need special handling
- User application works with CPU fallback but needs GPU for optimal performance