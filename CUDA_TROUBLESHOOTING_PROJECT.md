# Whisper API CUDA 13.0 Driver Compatibility Project

## Project Overview
**Status:** In Progress  
**Priority:** High  
**Created:** September 16, 2025  
**Issue:** CUDA error 999 after system update upgraded NVIDIA driver to 580.65.06 (CUDA 13.0)

## Problem Statement
Whisper API Docker container worked perfectly before system update but now fails with:
```
ERROR: CUDA unknown error (error 999)
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment
```

## Root Cause Analysis
1. **System Update Impact:** Fedora system update upgraded NVIDIA driver from 550.x (CUDA 12.4) to 580.65.06 (CUDA 13.0)
2. **Container Mismatch:** Existing container built for CUDA 12.4 but host now runs CUDA 13.0
3. **Forward Compatibility Issues:** CUDA 13.0 driver having compatibility problems with 12.4 containers

## Current System State
- **Host Driver:** NVIDIA 580.65.06 (CUDA 13.0)
- **Container Base:** nvcr.io/nvidia/pytorch:25.02-py3 (CUDA 12.4)
- **Hardware:** RTX 5060 Ti Blackwell GPU (16GB VRAM)
- **Docker MTU:** Fixed (1420 for VPN compatibility)

## Attempted Solutions

### ✅ Completed
1. **Fixed Docker MTU Issue**
   - Problem: VPN (us-phx) using MTU 1420, Docker defaulted to 1500
   - Solution: Updated `/etc/docker/daemon.json` with `"mtu": 1420`
   - Result: Container builds now work, registry timeouts resolved

2. **Reconfigured NVIDIA Container Runtime**
   - Command: `sudo nvidia-ctk runtime configure --runtime=docker`
   - Command: `sudo systemctl restart docker`
   - Result: Runtime updated but CUDA error persists

3. **Updated to Correct Container Image**
   - Problem: Using whisper-blackwell:d4-n1 (NeMo not properly installed)
   - Solution: Updated docker-compose.yml to use whisper-blackwell:d4-n2
   - Result: Container has proper NeMo support but CUDA error persists

4. **Rebuilt NVIDIA Kernel Modules After Kernel Update**
   - Problem: Kernel updated to 6.16.4-200.fc42.x86_64 breaking NVIDIA modules
   - Commands: `sudo dkms autoinstall` + `sudo systemctl restart docker`
   - Result: Modules rebuilt correctly but CUDA error 999 persists in containers

5. **Verified Driver Compatibility**
   - Research: Driver 580.65.06 IS compatible with Blackwell RTX 5060 Ti (sm_120)
   - Research: NGC 25.02 container IS optimized for Blackwell architecture
   - Result: No compatibility issue - problem is runtime state corruption

### 🔄 Current Status  
**DIAGNOSIS: CUDA Runtime State Corruption After Kernel Update**

**Evidence:**
- `nvidia-smi` works perfectly on host
- NVIDIA kernel modules loaded and match current kernel (6.16.4-200.fc42.x86_64)
- Docker containers consistently get CUDA error 999
- Host CUDA functionality intact, container CUDA access broken

**Conclusion:** The kernel update corrupted Docker's GPU runtime state. NVIDIA modules can't be manually unloaded (in use), preventing clean reload. **System reboot required** to fully reinitialize GPU driver stack.

## Next Actions

### Immediate (Required)
1. **System Reboot**
   - Reason: New NVIDIA driver (580.65.06) needs full initialization
   - Expected: May resolve CUDA runtime conflicts
   - Command: `sudo reboot`

2. **Post-Reboot Testing**
   ```bash
   cd /home/ice/whisper-api
   docker compose -f docker-compose.blackwell.yml up
   ```

### If Reboot Doesn't Fix

#### Option A: Driver Rollback (Risky)
- Downgrade to NVIDIA 550.x series (CUDA 12.4)
- **Risk:** May break system or cause boot issues on Fedora

#### Option B: Wait for Compatibility Updates
- Monitor for PyTorch updates with CUDA 13.0 support
- Check for NVIDIA container runtime updates
- **Timeline:** Likely weeks to months

#### Option C: Alternative Container Strategy
- Use older NGC container with known CUDA 12.4 compatibility
- Implement container-level CUDA version detection
- Add fallback mechanisms

## Technical Details

### Error Stack Trace
```
File "/workspace/gpu_validator.py", line 106, in enforce_gpu_requirements
    device_props = torch.cuda.get_device_properties(0)
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment
```

### Container Configuration
```yaml
# docker-compose.blackwell.yml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Working Dockerfile (Pre-Update)
```dockerfile
FROM nvcr.io/nvidia/pytorch:25.02-py3
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg sox
RUN pip install faster-whisper==1.0.3 fastapi uvicorn nemo_toolkit[asr]
```

## Success Criteria
- [ ] Container starts without CUDA errors
- [ ] GPU validation passes: `torch.cuda.is_available() == True`
- [ ] Whisper transcription works with GPU acceleration
- [ ] NeMo diarization functional
- [ ] API responds to health checks at http://localhost:8771/health

## Risk Assessment
- **High:** CUDA 13.0 is very new, limited ecosystem support
- **Medium:** Driver rollback could cause system instability
- **Low:** Current setup worked reliably before update

## ⚠️ UPDATE: Issue Recurred After New System Update

### September 16, 2025 - 7:10 PM

**Timeline:**
- **7:00 PM**: First reboot completed, container healthy, GPU working
- **7:05 PM**: Successfully processed one vocoder query  
- **7:06 PM**: Container crashed with CUDA error 999 after single request
- **7:10 PM**: Container stuck in restart loop, all queries failing
- **7:11 PM**: Detected another system update had been installed

**Current Status:**
- Container crashes immediately with CUDA error 999
- Exit code 1, not OOM (plenty of RAM/GPU memory available)
- Host nvidia-smi still works fine
- Docker containers cannot access GPU

**Pattern Identified:**
1. System update corrupts CUDA runtime
2. Reboot temporarily fixes issue
3. Works for 1-2 requests
4. CUDA runtime corrupts again
5. Requires another reboot

**Root Cause:** 
- NVIDIA driver 580.65.06 (CUDA 13.0) fundamentally incompatible with NGC PyTorch 2.7.0a0 (expects CUDA 12.4)
- Runtime becomes unstable after processing requests
- System updates triggering the corruption repeatedly

**Immediate Action:** Another reboot required after latest system update

### ✅ Post-Reboot Status - September 16, 2025 - 7:20 PM

**WORKING (Temporarily):**
- ✅ Container: Running and healthy
- ✅ GPU: RTX 5060 Ti detected with 15.48GB memory  
- ✅ CUDA: Available and working
- ✅ Diarization: NeMo backend loaded
- ✅ API: Responding on port 8771

**Warning:** Container working but expected to crash again after processing multiple requests due to CUDA 13.0/PyTorch compatibility issues.

### 🔄 ONGOING: Suspend/Resume Correlation - September 17, 2025

**Additional Pattern Discovered:**
- User reported: "i suspended my session and reopened it... and now vocoder isnt getting transcripts back"
- Container crashed with exit code 137 (killed) during monitoring
- **17:02**: Container started successfully with NeMo diarization loaded
- **17:03**: Container crashed and required restart
- **17:04**: Restarted successfully and currently stable

**Hypothesis:** System suspend/resume cycles corrupt CUDA runtime state, similar to system updates. GPU context may not properly restore after suspend, causing subsequent CUDA operations to fail with error 999.

**Current Status - 17:04 PM:**
- ✅ Container: Running and healthy  
- ✅ GPU: RTX 5060 Ti detected (15.48GB memory)
- ✅ CUDA: Available and working
- ✅ Diarization: NeMo backend loaded with HF token
- ✅ API: Health endpoint responding
- ⚠️ **Monitoring**: Watching for crashes during vocoder usage

## Long-term Solutions Needed

1. **Downgrade NVIDIA Driver** to 550.x series (CUDA 12.4 compatible)
2. **Build custom container** with CUDA 13.0 support
3. **Wait for NGC updates** with proper CUDA 13.0/PyTorch compatibility
4. **Disable automatic system updates** until stable configuration found

## Lessons Learned
1. **CUDA version mismatches cause instability** even if initially appearing to work
2. **Always reboot after ANY system update** affecting drivers
3. **Check MTU compatibility** when using VPN + Docker (1420 vs 1500)
4. **Use correct container images** (d4-n2 has NeMo, d4-n1 doesn't)
5. **Driver 580.65.06 unstable** with PyTorch containers expecting CUDA 12.4

## Related Documentation
- `/home/ice/fed/ice_whisper_guide.md` - Full Whisper setup guide
- `/home/ice/fed/vpn/DOCKER_VPN_TROUBLESHOOTING.md` - MTU fix documentation
- `/home/ice/whisper-api/README.md` - Container usage instructions

## Project Dependencies
- NVIDIA driver stability
- PyTorch CUDA 13.0 support timeline
- NGC container updates from NVIDIA
- Docker/container runtime compatibility

---
**Last Updated:** September 16, 2025  
**Next Review:** After reboot testing