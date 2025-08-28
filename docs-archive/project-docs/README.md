Fedora System README & Configuration Guide
Date Created: August 7, 2025
Last Updated: August 10, 2025
This document records the exact configuration of this Fedora installation so future maintenance, troubleshooting, and updates are predictable and safe.

**📋 Documentation Index**
- [MCP Servers Setup & Usage Guide](./MCP_SERVERS.md) - Complete guide to Model Context Protocol servers configured for this system

1) System Overview
OS: Fedora (42 or newer)
Disk Encryption: Full Disk Encryption (FDE) with LUKS, including /boot
Filesystem: Btrfs
Snapshots: Snapper configured for root and home

2) Critical Boot Process (Encrypted /boot)
Kernel/GRUB updates can overwrite the EFI GRUB wrapper and drop the cryptomount step needed to unlock the encrypted drive at boot. The system includes an automatic fix.
2.1 Automatic GRUB wrapper repair (post-install hook)
Script path: /etc/kernel/postinst.d/99-cryptomount-fix.sh
code Bash
downloadcontent_copyexpand_less
     #!/usr/bin/env bash
set -euo pipefail
MAINCFG="/boot/grub2/grub.cfg"
WRAPPER="/boot/efi/EFI/fedora/grub.cfg"
UUID=$(grep -om1 'UUID=[0-9a-fA-F]\{8\}-[0-9a-fA-F]\{4\}-[0-9a-fA-F]\{4\}-[0-9a-fA-F]\{4\}-[0-9a-fA-F]\{12\}' /etc/crypttab | cut -d= -f2 || true)
[[ -z ${UUID} ]] && UUID=$(blkid -t TYPE="crypto_LUKS" -o value -s UUID | head -n1 || true)
[[ -z ${UUID} ]] && exit 0


/usr/sbin/grub2-mkconfig -o "${MAINCFG}"
sed -i '/^cryptomount -u [0-9a-fA-F-]\{32,36\}$/d' "${WRAPPER}"
sed -i "1i cryptomount -u ${UUID}" "${WRAPPER}"
chattr +i "${WRAPPER}" 2>/dev/null || true
   
Note: If this hook doesn’t trigger after a kernel update, run the script manually once with sudo bash /etc/kernel/postinst.d/99-cryptomount-fix.sh.

3) Filesystem, Subvolumes & Snapshots
To keep app data persistent across rollbacks, specific directories are dedicated Btrfs subvolumes.
3.1 Subvolumes kept outside home rollbacks
~/.mozilla (Firefox profile)
~/.local/share/flatpak (Flatpak user data)
~/.npm-global (global npm CLI prefix)
npm user prefix
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"
if ! grep -q 'npm-global/bin' "$HOME/.profile" 2>/dev/null; then
  echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.profile"
fi
   
3.2 Snapshot checkpoints
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo snapper -c root create --description "pre-nvidia-open"
sudo snapper -c root create --description "hybrid-open+nouveau"
for C in root home; do
  sudo snapper -c "$C" create --description "GPU OK $(date +'%Y-%m-%d %H:%M') - stable hybrid setup" --cleanup-algorithm number
done
   

4) Software Configuration & Repositories
4.1 Package manager
dnf5
4.2 RPM Fusion```bash
sudo dnf5 install rpmfusion-free-release-tainted rpmfusion-nonfree-release-tainted
code Code
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     *Reference: https://rpmfusion.org/*

**4.3 VS Code**
```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo tee /etc/yum.repos.d/vscode.repo >/dev/null <<'EOF'
[code]
name=Visual Studio Code
baseurl=https://packages.microsoft.com/yumrepos/vscode
enabled=1
gpgcheck=1
gpgkey=https://packages.microsoft.com/keys/microsoft.asc
EOF
   
Reference: https://code.visualstudio.com/
4.4 Multimedia codecs
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo dnf5 group install multimedia --setopt=install_weak_deps=False --exclude=PackageKit-gstreamer-plugin
sudo dnf5 group install sound-and-video
   
4.5 Flatpak (Flathub)
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
   
Reference: https://flathub.org/

5) NVIDIA Driver & CUDA
This host mixes nouveau on Pascal and NVIDIA proprietary (open-kernel) on Ada.
5.1 CUDA 13 repo & toolchain
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo dnf5 install https://developer.download.nvidia.com/compute/cuda/repos/fedora42/$(uname -m)/cuda-fedora42.repo
sudo dnf5 install cuda-toolkit-13-0 cudnn9-cuda13
   
Reference: https://developer.nvidia.com/cuda-downloads
5.2 NVIDIA proprietary driver 580.65.06
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     wget https://us.download.nvidia.com/XFree86/Linux-x86_64/580.65.06/NVIDIA-Linux-x86_64-580.65.06.run
chmod +x NVIDIA-Linux-*.run
sudo systemctl isolate multi-user.target
sudo ./NVIDIA-Linux-*.run --dkms --kernel-module-type=open
   
5.3 Optional DKMS hook
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo install -m 0755 /dev/stdin /etc/kernel/postinst.d/99-rebuild-nvidia <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
KERN="${1:-$(uname -r)}"
/usr/sbin/dkms autoinstall -k "${KERN}"
/usr/bin/dracut --regenerate-all --force
EOF```
***
### **6) Hybrid-GPU Configuration**
GTX 1060 (nouveau) drives desktop; RTX 5060 Ti (NVIDIA) handles offload.

**6.1 Keep Pascal on nouveau**
`/etc/modprobe.d/nvidia-exclude-pascal.conf`
   
options nvidia NVreg_ExcludedGpus=0x1c03
code Code
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     **6.2 Load nouveau early**
`/etc/modules-load.d/00-nouveau-first.conf`
   
nouveau```
6.3 Nouveau power tuning
/etc/modprobe.d/nouveau-power.conf
code Code
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     options nouveau config=NvClkMode=15
   
6.4 CUDA preference
/etc/profile.d/cuda-primary-only.sh
code Code
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     export CUDA_VISIBLE_DEVICES=0
   
6.5 Late-boot re-bind
/etc/udev/rules.d/71-nouveau-gtx1060.rules
code Code
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{device}=="0x1c03", \
  RUN+="/bin/sh -c 'echo 0000:23:00.0 > /sys/bus/pci/drivers/nouveau/bind'"
   
6.6 Initramfs regen
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo dracut --regenerate-all --force
```***
### **7) GPU Usage & PRIME Offload Workflow**
*   **GTX 1060** → nouveau / Mesa NVK (desktop)
*   **RTX 5060 Ti** → NVIDIA proprietary (offload + CUDA)

**Check**
```bash
glxinfo -B | grep -E "OpenGL (vendor|renderer|version)"
vulkaninfo --summary | grep -E "deviceName|driverName|driverInfo" -A1
nvidia-smi
   
Run on RTX 5060 Ti
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia <app>
   
Alias:
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     echo "alias prime='__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia'" >> ~/.bashrc
   
7.1 Safe Update + Snapshot
Script: /usr/local/sbin/gpu-safe-update:
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     #!/usr/bin/env bash
set -euo pipefail
STAMP="$(date +'%Y-%m-%d %H:%M')"
DESC="Pre-reboot GPU checkpoint ${STAMP}"
if command -v snapper >/dev/null; then
  for C in root home; do
    if snapper list-configs | awk 'NR>2{print $1}' | grep -qx "$C"; then
      snapper -c "$C" create --description "$DESC" --cleanup-algorithm number
    fi
  done
fi
if command -v dkms >/dev/null; then
  dkms autoinstall -k "$(uname -r)"
fi
if command -v dracut >/dev/null; then
  dracut --regenerate-all --force
fi
for m in nvidia nvidia_drm nvidia_modeset nvidia_uvm; do
  modprobe -n "$m" >/dev/null || exit 1
done
echo "[✓] GPU safe-update complete"
   
Usage:
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo dnf5 upgrade
sudo gpu-safe-update
sudo systemctl reboot
   

8) System Services: Whisper API
This section details the production-ready local Whisper transcription and diarization FastAPI service running on the host. For detailed developer guide, see the separate "ice whisper usage guide" document.

8.1 Overview
OS: Fedora 42
GPU: NVIDIA RTX 5060 Ti, driver 580.65.06 (CUDA runtime 13.0)
Python: CPython 3.12 venv at ~/.venvs/whisper312
Whisper engines: faster-whisper + ctranslate2 + whisper-ctranslate2
Serving modes: CLI wrapper whisperx and production FastAPI service at http://127.0.0.1:8765
Features: Real-time transcription, speaker diarization, multi-format output, async job processing

8.2 Key packages (production-ready)
faster-whisper==1.2.0
ctranslate2==4.6.0  
whisper-ctranslate2==0.5.4
nvidia-cublas-cu12 (user-space cuBLAS)
nvidia-cudnn-cu12==9.* (user-space cuDNN)
API stack: fastapi==0.116.1, uvicorn[standard]==0.35.0, httpx==0.28.1, python-multipart==0.0.20
Diarization: torch CUDA 12.1 wheels, pyannote.audio>=3.1,<4.0

8.3 Central Token Management
**New centralized HuggingFace token system** - single location for all components:
Token storage: ~/.config/whisper/token
Management tool: ~/bin/whisper-token
```bash
# Set token (enables diarization)
whisper-token set hf_your_token_here
# Check token status  
whisper-token show
# Test diarization
whisper-token test
# Disable diarization
whisper-token clear
```

8.4 FastAPI Service Architecture
App file: ~/app.py (Full-featured FastAPI application)
Service file: ~/.config/systemd/user/whisper-api.service  
Launcher: ~/.local/bin/whisper-api (reads central token config)
Listen: 127.0.0.1:8765
Model: medium (GPU-accelerated)

**Production Environment:**
```bash
WHISPER_MODEL=medium
WHISPER_COMPUTE=float16
WHISPER_DEVICE=cuda
WHISPER_LANGUAGE=en
WHISPER_DIARIZE=true
WHISPER_DEFAULT_FORMAT=json
```

**API Endpoints:**
- `GET /health` → Service health check
- `POST /v1/transcribe` → Synchronous transcription with file upload/URL
  - Parameters: file, audio_url, language, vad, diarize, num_speakers, format
- `POST /v2/transcript` → Async job creation (AssemblyAI-compatible)
- `GET /v2/transcript/{id}` → Job status/results polling

8.5 Usage Examples
**CLI (with auto-diarization):**
```bash
whisperx sample.wav --model medium --language en --compute_type float16 --output_format json
```

**API Sync (JSON with speakers):**
```bash
curl -F "file=@sample.wav" -F "diarize=true" -F "num_speakers=2" \
  http://127.0.0.1:8765/v1/transcribe | jq .
```

**API Sync (Text only):**
```bash  
curl -F "file=@sample.wav" -F "format=text" http://127.0.0.1:8765/v1/transcribe
```

**API Async (AAI-style):**
```bash
# Create job
job_id=$(curl -sX POST http://127.0.0.1:8765/v2/transcript \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"http://example.com/audio.wav", "speaker_labels": true}' | jq -r .id)

# Poll results  
curl -s http://127.0.0.1:8765/v2/transcript/$job_id | jq .
```

8.6 Service Management
```bash
# Service control
systemctl --user restart whisper-api.service
systemctl --user status whisper-api.service
journalctl --user -u whisper-api.service -f

# Token management (auto-restarts service)
whisper-token set hf_new_token    # Enable diarization
whisper-token clear               # Disable diarization
```

8.7 Production Features
✅ **GPU Acceleration**: RTX 5060 Ti with CUDA float16 compute
✅ **Speaker Diarization**: Automatic speaker identification and labeling
✅ **Multiple Formats**: JSON, text, VTT, SRT output formats
✅ **Async Processing**: Background job processing with status polling
✅ **Robust Error Handling**: Graceful fallbacks and detailed error responses
✅ **Security**: Token-based auth with centralized management
✅ **Auto-restart**: Service auto-recovery and token change handling

8.8 Default Configuration Rationale
- **English language**: Skips auto-detection, faster processing for primary use case
- **Diarization enabled**: Automatic speaker labels when token available
- **Medium model**: Best balance of accuracy and speed on this hardware
- **JSON format**: Structured output with timing and speaker metadata

9) Quick Diagnostics
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     lspci -nnk | grep -A3 -E 'VGA|3D|Display'
lsmod | grep -E 'nvidia|nouveau'
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo -B | grep 'OpenGL renderer'
nvidia-smi
   

10) System Performance
code Bash
downloadcontent_copyexpand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
     sudo systemctl disable --now systemd-oomd
   

11) Discord Setup (RPM + Wayland + Hybrid GPU)
This section details the production-ready Discord setup optimized for the hybrid GPU configuration and Wayland session.

11.1 Overview
Package: discord RPM (discord-0.0.103-1.fc42.x86_64)
Binary: /usr/bin/Discord
Session: Wayland with Electron/Chromium flags
GPU: Runs on GTX 1060 (nouveau) desktop, with automatic GPU fallback
Launcher: Smart wrapper with crash detection and logging

11.2 Smart Launcher
Location: ~/bin/discord
Features: Wayland detection, GPU acceleration with fallback, detailed logging
```bash
# Usage
discord
```

11.3 Key Features
✅ **Wayland Support**: Auto-detects Wayland and applies safe Electron flags
✅ **GPU Acceleration**: Tries GPU first, falls back to --disable-gpu on crashes  
✅ **Hybrid GPU Safe**: Runs on nouveau (GTX 1060) without interfering with CUDA RTX 5060 Ti
✅ **Crash Recovery**: Automatic retry with safer flags if initial launch fails
✅ **Debug Logging**: Detailed logs in ~/.local/state/discord/ for troubleshooting
✅ **Desktop Integration**: Consistent behavior between terminal and GUI launches
✅ **Autostart Support**: Optional login autostart configured

11.4 Troubleshooting
```bash
# Check if running
pgrep -ax Discord

# View recent logs  
tail -60 ~/.local/state/discord/rpm-gpu.log
tail -60 ~/.local/state/discord/rpm-nogpu.log

# Manual launch with NVIDIA offload (if needed)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia discord
```

11.5 Integration Notes
- Desktop file overridden in ~/.local/share/applications/discord.desktop
- Autostart configured in ~/.config/autostart/discord.desktop
- Logs stored in ~/.local/state/discord/ (safe with Btrfs snapshots)
- Compatible with existing snapshot workflow

12) Claude Code & MCP Servers
This system includes a comprehensive setup of Model Context Protocol (MCP) servers for enhanced AI-assisted development.

**Configured MCP Servers (USER scope - globally available):**
- **Brave Search MCP** - Real-time web search and local business discovery
- **Docker MCP** - Container lifecycle management and Docker operations  
- **Playwright MCP** - Browser automation, testing, and web scraping (containerized)
- **Serena MCP** - Semantic code analysis, editing, and project management

**Status Check:**
```bash
claude mcp list
```

**Complete Documentation:** [MCP_SERVERS.md](./MCP_SERVERS.md)

All MCP servers are optimized for Fedora 42 with hybrid GPU configuration and follow containerization-first principles for maximum reliability.

13) Supabase CLI
This system has Supabase CLI installed for local development and database management.

**Version:** 2.34.3
**Installation Date:** August 14, 2025
**Installation Method:** RPM from official GitHub releases

**Installation Command Used:**
```bash
# Grab the latest release RPM dynamically (x86_64)
LATEST_RPM_URL=$(curl -s https://api.github.com/repos/supabase/cli/releases/latest \
  | grep browser_download_url \
  | grep -i 'linux_amd64\.rpm' \
  | cut -d '"' -f 4)

curl -LO "$LATEST_RPM_URL"
sudo rpm -Uvh "$(basename "$LATEST_RPM_URL")"
```

**Available Commands:**
- `supabase init` - Initialize a new Supabase project
- `supabase start` - Start local development stack (requires Docker)
- `supabase stop` - Stop local development stack
- `supabase db reset` - Reset local database
- `supabase migration new` - Create a new migration
- `supabase login` - Authenticate with Supabase platform
- `supabase link` - Link to an existing Supabase project
- `supabase functions` - Manage Edge Functions

**Verify Installation:**
```bash
supabase --version
```

14) References & Resources
Installation Guide: 
https://sysguides.com/install-fedora-42-with-full-disk-encryption-snapshot-and-rollback-support 
 How to Install Fedora 42 with Full Disk Encryption, Snapshots, and Rollback (LUKS2 + TPM2 Guide)
Post-Update Fix Video: 
Fedora 42 FDE: GRUB boot failure after update – here's the fix
Post-Installation Setup Video: 
9 Things to Do After Installing Fedora 42 (Post Setup Guide)
Performance Tweak Video: 
18 Things You MUST DO After Installing Fedora 42 (Right Now!)
RPM Fusion: https://rpmfusion.org/
Flathub: https://flathub.org/
NVIDIA CUDA Downloads: https://developer.nvidia.com/cuda-downloads
Claude Code MCP Documentation: https://docs.anthropic.com/en/docs/claude-code/mcp


