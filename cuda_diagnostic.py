#!/usr/bin/env python3
"""
CUDA Diagnostic and Compatibility Checker for Diarization
Part of the Diarization Testing & Hardening Initiative
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class CUDADiagnostic:
    """Comprehensive CUDA diagnostics for PyTorch and Pyannote compatibility"""
    
    def __init__(self):
        self.results = {
            "system": {},
            "cuda": {},
            "python": {},
            "pytorch": {},
            "compatibility": {},
            "recommendations": []
        }
        
    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def check_system_info(self):
        """Gather system information"""
        print("\n🔍 Checking System Information...")
        
        self.results["system"] = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": sys.version,
            "python_executable": sys.executable
        }
        
        # Check for CUDA environment variables
        cuda_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"]
        self.results["system"]["cuda_env_vars"] = {
            var: os.environ.get(var, "Not set") for var in cuda_vars
        }
        
    def check_nvidia_driver(self):
        """Check NVIDIA driver and GPU information"""
        print("\n🎮 Checking NVIDIA Driver and GPU...")
        
        success, output = self.run_command(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"])
        
        if success:
            lines = output.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                if len(parts) >= 3:
                    self.results["cuda"]["gpu_name"] = parts[0].strip()
                    self.results["cuda"]["driver_version"] = parts[1].strip()
                    self.results["cuda"]["compute_capability"] = parts[2].strip()
                    print(f"  ✓ GPU: {parts[0].strip()}")
                    print(f"  ✓ Driver: {parts[1].strip()}")
                    print(f"  ✓ Compute Capability: {parts[2].strip()}")
        else:
            self.results["cuda"]["error"] = "nvidia-smi not found or failed"
            print("  ✗ nvidia-smi failed")
            
    def check_cuda_toolkit(self):
        """Check CUDA toolkit installation"""
        print("\n🔧 Checking CUDA Toolkit...")
        
        # Check nvcc
        success, output = self.run_command(["nvcc", "--version"])
        if success:
            for line in output.split('\n'):
                if 'release' in line.lower():
                    self.results["cuda"]["toolkit_version"] = line.strip()
                    print(f"  ✓ CUDA Toolkit: {line.strip()}")
                    break
        else:
            print("  ⚠️  nvcc not found (CUDA toolkit may not be in PATH)")
            
    def check_pytorch_compatibility(self):
        """Check PyTorch and CUDA compatibility"""
        print("\n🔗 Checking PyTorch Compatibility...")
        
        try:
            import torch
            self.results["pytorch"]["installed"] = True
            self.results["pytorch"]["version"] = torch.__version__
            self.results["pytorch"]["cuda_available"] = torch.cuda.is_available()
            self.results["pytorch"]["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            
            print(f"  ✓ PyTorch Version: {torch.__version__}")
            print(f"  {'✓' if torch.cuda.is_available() else '✗'} CUDA Available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"  ✓ PyTorch CUDA Version: {torch.version.cuda}")
                
                # Test CUDA operations
                try:
                    # Simple tensor operation
                    test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                    result = test_tensor * 2
                    self.results["pytorch"]["cuda_test"] = "passed"
                    print("  ✓ CUDA tensor operations: Working")
                except Exception as e:
                    self.results["pytorch"]["cuda_test"] = f"failed: {str(e)}"
                    print(f"  ✗ CUDA tensor operations failed: {str(e)}")
                    
                    # This is likely the issue you're facing
                    if "no kernel image" in str(e):
                        self.results["compatibility"]["issue"] = "CUDA architecture mismatch"
                        print("\n  ⚠️  CRITICAL: CUDA architecture mismatch detected!")
                        print("     Your GPU requires a different PyTorch build.")
                
                # Check GPU properties
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    self.results["pytorch"]["gpu_properties"] = {
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory_gb": props.total_memory / (1024**3)
                    }
                    print(f"  ✓ GPU in PyTorch: {props.name}")
                    print(f"  ✓ Compute Capability: {props.major}.{props.minor}")
                    
        except ImportError:
            self.results["pytorch"]["installed"] = False
            print("  ✗ PyTorch not installed")
            
    def check_pyannote_compatibility(self):
        """Check pyannote.audio installation and compatibility"""
        print("\n🎤 Checking Pyannote Compatibility...")
        
        try:
            import pyannote.audio
            self.results["pyannote"] = {
                "installed": True,
                "version": pyannote.audio.__version__
            }
            print(f"  ✓ Pyannote.audio Version: {pyannote.audio.__version__}")
            
            # Try to load pipeline
            token_file = Path.home() / ".config" / "whisper" / "token"
            if token_file.exists():
                with open(token_file) as f:
                    for line in f:
                        if line.startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                            os.environ["HF_TOKEN"] = token
                            
                try:
                    from pyannote.audio import Pipeline
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=token
                    )
                    self.results["pyannote"]["pipeline_load"] = "success"
                    print("  ✓ Pipeline loading: Success")
                except Exception as e:
                    self.results["pyannote"]["pipeline_load"] = f"failed: {str(e)}"
                    print(f"  ✗ Pipeline loading failed: {str(e)[:100]}")
            else:
                print("  ⚠️  No HF token found")
                
        except ImportError:
            self.results["pyannote"] = {"installed": False}
            print("  ✗ Pyannote.audio not installed")
            
    def analyze_compatibility(self):
        """Analyze compatibility and provide recommendations"""
        print("\n📊 Compatibility Analysis...")
        
        # Determine the issue
        issues = []
        
        # Check CUDA architecture compatibility
        if "pytorch" in self.results and self.results["pytorch"].get("cuda_test") == "failed":
            if "no kernel image" in self.results["pytorch"].get("cuda_test", ""):
                issues.append("cuda_architecture_mismatch")
                
        # Check compute capability
        if "cuda" in self.results and "compute_capability" in self.results["cuda"]:
            cc = self.results["cuda"]["compute_capability"]
            # RTX 5060 Ti should have compute capability 8.9 or 9.0
            if cc and float(cc.replace(".", "")) >= 89:
                issues.append("newer_gpu_architecture")
                
        self.results["compatibility"]["issues"] = issues
        
        # Generate recommendations
        self.generate_recommendations(issues)
        
    def generate_recommendations(self, issues: List[str]):
        """Generate specific recommendations based on issues found"""
        print("\n💡 Recommendations:")
        
        recommendations = []
        
        if "cuda_architecture_mismatch" in issues or "newer_gpu_architecture" in issues:
            recommendations.append({
                "priority": "HIGH",
                "action": "Reinstall PyTorch with correct CUDA version",
                "commands": [
                    "# First, uninstall existing PyTorch",
                    "pip uninstall torch torchvision torchaudio -y",
                    "",
                    "# For RTX 5060 Ti (Ada Lovelace architecture), use CUDA 12.1",
                    "pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121",
                    "",
                    "# Alternative: Try nightly build for newer GPU support",
                    "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121"
                ]
            })
            
        if not self.results.get("pytorch", {}).get("installed"):
            recommendations.append({
                "priority": "HIGH", 
                "action": "Install PyTorch",
                "commands": [
                    "pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121"
                ]
            })
            
        self.results["recommendations"] = recommendations
        
        for rec in recommendations:
            print(f"\n  [{rec['priority']}] {rec['action']}")
            if "commands" in rec:
                print("  Commands to run:")
                for cmd in rec["commands"]:
                    if cmd:
                        print(f"    {cmd}")
                        
    def save_report(self):
        """Save diagnostic report"""
        report_path = Path("cuda_diagnostic_report.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📄 Full report saved to: {report_path}")
        
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("=" * 60)
        print("CUDA Diagnostic Tool for Whisper Diarization")
        print("=" * 60)
        
        self.check_system_info()
        self.check_nvidia_driver()
        self.check_cuda_toolkit()
        self.check_pytorch_compatibility()
        self.check_pyannote_compatibility()
        self.analyze_compatibility()
        self.save_report()
        
        print("\n" + "=" * 60)
        print("Diagnostic Complete")
        print("=" * 60)
        
        # Return exit code based on critical issues
        if self.results["compatibility"].get("issues"):
            return 1
        return 0


def main():
    """Main entry point"""
    diagnostic = CUDADiagnostic()
    exit_code = diagnostic.run_full_diagnostic()
    
    # Print quick fix if CUDA architecture mismatch detected
    if "cuda_architecture_mismatch" in diagnostic.results["compatibility"].get("issues", []):
        print("\n🚀 QUICK FIX:")
        print("-" * 40)
        print("Run these commands to fix CUDA compatibility:")
        print("\n# 1. Activate your virtual environment")
        print("source ~/.venvs/whisper-diarize/bin/activate")
        print("\n# 2. Reinstall PyTorch with CUDA 12.1 support")
        print("pip uninstall torch torchvision torchaudio -y")
        print("pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121")
        print("\n# 3. Restart the Whisper service")
        print("./start_whisper.sh restart")
        print("-" * 40)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()