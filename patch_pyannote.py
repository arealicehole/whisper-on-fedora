#!/usr/bin/env python3
"""
Patch PyAnnote to work with torchaudio 2.8.0 (nightly)
The issue is that torchaudio.AudioMetaData was removed in newer versions.
"""

import sys
import torchaudio
from dataclasses import dataclass

# Create a compatibility shim for AudioMetaData
@dataclass
class AudioMetaData:
    """Compatibility shim for removed torchaudio.AudioMetaData"""
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "UNKNOWN"

# Monkey-patch torchaudio to include AudioMetaData
if not hasattr(torchaudio, 'AudioMetaData'):
    torchaudio.AudioMetaData = AudioMetaData
    print("✅ Patched torchaudio.AudioMetaData for PyAnnote compatibility")

# Now try importing PyAnnote
try:
    from pyannote.audio import Pipeline
    print("✅ PyAnnote imported successfully with patch!")
    
    # Test basic functionality
    print("\n📊 Testing PyAnnote components:")
    print(f"- Pipeline class: {Pipeline}")
    
except ImportError as e:
    print(f"❌ PyAnnote import failed even with patch: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print("\n🎉 Patch successful! PyAnnote should now work with torchaudio 2.8.0")
print("\nTo use this patch in your code, add these lines at the top:")
print("```python")
print("import torchaudio")
print("from dataclasses import dataclass")
print("")
print("@dataclass")
print("class AudioMetaData:")
print("    sample_rate: int")
print("    num_frames: int")
print("    num_channels: int")
print("    bits_per_sample: int = 16")
print("    encoding: str = 'UNKNOWN'")
print("")
print("if not hasattr(torchaudio, 'AudioMetaData'):")
print("    torchaudio.AudioMetaData = AudioMetaData")
print("```")