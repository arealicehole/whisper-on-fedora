# Contributing to Whisper API

Thank you for your interest in contributing to Whisper API! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in [Issues](https://github.com/arealicehole/whisper-on-fedora/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Relevant logs or error messages

### Suggesting Enhancements

We'd love to hear your ideas! Please:

1. Check the [Roadmap](README.md#roadmap) and existing issues
2. Open a new issue with the `enhancement` label
3. Describe the feature and its use case
4. Explain why this would be useful to most users

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up your development environment**:
   ```bash
   git clone https://github.com/arealicehole/whisper-on-fedora.git
   cd whisper-on-fedora
   ./setup_isolated_python.sh
   source ~/.venvs/whisper-diarize/bin/activate
   pip install -r requirements_diarization.txt
   ```

3. **Make your changes**:
   - Write clean, readable code
   - Follow Python PEP 8 style guide
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**:
   ```bash
   # Test basic functionality
   python test_transcribe.py sample.wav
   
   # Test diarization
   python test_diarization.py
   
   # Test API endpoints
   ./start_whisper.sh start
   curl http://localhost:8765/health
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. **Push to your fork** and submit a pull request

### Code Style

- Use Python 3.11+ features where appropriate
- Follow PEP 8 (use `black` formatter if possible)
- Add type hints for function parameters and returns
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

Example:
```python
def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    diarize: bool = False
) -> Dict[str, Any]:
    """
    Transcribe an audio file with optional diarization.
    
    Args:
        audio_path: Path to the audio file
        language: Language code (e.g., 'en')
        diarize: Enable speaker diarization
        
    Returns:
        Dictionary containing transcription results
    """
    # Implementation here
```

### Testing

Before submitting a PR, ensure:

- [ ] All existing tests pass
- [ ] New features have test coverage
- [ ] API endpoints return expected responses
- [ ] Documentation is updated
- [ ] Code follows style guidelines

### Documentation

Update documentation for:

- New features or API endpoints
- Changed behavior
- New configuration options
- Installation or setup changes

Documentation locations:
- `README.md` - Main documentation
- `QUICKSTART.md` - Quick reference
- `examples/` - Usage examples
- Docstrings - In-code documentation

## Development Tips

### Running in Development Mode

```bash
# Start with auto-reload for development
uvicorn main:app --reload --host 0.0.0.0 --port 8765
```

### Debugging Diarization Issues

```bash
# Use the diagnostic tool
python test_diarization.py

# Check specific model loading
python -c "from pyannote.audio import Pipeline; print('OK')"
```

### Performance Testing

```python
import time
from whisper_client import WhisperClient

client = WhisperClient()

# Benchmark transcription
start = time.time()
result = client.transcribe("test.wav")
print(f"Transcription: {time.time() - start:.2f}s")

# Benchmark with diarization
start = time.time()
result = client.transcribe("test.wav", diarize=True)
print(f"With diarization: {time.time() - start:.2f}s")
```

## Community

- Join discussions in [GitHub Discussions](https://github.com/arealicehole/whisper-on-fedora/discussions)
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Be respectful and constructive
- Help others when you can

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for helping make Whisper API better!