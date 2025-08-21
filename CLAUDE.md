# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Nature

This is a **PRP (Product Requirement Prompt) Framework** repository, not a traditional software project. The core concept: **"PRP = PRD + curated codebase intelligence + agent/runbook"** - designed to enable AI agents to ship production-ready code on the first pass.

## Core Architecture

### Command-Driven System

- **pre-configured Claude Code commands** in `.claude/commands/`
- Commands organized by function:
  - `PRPs/` - PRP creation and execution workflows
  - `development/` - Core development utilities (prime-core, onboarding, debug)
  - `code-quality/` - Review and refactoring commands
  - `rapid-development/experimental/` - Parallel PRP creation and hackathon tools
  - `git-operations/` - Conflict resolution and smart git operations

### Template-Based Methodology

- **PRP Templates** in `PRPs/templates/` follow structured format with validation loops
- **Context-Rich Approach**: Every PRP must include comprehensive documentation, examples, and gotchas
- **Validation-First Design**: Each PRP contains executable validation gates (syntax, tests, integration)

### AI Documentation Curation

- `PRPs/ai_docs/` contains curated Claude Code documentation for context injection
- `claude_md_files/` provides framework-specific CLAUDE.md examples

## Development Commands

### PRP Execution

```bash
# Interactive mode (recommended for development)
uv run PRPs/scripts/prp_runner.py --prp [prp-name] --interactive

# Headless mode (for CI/CD)
uv run PRPs/scripts/prp_runner.py --prp [prp-name] --output-format json

# Streaming JSON (for real-time monitoring)
uv run PRPs/scripts/prp_runner.py --prp [prp-name] --output-format stream-json
```

### Key Claude Commands

- `/prp-base-create` - Generate comprehensive PRPs with research
- `/prp-base-execute` - Execute PRPs against codebase
- `/prp-planning-create` - Create planning documents with diagrams
- `/prime-core` - Prime Claude with project context
- `/review-staged-unstaged` - Review git changes using PRP methodology

## Critical Success Patterns

### The PRP Methodology

1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance

### PRP Structure Requirements

- **Goal**: Specific end state and desires
- **Why**: Business value and user impact
- **What**: User-visible behavior and technical requirements
- **All Needed Context**: Documentation URLs, code examples, gotchas, patterns
- **Implementation Blueprint**: Pseudocode with critical details and task lists
- **Validation Loop**: Executable commands for syntax, tests, integration

### Validation Gates (Must be Executable)

```bash
# Level 1: Syntax & Style
ruff check --fix && mypy .

# Level 2: Unit Tests
uv run pytest tests/ -v

# Level 3: Integration
uv run uvicorn main:app --reload
curl -X POST http://localhost:8000/endpoint -H "Content-Type: application/json" -d '{...}'

# Level 4: Deployment
# mcp servers, or other creative ways to self validate
```

## Anti-Patterns to Avoid

- L Don't create minimal context prompts - context is everything - the PRP must be comprehensive and self-contained, reference relevant documentation and examples.
- L Don't skip validation steps - they're critical for one-pass success - The better The AI is at running the validation loop, the more likely it is to succeed.
- L Don't ignore the structured PRP format - it's battle-tested
- L Don't create new patterns when existing templates work
- L Don't hardcode values that should be config
- L Don't catch all exceptions - be specific

## Working with This Framework

### When Creating new PRPs

1. **Context Process**: New PRPs must consist of context sections, Context is King!
2.

### When Executing PRPs

1. **Load PRP**: Read and understand all context and requirements
2. **ULTRATHINK**: Create comprehensive plan, break down into todos, use subagents, batch tool etc check prps/ai_docs/
3. **Execute**: Implement following the blueprint
4. **Validate**: Run each validation command, fix failures
5. **Complete**: Ensure all checklist items done

### Command Usage

- Read the .claude/commands directory
- Access via `/` prefix in Claude Code
- Commands are self-documenting with argument placeholders
- Use parallel creation commands for rapid development
- Leverage existing review and refactoring commands

## Project Structure Understanding

```
PRPs-agentic-eng/
.claude/
  commands/           # 28+ Claude Code commands
  settings.local.json # Tool permissions
PRPs/
  templates/          # PRP templates with validation
  scripts/           # PRP runner and utilities
  ai_docs/           # Curated Claude Code documentation
   *.md               # Active and example PRPs
 claude_md_files/        # Framework-specific CLAUDE.md examples
 pyproject.toml         # Python package configuration
```

Remember: This framework is about **one-pass implementation success through comprehensive context and validation**. Every PRP should contain the exact context for an AI agent to successfully implement working code in a single pass.

## Whisper API Project Overview

This is a FastAPI-based Whisper transcription service that provides GPU-accelerated speech-to-text capabilities with optional speaker diarization. The service exposes REST API endpoints for both synchronous and asynchronous audio transcription.

## Quick Usage Guide

### Starting the Service
```bash
# One-command start (handles environment automatically)
./start_whisper.sh start

# Check if running
./start_whisper.sh status
```

### Using the Service

**Basic Transcription (no speakers):**
```bash
curl -X POST http://localhost:8765/v1/transcribe -F "file=@audio.wav"
```

**With Speaker Identification:**
```bash
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=true"
```

**From Python (any version):**
```python
from whisper_client import WhisperClient
client = WhisperClient()
result = client.transcribe("audio.wav", diarize=True)
```

**Key Point**: Diarization is optional per request - use `diarize=false` (default) for speed, `diarize=true` for speaker identification.

## Whisper API Core Architecture

### Main Components

1. **FastAPI Application** (`main.py`): 
   - RESTful API service with two transcription modes (sync/async)
   - Uses `faster-whisper` for transcription (GPU-accelerated)
   - Optional speaker diarization via `pyannote.audio` pipeline
   - Job queue system for async processing

2. **Test Script** (`test_transcribe.py`): 
   - Debugging utility for testing transcription with different settings
   - Tests VAD (Voice Activity Detection) filter configurations

### Key Dependencies

- `faster-whisper`: GPU-accelerated Whisper implementation
- `FastAPI`: Web framework for REST API
- `pyannote.audio`: Speaker diarization (optional)
- `httpx`: Async HTTP client for downloading audio
- `uvicorn`: ASGI server
- `numpy`: Audio processing
- CUDA/cuDNN: GPU acceleration libraries

### Environment Configuration

The service uses environment variables for configuration:
- `WHISPER_MODEL`: Model size (default: "tiny")
- `WHISPER_DEVICE`: Computing device ("cuda" or "cpu")
- `WHISPER_COMPUTE`: Compute type (e.g., "float16")
- `WHISPER_LANGUAGE`: Default language (default: "en")
- `WHISPER_DIARIZE`: Enable speaker diarization (default: "true")
- `WHISPER_DEFAULT_FORMAT`: Default output format (default: "json")

### Authentication

- HuggingFace token required for diarization features
- Token stored in `~/.config/whisper/token` file
- Format: `HF_TOKEN=hf_xxxxx`

## Common Whisper API Development Tasks

### Running the Service

```bash
# Start the API server
python main.py
# Server runs on http://127.0.0.1:8765
```

### Testing Transcription

```bash
# Test with a specific audio file
python test_transcribe.py /path/to/audio.wav
```

### API Endpoints

- `GET /`: Service information
- `GET /health`: Health check
- `POST /v1/transcribe`: Synchronous transcription
- `POST /v2/transcript`: Asynchronous transcription (AssemblyAI compatible)
- `GET /v2/transcript/{job_id}`: Get async job status/results

### Important Transcription Settings

The service has been configured with specific settings to improve transcription accuracy:
- VAD filter disabled by default (was filtering out valid speech)
- Initial prompt added to guide model
- More lenient thresholds for compression ratio and log probability
- Lower no-speech threshold for better speech detection

### Output Formats

Supports multiple output formats via the `format` parameter:
- `json`: Full transcription with segments and metadata
- `text`: Plain text output
- `vtt`: WebVTT subtitle format
- `srt`: SRT subtitle format

## Whisper API Development Notes

- The service uses tempfile for handling uploaded/downloaded audio files
- Background tasks handle async transcription jobs
- Job results are stored in memory (`jobs_storage` dictionary)
- Audio files are cleaned up after processing
- CUDA library paths are configured in test script for GPU support

## Speaker Diarization Setup

Diarization requires specific version combinations due to pyannote compatibility issues:

### Quick Setup
```bash
# Option 1: Install in current environment (Python 3.11 recommended)
./install_diarization.sh

# Option 2: Create dedicated virtual environment
./setup_venv.sh
source ~/.venvs/whisper-diarize/bin/activate
```

### Known Working Combinations
- Python 3.11 + torch 2.2.0 + pyannote.audio 3.1.1 (recommended)
- Python 3.10 + torch 2.1.0 + pyannote.audio 3.0.1
- Python 3.12 + torch 2.3.0 + pyannote.audio 3.1.1 (may have issues)

### Troubleshooting Diarization

1. **Test current setup:**
   ```bash
   python test_diarization.py
   ```

2. **Common issues:**
   - Missing HF token: Add to `~/.config/whisper/token`
   - License not accepted: Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Version conflicts: Use setup_venv.sh for clean environment

3. **Check service health:**
   ```bash
   curl http://localhost:8765/health | jq .diarization
   ```

The service will attempt multiple model versions and provide detailed error messages if diarization fails to load.
