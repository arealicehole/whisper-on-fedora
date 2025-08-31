# Whisper API Branch Organization Analysis

## Executive Summary
The whisper-api codebase contains 140+ files, but only 7-9 are essential for production runtime. This analysis provides a complete categorization for implementing separate `main` (runtime) and `dev` (full) branches.

## Runtime Essential Files (Main Branch)

### Core Application (4 files)
- `main.py` - FastAPI application server
- `gpu_validator.py` - GPU enforcement module  
- `sitecustomize.py` - Python auto-loader for Blackwell GPU fixes
- `startup.py` - Application startup wrapper (optional)

### Dependencies (2-3 files)
- `requirements_api.txt` - Core runtime dependencies
- `requirements_blackwell.txt` - Blackwell GPU specific (conditional)
- `whisper_client.py` - Python client library

### Legal (1 file)
- `LICENSE` - MIT license (required for distribution)

**Total: 7-9 files for production deployment**

## Development Files (Dev Branch Only)

### Documentation (38 files)
- Root: README.md, QUICKSTART.md, ONBOARDING.md, etc. (6 files)
- docs-archive/: 30+ markdown files across subdirectories
- Implementation guides, troubleshooting, Docker docs

### Testing (8 files)
- test_*.py scripts for GPU, diarization, MP3, WhisperX testing

### Claude Code Configuration (40+ files)
- .claude/commands/: Development workflows, PRP commands, git operations
- Organized in: code-quality/, development/, git-operations/, prp-commands/, rapid-development/, typescript/

### Prompt Engineering Resources (20+ files)
- PRPs/templates/: 6 template files
- PRPs/ai_docs/: Technical documentation
- PRPs/*.md: Specific implementation PRPs

### Development Tools (15 files)
- Setup scripts: setup_venv.sh, setup_blackwell_venv.sh
- Fix utilities: fix_pyannote_blackwell.py, torchvision_fix.py, patch_pyannote.py
- Alternative implementations: main_pyannote.py, whisperx_diarization.py
- Validation tools: validate_blackwell.py

### Archives (15+ files)
- docker-historic/: Historical Docker configurations
- Multiple Dockerfile versions and compose files

### Configuration Files
- .gitignore
- .serena/project.yml (AI assistant config)

### Empty Directories (for runtime use)
- logs/ - Runtime log storage
- audio_files/ - Test audio storage  
- volumes/ - Docker volume mounts

## Key Insights

### File Distribution
- **93% development files** (133 of 142 files)
- **7% runtime essential** (9 files maximum)
- Development files span 6 major categories

### Dependency Analysis
- Runtime requires only: FastAPI, Uvicorn, PyTorch, faster-whisper, httpx
- Development adds: Testing frameworks, documentation tools, build utilities
- Blackwell GPU support integrated into runtime (not separate)

### Branch Strategy Rationale
1. **Main branch** deployment reduces attack surface by 93%
2. **Dev branch** preserves full development context and tooling
3. Automated sync via GitHub Actions maintains consistency
4. No manual file copying between branches needed

## Implementation Priority

1. **Immediate**: Backup current state before reorganization
2. **Phase 1**: Create dev branch with all files
3. **Phase 2**: Clean main branch to runtime only
4. **Phase 3**: Setup automation and validation
5. **Ongoing**: Maintain via established workflow

## Risk Mitigation

- Full backup before changes
- Incremental implementation with validation gates
- Automated workflows prevent human error
- Clear documentation for team onboarding
- Rollback procedure documented

## Metrics for Success

- Main branch: ≤10 files
- Dev branch: 140+ files
- Zero merge conflicts after setup
- Production deployment time: <30 seconds
- Development environment intact