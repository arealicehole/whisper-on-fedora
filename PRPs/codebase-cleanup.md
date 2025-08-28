name: "Whisper API Codebase Cleanup and Organization"
description: |
  Comprehensive cleanup and reorganization of the Whisper API codebase to remove obsolete files,
  organize documentation, and create a clean structure for future development.

---

## Goal

**Feature Goal**: Transform the current cluttered Whisper API codebase into a clean, organized structure by removing obsolete files, archiving documentation, and preserving only essential components for local operation.

**Deliverable**: A streamlined codebase with:
- Only essential files for running the Whisper API locally
- All markdown documentation moved to `docs-archive/` folder
- All Docker-related files moved to `docker-historic/` folder
- Removal of diagnostic, test, and one-time setup scripts

**Success Definition**: The API runs successfully with `python main.py` using only the preserved files, with all documentation archived for reference.

## User Persona

**Target User**: Developer maintaining and extending the Whisper API locally

**Use Case**: Running a clean, production-ready Whisper API without Docker dependencies

**User Journey**: 
1. Navigate to clean project root
2. Activate virtual environment
3. Run `python main.py` to start the API
4. Access documentation in organized folders when needed

**Pain Points Addressed**: 
- Cluttered root directory with 82+ files
- Mixed diagnostic/production code
- Scattered documentation making navigation difficult
- Obsolete Docker configurations causing confusion

## Why

- **Clean Development**: Reduce cognitive load from 82 files to ~10 essential files
- **Clear Purpose**: Each remaining file has a specific, necessary function
- **Documentation Preservation**: All docs archived but accessible for reference
- **Future Ready**: Clean base for building new features without legacy cruft

## What

Transform the existing cluttered structure into a clean, organized codebase while preserving all documentation and Docker configurations for reference.

### Success Criteria

- [ ] All essential API files remain in root directory
- [ ] All .md files moved to `docs-archive/` (except PRPs/README.md)
- [ ] All Docker files moved to `docker-historic/`
- [ ] All diagnostic/test scripts removed
- [ ] API starts successfully with remaining files
- [ ] PRPs directory structure preserved intact

## All Needed Context

### Context Completeness Check

_This PRP contains all file paths, categorizations, and preservation requirements needed to reorganize the codebase without breaking functionality._

### Documentation & References

```yaml
- file: /home/ice/whisper-api/main.py
  why: Core API application - imports gpu_validator.py
  pattern: FastAPI application with GPU enforcement
  gotcha: Must preserve gpu_validator.py as direct dependency

- file: /home/ice/whisper-api/start_whisper.sh
  why: Production startup script with GPU validation
  pattern: Service management with PID tracking
  gotcha: References virtual environments that may need updates

- file: /home/ice/whisper-api/requirements.txt
  why: Current dependency list for pip installation
  pattern: Standard pip requirements format
  gotcha: More current than requirements_diarization.txt
```

### Current Codebase tree

```bash
whisper-api/
├── 82 files in root (mix of .py, .sh, .md)
├── docker/ (5 files)
├── examples/ (2 files)
├── fed/ (2 files)
├── PRPs/ (with ai_docs/, templates/, 6 .md files)
├── scripts/ (8 files)
├── tests/ (2 files)
├── volumes/ (empty config/, models/)
└── venv/ (virtual environment)
```

### Desired Codebase tree

```bash
whisper-api/
├── Core API Files (root):
│   ├── main.py                    # Core API
│   ├── gpu_validator.py           # GPU enforcement (required by main.py)
│   ├── requirements.txt           # Dependencies
│   ├── .gitignore                 # Git ignore rules
│   ├── LICENSE                    # License file
│   ├── setup_venv.sh              # Virtual env setup
│   ├── start_whisper.sh           # API startup script
│   └── whisper_client.py          # Python client library (optional but useful)
│
├── PRPs/                          # PRESERVED AS-IS
│   ├── ai_docs/
│   ├── templates/
│   └── *.md files
│
├── docs-archive/                  # NEW - All documentation
│   ├── core-docs/
│   │   ├── README.md
│   │   ├── CLAUDE.md
│   │   ├── QUICKSTART.md
│   │   └── CONTRIBUTING.md
│   ├── troubleshooting/
│   │   ├── BLACKWELL_CUDA_IMPLEMENTATION.md
│   │   ├── CUDA_INITIALIZATION_ISSUE.md
│   │   ├── SYSTEM_ADMIN_CUDA_FIX.md
│   │   └── PyTorch Ada Lovelace GPU Workaround.md
│   └── implementation/
│       ├── DIARIZATION_HARDENING_SUMMARY.md
│       ├── IMPLEMENTATION_SUMMARY.md
│       └── ONBOARDING.md
│
├── docker-historic/               # NEW - All Docker files
│   ├── dockerfiles/
│   │   ├── Dockerfile
│   │   ├── Dockerfile.blackwell
│   │   └── docker/*.Dockerfile.*
│   ├── compose/
│   │   ├── docker-compose.yml
│   │   └── docker-compose.blackwell.yml
│   ├── scripts/
│   │   └── scripts/docker-*.sh
│   └── volumes/
│       └── volumes/* directories
│
├── examples/                      # KEEP AS-IS (useful)
└── venv/                          # Virtual environment
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: main.py imports gpu_validator directly
# Must preserve: from gpu_validator import GPUValidator, GPUEnforcementError

# CRITICAL: HuggingFace token path is hardcoded
# Path: ~/.config/whisper/token

# WARNING: Blackwell GPU (RTX 5060 Ti) has sm_120 architecture
# PyTorch shows warnings but works for basic operations
```

## Implementation Blueprint

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE docs-archive directory structure
  - EXECUTE: mkdir -p docs-archive/{core-docs,troubleshooting,implementation,docker-docs,project-docs,examples-docs}
  - PLACEMENT: Root level directory for all documentation
  - NAMING: Semantic folder names for easy navigation
  
Task 2: MOVE all markdown files to docs-archive
  - FIND: All .md files in root directory (17 files)
  - MOVE: README.md, CLAUDE.md, QUICKSTART.md, CONTRIBUTING.md → docs-archive/core-docs/
  - MOVE: BLACKWELL_CUDA_IMPLEMENTATION.md, CUDA_*.md, PyTorch*.md → docs-archive/troubleshooting/
  - MOVE: DIARIZATION_*.md, IMPLEMENTATION_*.md, ONBOARDING.md → docs-archive/implementation/
  - MOVE: DOCKER_*.md, REBOOT_INSTRUCTIONS.md → docs-archive/docker-docs/
  - MOVE: README_SUGGESTIONS.md, QUICKSTART_ENHANCED.md → docs-archive/project-docs/
  - PRESERVE: PRPs/*.md files in place (DO NOT MOVE)
  
Task 3: CREATE docker-historic directory structure
  - EXECUTE: mkdir -p docker-historic/{dockerfiles,compose,scripts,configs}
  - PLACEMENT: Root level directory for Docker artifacts
  
Task 4: MOVE all Docker-related files
  - MOVE: Dockerfile, Dockerfile.blackwell → docker-historic/dockerfiles/
  - MOVE: docker/* → docker-historic/dockerfiles/
  - MOVE: docker-compose*.yml → docker-historic/compose/
  - MOVE: scripts/docker-*.sh → docker-historic/scripts/
  - MOVE: scripts/whisper-docker.service → docker-historic/configs/
  - MOVE: docker_build.log, test_docker_blackwell.sh → docker-historic/
  - MOVE: volumes/ → docker-historic/volumes/
  - MOVE: .dockerignore → docker-historic/configs/
  
Task 5: REMOVE diagnostic and test Python files
  - DELETE: blackwell_diagnostic.py, cuda_diagnostic.py, cuda_diagnostic_report.json
  - DELETE: test_blackwell_gpu.py, test_cuda_init.py, test_nvidia_ml.py
  - DELETE: test_gpu_whisper.py, test_hybrid_mode.py, test_openai_whisper.py
  - DELETE: test_transcribe.py, test_diarization.py
  - DELETE: scripts/validate_cuda_solution.py
  - DELETE: tests/ directory completely
  
Task 6: REMOVE one-time setup and fix scripts
  - DELETE: fix_blackwell_cuda.sh, fix_cuda_fallback.sh
  - DELETE: setup_gpu.sh, setup_isolated_python.sh, setup.sh
  - DELETE: install_diarization.sh, install_python311.sh, install_service.sh
  - DELETE: init_github.sh
  - DELETE: start_whisper_blackwell.sh (keeping main start_whisper.sh)
  - DELETE: scripts/install-systemd-service.sh
  
Task 7: REMOVE unused support files
  - DELETE: diarization_handler.py (unused, not imported by main.py)
  - DELETE: whisper_config_override.py (only used in tests)
  - DELETE: fed/ directory (personal documentation)
  - DELETE: whisper.log (runtime log)
  - DELETE: 250826_1602.mp3 (sample audio file)
  
Task 8: CLEAN scripts directory
  - DELETE: scripts/ directory after moving Docker scripts
  - NOTE: All useful scripts already in root or moved to docker-historic
  
Task 9: ORGANIZE examples directory
  - KEEP: examples/ as-is (useful client examples)
  - OPTIONALLY MOVE: examples/README.md → docs-archive/examples-docs/
  
Task 10: HANDLE optional files
  - KEEP: whisper-cli.sh in root (useful CLI wrapper)
  - KEEP: requirements_diarization.txt (alternative deps)
  - KEEP OR MOVE: whisper-api.service → docs-archive/configs/ (systemd service)
```

### Implementation Patterns & Key Details

```bash
# Pattern for moving files while preserving structure
mkdir -p docs-archive/core-docs
mv README.md CLAUDE.md QUICKSTART.md CONTRIBUTING.md docs-archive/core-docs/

# Pattern for batch moving with glob
mv BLACKWELL*.md CUDA*.md "PyTorch Ada Lovelace GPU Workaround.md" docs-archive/troubleshooting/

# Pattern for removing test files
rm -f test_*.py
rm -rf tests/

# Critical preservation check
# Ensure these files remain in root:
ls main.py gpu_validator.py requirements.txt start_whisper.sh setup_venv.sh
```

### Integration Points

```yaml
DEPENDENCIES:
  - verify: main.py still imports gpu_validator.py correctly
  - check: start_whisper.sh references correct paths
  
VIRTUAL_ENV:
  - preserve: venv/ directory if exists
  - update: Any hardcoded paths in scripts
  
GIT:
  - preserve: .gitignore in root
  - update: .gitignore to exclude docs-archive/ and docker-historic/ if needed
```

## Validation Loop

### Level 1: File Structure Validation

```bash
# Verify essential files remain
ls main.py gpu_validator.py requirements.txt start_whisper.sh setup_venv.sh
# Expected: All files present

# Verify PRPs preserved
ls -la PRPs/
# Expected: All PRP files and structure intact

# Verify archives created
ls -la docs-archive/ docker-historic/
# Expected: Organized subdirectories with moved files
```

### Level 2: Import Validation

```bash
# Test Python imports work
python -c "import gpu_validator; print('GPU validator imports OK')"
# Expected: No import errors

# Check main.py dependencies
python -c "from gpu_validator import GPUValidator, GPUEnforcementError; print('Imports OK')"
# Expected: Successful import
```

### Level 3: API Startup Validation

```bash
# Test API starts with cleaned structure
source venv/bin/activate || source ~/.venvs/whisper-blackwell/bin/activate
python main.py &
sleep 3
curl http://localhost:8765/health | jq .status
# Expected: "healthy"

# Kill test server
pkill -f "python main.py"
```

### Level 4: Client Functionality

```bash
# Test client library still works
python -c "import whisper_client; print('Client imports OK')"
# Expected: No errors

# Test startup script
./start_whisper.sh start
./start_whisper.sh status
./start_whisper.sh stop
# Expected: All commands work
```

## Final Validation Checklist

### Technical Validation

- [ ] Only ~10 files remain in root directory
- [ ] main.py and gpu_validator.py present and connected
- [ ] API starts successfully with python main.py
- [ ] start_whisper.sh script functional

### Organization Validation

- [ ] All .md files moved to docs-archive/ (except PRPs/)
- [ ] All Docker files moved to docker-historic/
- [ ] All diagnostic/test files removed
- [ ] PRPs directory structure preserved completely

### Code Quality Validation

- [ ] No broken imports in remaining Python files
- [ ] Virtual environment still functional
- [ ] Client library (if kept) still imports correctly
- [ ] No references to deleted files in remaining code

### Documentation & Cleanup

- [ ] docs-archive/ organized by category
- [ ] docker-historic/ preserves Docker knowledge
- [ ] Root directory clean and navigable
- [ ] File count reduced from 82 to ~10-15 files

---

## Anti-Patterns to Avoid

- ❌ Don't delete PRPs directory or its contents
- ❌ Don't move gpu_validator.py (required by main.py)
- ❌ Don't delete .gitignore or LICENSE
- ❌ Don't move files referenced by active scripts
- ❌ Don't delete virtual environment if actively used
- ❌ Don't forget to test API after cleanup