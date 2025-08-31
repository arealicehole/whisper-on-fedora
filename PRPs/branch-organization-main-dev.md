# PRP: Git Branch Organization - Main (Runtime) vs Dev (Full)

## Goal

**Feature Goal**: Reorganize the whisper-api repository into two distinct branches - `main` containing only production runtime essentials and `dev` containing all files including development resources.

**Deliverable**: A properly organized git repository with:
- `main` branch: Minimal production deployment files only
- `dev` branch: Complete codebase including all development resources
- Automated workflow to sync production builds from dev to main

**Success Definition**: 
- Main branch contains exactly 7 essential runtime files
- Dev branch maintains all 140+ project files
- Clear git workflow prevents merge conflicts
- Production deployments use main branch exclusively

## Context

```yaml
project_overview:
  name: whisper-api
  type: FastAPI Speech-to-Text Service
  runtime_files: 7 essential files
  total_files: 140+ files across development
  key_constraint: GPU-only enforcement (no CPU fallback)

technical_requirements:
  runtime_essentials:
    - main.py (FastAPI application)
    - gpu_validator.py (GPU enforcement)
    - sitecustomize.py (Blackwell compatibility)
    - requirements_api.txt (runtime dependencies)
    - requirements_blackwell.txt (Blackwell GPU deps)
    - LICENSE (legal requirement)
    - whisper_client.py (client library)
  
  development_categories:
    documentation: 30+ markdown files
    testing: 8 test scripts
    claude_config: 40+ command files
    prps: 15+ prompt resources
    setup_scripts: 5 bash scripts
    archives: 10+ historical files

git_strategy:
  approach: selective_merge_with_automation
  not_using: branch-per-environment (anti-pattern)
  key_tools:
    - git sparse-checkout
    - .gitattributes for merge control
    - GitHub Actions for automation
```

## Implementation Tasks

### Task 1: Backup Current State
**Objective**: Create safety backup before reorganization
```bash
# Create backup branch
git checkout -b backup-$(date +%Y%m%d-%H%M%S)
git push origin backup-$(date +%Y%m%d-%H%M%S)

# Create local backup
tar -czf whisper-api-backup-$(date +%Y%m%d).tar.gz .
```

### Task 2: Create Dev Branch with Everything
**Objective**: Establish dev branch as primary development branch
```bash
# From current main, create dev with all files
git checkout main
git checkout -b dev
git add -A
git commit -m "feat: establish dev branch with complete codebase"
git push origin dev
```

### Task 3: Setup Branch-Specific Git Attributes
**Objective**: Configure merge strategies to prevent conflicts

Create `.gitattributes` in both branches:
```gitattributes
# Development files - keep branch-specific during merges
PRPs/** merge=ours
docs-archive/** merge=ours
test_*.py merge=ours
.claude/** merge=ours
*.md merge=ours
setup_*.sh merge=ours

# Runtime files - allow merging
main.py merge=recursive
gpu_validator.py merge=recursive
requirements_api.txt merge=recursive
LICENSE merge=recursive
```

### Task 4: Clean Main Branch to Runtime Only
**Objective**: Remove all non-essential files from main branch
```bash
git checkout main

# Remove all development files
git rm -r PRPs/ docs-archive/ .claude/ docker-historic/ examples/
git rm -r .serena/ volumes/ logs/ audio_files/
git rm test_*.py setup_*.sh
git rm *.md  # except LICENSE
git rm requirements_diarization.txt requirements.txt requirements_whisperx.txt
git rm fix_pyannote_blackwell.py torchvision_fix.py pyannote_fix.py
git rm patch_pyannote.py main_pyannote.py whisperx_diarization.py
git rm validate_blackwell.py whisper-cli.sh start_whisper.sh

# Keep only essentials
# main.py, gpu_validator.py, sitecustomize.py, startup.py
# requirements_api.txt, requirements_blackwell.txt
# LICENSE, whisper_client.py

git commit -m "refactor: main branch with runtime essentials only"
git push origin main --force-with-lease
```

### Task 5: Create Selective Merge Workflow
**Objective**: Establish process for syncing runtime updates

Create `.github/workflows/sync-to-main.yml` on dev branch:
```yaml
name: Sync Runtime Files to Main

on:
  push:
    branches: [dev]
    paths:
      - 'main.py'
      - 'gpu_validator.py'
      - 'requirements_api.txt'
      - 'requirements_blackwell.txt'
      - 'whisper_client.py'

jobs:
  sync-runtime:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          
      - name: Configure Git
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          
      - name: Sync Runtime Files
        run: |
          git checkout main
          git checkout dev -- main.py gpu_validator.py requirements_api.txt
          git checkout dev -- requirements_blackwell.txt whisper_client.py
          
          if git diff --staged --quiet; then
            echo "No runtime changes to sync"
          else
            git commit -m "sync: runtime files from dev branch"
            git push origin main
          fi
```

### Task 6: Setup Development Environment Variables
**Objective**: Configure environment-specific settings

On dev branch, create `.env.development`:
```env
# Development settings
DEBUG=true
RELOAD=true
LOG_LEVEL=DEBUG
TEST_MODE=true
```

On main branch, create `.env.production`:
```env
# Production settings
DEBUG=false
RELOAD=false
LOG_LEVEL=INFO
TEST_MODE=false
```

### Task 7: Document Branch Strategy
**Objective**: Create clear documentation for team

Create `BRANCH_STRATEGY.md` on dev branch:
```markdown
# Branch Organization Strategy

## Branch Structure
- **main**: Production runtime files only (7 files)
- **dev**: Complete development environment (140+ files)

## Workflow
1. All development happens on `dev` branch
2. Runtime file changes auto-sync to `main` via GitHub Actions
3. Production deployments use `main` branch exclusively

## Commands
- Switch to dev: `git checkout dev`
- Update runtime on main: Automatic via push to dev
- Manual sync: `git checkout main && git checkout dev -- main.py`
```

### Task 8: Create Sparse Checkout Configuration
**Objective**: Enable efficient cloning of main branch

Setup sparse checkout for production deployments:
```bash
# On main branch
git sparse-checkout init --cone
git sparse-checkout set . --no-cone
cat > .git/info/sparse-checkout <<EOF
main.py
gpu_validator.py
sitecustomize.py
startup.py
requirements_api.txt
requirements_blackwell.txt
LICENSE
whisper_client.py
EOF
git read-tree -m -u HEAD
```

### Task 9: Setup Pre-commit Hooks
**Objective**: Prevent accidental commits of dev files to main

Create `.git/hooks/pre-commit` on main branch:
```bash
#!/bin/bash
# Prevent development files from being committed to main

FORBIDDEN_FILES=$(git diff --cached --name-only | grep -E '(test_|\.md$|^PRPs/|^docs-archive/|^\.claude/)')

if [ ! -z "$FORBIDDEN_FILES" ]; then
    echo "Error: Attempting to commit development files to main branch:"
    echo "$FORBIDDEN_FILES"
    echo "These files belong on the dev branch only."
    exit 1
fi
```

### Task 10: Validate Final Structure
**Objective**: Ensure branches are correctly organized

Validation script:
```bash
#!/bin/bash
# validate_branches.sh

echo "Validating main branch..."
git checkout main
FILE_COUNT=$(find . -type f -not -path "./.git/*" | wc -l)
if [ $FILE_COUNT -gt 10 ]; then
    echo "ERROR: Main branch has $FILE_COUNT files (expected ≤10)"
    exit 1
fi

echo "Validating dev branch..."
git checkout dev
if [ ! -d "PRPs" ] || [ ! -d ".claude" ]; then
    echo "ERROR: Dev branch missing development directories"
    exit 1
fi

echo "✅ Branch structure validated successfully"
```

## Validation Gates

### Gate 1: File Count Verification
```bash
# Main branch should have exactly these files:
git checkout main && ls -la
# Expected: main.py, gpu_validator.py, sitecustomize.py, startup.py
#          requirements_api.txt, requirements_blackwell.txt
#          LICENSE, whisper_client.py, .gitignore
```

### Gate 2: Runtime Test
```bash
# Test that main branch can run standalone
git checkout main
python -m venv test_env
source test_env/bin/activate
pip install -r requirements_api.txt
python main.py  # Should start on port 8767
```

### Gate 3: Merge Test
```bash
# Test selective merge workflow
git checkout dev
echo "# test comment" >> main.py
git commit -am "test: runtime update"
git push origin dev
# Wait for GitHub Action
git checkout main && git pull
# Verify main.py has the update
```

### Gate 4: Deployment Test
```bash
# Clone only main branch for production
git clone --single-branch --branch main https://github.com/user/whisper-api.git production-test
cd production-test
ls -la  # Should show only runtime files
```

## Final Validation Checklist

- [ ] Main branch contains exactly 7-9 files (runtime only)
- [ ] Dev branch contains all 140+ project files
- [ ] .gitattributes configured on both branches
- [ ] GitHub Actions workflow created and tested
- [ ] Sparse checkout configuration working
- [ ] Pre-commit hooks preventing dev files on main
- [ ] Documentation updated (BRANCH_STRATEGY.md)
- [ ] Production deployment tested from main branch
- [ ] Development environment functional on dev branch
- [ ] Selective merge workflow validated

## Rollback Plan

If issues arise, rollback procedure:
```bash
# Restore from backup branch
git checkout main
git reset --hard backup-20240831-120000
git push origin main --force-with-lease

git checkout dev  
git reset --hard backup-20240831-120000
git push origin dev --force-with-lease
```

## Confidence Score: 9/10

This PRP provides comprehensive guidance for reorganizing the whisper-api repository into distinct runtime (main) and development (dev) branches. The approach uses proven git strategies, automation via GitHub Actions, and includes multiple validation gates to ensure successful implementation.