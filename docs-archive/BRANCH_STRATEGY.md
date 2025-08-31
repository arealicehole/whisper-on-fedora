# Branch Strategy for Whisper API

## Overview
This repository uses a dual-branch strategy to separate production runtime from development resources.

## Branch Structure

### main Branch
- **Purpose**: Production-ready, minimal runtime
- **Contents**: Only files required to run the Whisper API in production
- **File Count**: ~7-9 essential files
- **Key Files**:
  - `main.py` - API server entry point
  - `gpu_validator.py` - GPU capability validation
  - `sitecustomize.py` - PyTorch patches for Blackwell
  - `startup.py` - Initialization script
  - `requirements_api.txt` - Core dependencies
  - `requirements_blackwell.txt` - Blackwell-specific dependencies
  - `whisper_client.py` - Client library
  - `LICENSE` - License file
  - `.env.production` - Production settings

### dev Branch  
- **Purpose**: Complete development environment
- **Contents**: All project files including development tools, tests, documentation
- **File Count**: 140+ files
- **Additional Resources**:
  - `/PRPs` - Project requirement plans (40+ files)
  - `/docs-archive` - Documentation (38 files)
  - `/.claude` - Claude assistant commands
  - `/test_*.py` - Test scripts (8 files)
  - Development tools and configurations

## Workflow Guidelines

### For Production Deployment
```bash
git checkout main
python main.py
```

### For Development
```bash
git checkout dev
# Full access to all development resources
```

### Syncing Runtime Changes
Runtime file changes in `dev` automatically sync to `main` via GitHub Actions:
1. Make changes to runtime files in `dev` branch
2. Commit and push to `dev`
3. GitHub Actions automatically syncs to `main`

### Manual Sync (if needed)
```bash
git checkout main
git checkout dev -- main.py gpu_validator.py  # etc for runtime files
git commit -m "sync: runtime updates from dev"
git push
```

## Merge Strategy
The `.gitattributes` file prevents merge conflicts:
- Development files use `merge=ours` (keep branch-specific versions)
- Runtime files use `merge=recursive` (allow merging)

## Environment Variables
- **main branch**: Uses `.env.production` (production settings)
- **dev branch**: Uses `.env.development` (development settings)

## Best Practices
1. **Never add development files to main** - Keep it minimal
2. **Always develop in dev branch** - All new features start here
3. **Let automation handle syncing** - GitHub Actions manages runtime file sync
4. **Test on main before production** - Verify minimal runtime works

## Validation
Run validation checks:
```bash
# On main branch - should have ~8 files
ls -la | grep -v "^d" | wc -l

# On dev branch - should have 140+ files  
find . -type f -not -path "./.git/*" | wc -l
```

## Rollback Plan
If issues arise:
1. `backup-before-reorganization` branch contains original state
2. Revert to backup: `git checkout backup-before-reorganization`
3. Force push if needed: `git push --force origin main`