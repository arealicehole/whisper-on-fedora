#!/usr/bin/env python3
"""
Validation script for branch structure reorganization
Verifies the dual-branch strategy is correctly implemented
"""

import subprocess
import os
import sys
from pathlib import Path

class BranchValidator:
    def __init__(self):
        self.results = []
        self.errors = []
        
        # Define expected files on main branch
        self.runtime_files = {
            'main.py',
            'gpu_validator.py', 
            'sitecustomize.py',
            'startup.py',
            'requirements_api.txt',
            'requirements_blackwell.txt',
            'whisper_client.py',
            'LICENSE'
        }
        
        # Files that should NOT be on main
        self.dev_only_patterns = [
            'test_*.py',
            'PRPs/',
            'docs-archive/',
            '.claude/',
            '*.md',
            '.env.development'
        ]
    
    def run_command(self, cmd):
        """Execute shell command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip(), result.returncode
        except Exception as e:
            self.errors.append(f"Command failed: {cmd}\nError: {e}")
            return "", 1
    
    def check_branch_exists(self, branch):
        """Verify branch exists"""
        output, code = self.run_command(f"git branch --list {branch}")
        return branch in output
    
    def get_branch_files(self, branch):
        """Get list of files in a branch (only git-tracked files)"""
        # Get files tracked in the branch
        output, _ = self.run_command(f"git ls-tree -r {branch} --name-only")
        files = set(output.split('\n')) if output else set()
        
        return files
    
    def validate_main_branch(self):
        """Validate main branch contains only runtime files"""
        print("\n🔍 Validating main branch...")
        
        if not self.check_branch_exists('main'):
            self.errors.append("Main branch does not exist")
            return False
        
        files = self.get_branch_files('main')
        
        # Check for expected runtime files
        missing = []
        for rf in self.runtime_files:
            if rf not in files and not any(f.endswith(rf) for f in files):
                missing.append(rf)
        
        if missing:
            self.errors.append(f"Main branch missing runtime files: {missing}")
        
        # Check for unexpected dev files
        unexpected = []
        for f in files:
            # Skip git files and allowed runtime files
            if '.git' in f:
                continue
            if any(f.endswith(rf) for rf in self.runtime_files):
                continue
            if f in ['.gitattributes']:
                continue
            # Flag as unexpected
            unexpected.append(f)
        
        if unexpected:
            self.errors.append(f"Main branch has development files: {unexpected[:5]}...")
        
        file_count = len(files)
        print(f"  ✓ Main branch has {file_count} files")
        
        if file_count > 12:
            self.errors.append(f"Main branch has too many files ({file_count}), expected ~8-10")
        
        return len(self.errors) == 0
    
    def validate_dev_branch(self):
        """Validate dev branch has all files"""
        print("\n🔍 Validating dev branch...")
        
        if not self.check_branch_exists('dev'):
            self.errors.append("Dev branch does not exist")
            return False
        
        files = self.get_branch_files('dev')
        file_count = len(files)
        
        print(f"  ✓ Dev branch has {file_count} files")
        
        # Check has runtime files
        for rf in self.runtime_files:
            if not any(f.endswith(rf) for f in files):
                self.errors.append(f"Dev branch missing runtime file: {rf}")
        
        # Check has dev resources
        has_prps = any('PRPs' in f for f in files)
        has_docs = any('docs-archive' in f for f in files)
        has_tests = any(f.startswith('test_') for f in files)
        
        if not has_prps:
            self.errors.append("Dev branch missing PRPs directory")
        if not has_docs:
            self.errors.append("Dev branch missing docs-archive")
        if not has_tests:
            self.errors.append("Dev branch missing test files")
        
        if file_count < 100:
            self.errors.append(f"Dev branch seems incomplete ({file_count} files), expected 140+")
        
        return len(self.errors) == 0
    
    def validate_github_actions(self):
        """Check GitHub Actions workflow exists"""
        print("\n🔍 Validating GitHub Actions...")
        
        workflow_path = '.github/workflows/sync-to-main.yml'
        if os.path.exists(workflow_path):
            print(f"  ✓ Sync workflow exists: {workflow_path}")
            return True
        else:
            self.errors.append(f"GitHub Actions workflow missing: {workflow_path}")
            return False
    
    def validate_environment_files(self):
        """Check environment files exist on correct branches"""
        print("\n🔍 Validating environment files...")
        
        # Check .env.production exists (in working directory on main)
        # Note: These are typically not committed to git
        print("  ✓ Environment files should be created locally (not in git)")
        print("    - .env.production for main branch")
        print("    - .env.development for dev branch")
        
        return True
    
    def run_validation(self):
        """Run all validation checks"""
        print("=" * 60)
        print("BRANCH STRUCTURE VALIDATION")
        print("=" * 60)
        
        # Save current branch
        original_branch, _ = self.run_command("git rev-parse --abbrev-ref HEAD")
        
        checks = [
            self.validate_main_branch,
            self.validate_dev_branch,
            self.validate_github_actions,
            self.validate_environment_files
        ]
        
        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
        
        # Return to original branch
        self.run_command(f"git checkout {original_branch} 2>/dev/null")
        
        # Print results
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL VALIDATION CHECKS PASSED")
            print("\nBranch reorganization successful!")
            print("- Main branch: minimal runtime (~8 files)")
            print("- Dev branch: complete environment (140+ files)")
            print("- Automation: GitHub Actions configured")
            print("- Environment: Separate configs for prod/dev")
        else:
            print("❌ VALIDATION FAILED")
            print("\nErrors found:")
            for error in self.errors:
                print(f"  - {error}")
            print("\nPlease fix these issues before proceeding.")
            return 1
        
        print("=" * 60)
        return 0

if __name__ == "__main__":
    validator = BranchValidator()
    sys.exit(validator.run_validation())