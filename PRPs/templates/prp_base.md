# PRP: [Feature Name]

## Goal
**Feature Goal**: [Specific, measurable objective]
**Deliverable**: [Concrete outcome/artifact]
**Success Definition**: [Clear criteria for completion]

## Context
```yaml
references:
  - documentation:
      - url: [URL with #section-anchor]
        purpose: [Why this reference is critical]
  - examples:
      - url: [GitHub/StackOverflow URL]
        patterns: [Specific patterns to follow]
  - internal:
      - file: [/path/to/file.ext:line-range]
        convention: [Convention to maintain]

gotchas:
  - issue: [Known problem]
    solution: [How to handle it]
  - issue: [Compatibility concern]
    solution: [Workaround or approach]

dependencies:
  - package: [name==version]
    reason: [Why this specific version]
  - system: [requirement]
    check: [Command to verify]

patterns:
  - naming: [Convention from codebase]
  - structure: [File organization pattern]
  - error_handling: [Approach to follow]
```

## Implementation Tasks
```markdown
### Task 1: [Setup/Foundation]
- Location: [Exact file/directory placement]
- Pattern: Follow [/path/to/example.py:10-50]
- Dependencies: Requires [Task X] completion
- Validation: Run `[command]` to verify

### Task 2: [Core Implementation]
- Key Classes/Functions: [Specific names following convention]
- Integration Points: [Where this connects to existing code]
- Configuration: [Environment variables or settings]
- Testing: [How to test this component]

### Task 3: [Integration]
- API Changes: [Endpoints or interfaces affected]
- Data Flow: [Input -> Processing -> Output]
- Error Cases: [What to handle and how]
- Performance: [Optimization considerations]
```

## Validation Gates
```bash
# Component Test
[specific test command that should pass]

# Integration Test
[command to verify integration]

# Performance Check
[command or metric to validate]

# System Validation
[end-to-end test command]
```

## Final Validation Checklist
- [ ] All existing tests pass: `[test command]`
- [ ] New functionality works: [How to verify]
- [ ] Performance acceptable: [Metric and threshold]
- [ ] Documentation updated: [Where to check]
- [ ] No regressions: [What to verify]
- [ ] Production ready: [Final checks]

## Implementation Notes
[Any additional context that helps with implementation but doesn't fit above]