# Type Hints Roadmap for dicee

This document outlines the strategy for gradually enforcing type hints across the dicee codebase using mypy.

## Motivation

Type hints improve:
- Code readability and IDE support
- Detection of logic errors before runtime
- API documentation clarity
- Maintenance and refactoring safety

## Current Status

**mypy Results:**
- **CI error budget**: 1367 type issues, recorded in `.github/mypy-baseline.txt`. Run `mypy dicee/ --config-file=pyproject.toml` to measure the current count.
- **CI status**: The mypy step blocks the build when its error count exceeds that budget. Existing errors are printed in the log even when the step passes; lint and runtime tests are separate checks.
- **Coverage**: partial; growing gradually per the tiers below

## Priority Tiers

### Tier 1: Public API & Core Inference (HIGH PRIORITY)
Critical paths that users interact with directly. Should have complete type hints.

- `dicee/knowledge_graph_embeddings.py` (1423 lines, main inference API)
  - Status: ⏳ Partial (public methods typed, internal utilities need work)
  - Goal: 100% before next minor release
  - Impact: Highest - users rely on this for link prediction, embeddings

- `dicee/config.py`
  - Status: 🔴 Issues with Optional types (fields assigned None but not Optional[T])
  - Goal: Complete by next release
  - Fix: Use `Optional[Type]` for nullable fields

- `dicee/executer.py`
  - Status: ⏳ Minimal typing
  - Goal: Function signatures typed by next patch
  - Impact: Training entry point

### Tier 2: Core Utilities & Trainers (MEDIUM PRIORITY)
Internal functions that support public APIs. Type hints reduce integration bugs.

- `dicee/trainer/` (all trainers)
  - Status: ⏳ Partial
  - Target: 80% within 2 releases

- `dicee/models/base_model.py`
  - Status: ⏳ Partial (abstract methods need types)
  - Target: Next patch

- `dicee/read_preprocess_save_load_kg/`
  - Status: 🔴 Minimal typing
  - Target: Next minor

### Tier 3: Scripts & Experimental (LOW PRIORITY)
Ad-hoc utilities and optional features. Type hints optional.

- `dicee/scripts/`
  - Status: 🔴 Minimal typing
  - Target: Nice-to-have, low impact

- Analysis scripts, helpers
  - Status: 🔴 Minimal typing
  - Target: When convenient

## Implementation Strategy

### Phase 1: Foundation (This PR)
- [x] Enforce a mypy error-count budget in CI while existing typing issues are fixed
- [x] Create this roadmap
- [x] Fix config.py Optional types (quick win)
- [ ] Add py.typed marker to package

### Phase 2: Priority Modules (Next PR)
- [ ] Complete type hints for `knowledge_graph_embeddings.py`
- [ ] Fix `config.py` Optional issues
- [ ] Add types to `executer.py` public interface

### Phase 3: Utilities (Following Releases)
- [ ] Type trainers and models
- [ ] Update preprocessing pipeline

### Phase 4: Coverage (Long-term)
- [ ] Gradual coverage of scripts and helpers
- [ ] Consider `disallow_untyped_defs = true` for key modules

## mypy Configuration

### Current Approach: Per-Module Strictness
Instead of global `disallow_untyped_defs = true/false`, we use mypy's `per_module_options` to enforce different levels for different parts of the code.

```toml
# pyproject.toml [tool.mypy]

# Tier 1: Strict enforcement
[[tool.mypy.overrides]]
module = [
  "dicee.knowledge_graph_embeddings",
  "dicee.config",
  "dicee.executer",
]
disallow_untyped_defs = true        # Functions must be typed

# Tier 2: Relaxed for now (warnings only)
[[tool.mypy.overrides]]
module = [
  "dicee.trainer.*",
  "dicee.models.*",
]
disallow_untyped_defs = false       # Warnings but not required
warn_return_any = true

# Tier 3: Scripts (pass for now)
[[tool.mypy.overrides]]
module = "dicee.scripts.*"
ignore_errors = true               # Revisit later
```

## File Tracker

Run this to see current state:
```bash
mypy dicee/ --config-file=pyproject.toml | grep "error:" | sort | uniq -c | sort -rn
```

## Contributing

When adding new code:
1. Always include type hints on public function signatures
2. Use `Optional[T]` for nullable parameters/returns
3. Use `Union`, `Protocol`, and `Generic` when appropriate
4. Run `mypy dicee/ --config-file=pyproject.toml` before committing

## See Also

- [PEP 484: Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Python typing best practices](https://peps.python.org/pep-0586/)
