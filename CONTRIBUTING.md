# Contributing to DICE Embeddings

Thank you for your interest in contributing to DICE Embeddings! This guide outlines our development workflow and best practices.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/dice-group/dice-embeddings.git
cd dice-embeddings

# Install in development mode with all dependencies
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu

# For GPU support, use:
pip install -e '.[dev]'
```

### 2. Download Test Datasets

```bash
wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate
unzip KGs.zip
```

## Code Quality Workflow

### Before Every Commit

**Always run these checks before committing:**

```bash
# 1. Run ruff linter (required)
ruff check dicee/ --line-length=200

# 2. Run type checker (recommended)
mypy dicee/ --config-file=pyproject.toml

# 3. Run tests
python -m pytest -p no:warnings -x
```

### Continuous Integration

Our CI pipeline automatically runs:
- ✅ Ruff linting (blocking)
- ✅ Type checking with mypy (non-blocking, currently)
- ✅ Full test suite with coverage

**Pull requests must pass ruff and pytest to be merged.**

## Coding Standards

### Type Hints

We are progressively adding type hints to the codebase. For new code:

- ✅ **DO** add type hints to all public functions
- ✅ **DO** use `Optional[T]` for optional parameters
- ✅ **DO** specify return types
- ⚠️  **CONSIDER** adding type hints to internal functions

Example:

```python
from typing import Optional, List, Tuple, Union

def predict_topk(
    self,
    *,
    h: Optional[Union[str, List[str]]] = None,
    r: Optional[Union[str, List[str]]] = None,
    t: Optional[Union[str, List[str]]] = None,
    topk: int = 10
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """
    Predict top-k missing items in a triple pattern.
    
    Args:
        h: Head entity/entities. None to predict heads.
        r: Relation/relations. None to predict relations.
        t: Tail entity/entities. None to predict tails.
        topk: Number of top predictions to return.
    
    Returns:
        For single query: List[(item, score), ...]
        For batch query: List of such lists
    """
    ...
```

### Error Messages

Provide actionable error messages with:

1. **Clear problem description**
2. **Suggested solutions**
3. **Example commands**
4. **Link to documentation**

Example:

```python
raise ValueError(
    f"Dataset directory not found: {path}\\n"
    f"\\nSuggestions:\\n"
    f"  1. Download datasets:\\n"
    f"     wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip\\n"
    f"  2. Use absolute path: --dataset_dir /absolute/path/to/KGs/UMLS\\n"
    f"\\nSee docs/guides/troubleshooting.md for more solutions\\n"
)
```

### Docstrings

Use **Google-style docstrings** with type information:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """
    One-line summary of function purpose.
    
    Longer description if needed, with examples and context.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    
    Examples:
        >>> my_function("test", 5)
        True
        
        >>> my_function("example")
        False
    
    See Also:
        - related_function(): Related functionality
        - docs/guide.md: Documentation reference
    """
    ...
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names

Example test:

```python
def test_predict_topk_single_query():
    """Test predict_topk with a single (h, r, ?) query."""
    from dicee import KGE
    
    model = KGE(path="path/to/trained/model")
    results = model.predict_topk(h="Mongolia", r="isLocatedIn", topk=3)
    
    assert len(results) == 3
    assert all(isinstance(item, tuple) for item in results)
    assert all(len(item) == 2 for item in results)
```

### Running Tests

```bash
# Run all tests
python -m pytest -p no:warnings -x

# Run specific test file
python -m pytest tests/test_predict_kge.py -p no:warnings

# Run with coverage
coverage run -m pytest -p no:warnings -x
coverage report -m
```

### Test Documentation

**Tests serve as living documentation!** When adding examples to the README:

1. ✅ **DO** link to test files instead of creating standalone examples
2. ✅ **DO** keep tests up-to-date and CI-verified
3. ⚠️  **DON'T** create example scripts that can become stale

## Documentation

### Where to Document

| Content Type | Location |
|--------------|----------|
| API Reference | Docstrings (auto-generated to docs/) |
| User Guides | `docs/guides/*.md` |
| Troubleshooting | `docs/guides/troubleshooting.md` |
| Multi-hop Queries | `docs/guides/multi_hop_queries.md` |
| Examples | Test files in `tests/test_*.py` |
| Dataset Formats | `docs/guides/datasets.md` |

### Updating Documentation

When adding new features:

1. ✅ Add comprehensive docstrings with examples
2. ✅ Create or update relevant guide in `docs/guides/`
3. ✅ Add test cases demonstrating usage
4. ✅ Update README.md with link to test file
5. ✅ Update CHANGELOG if applicable

## Pull Request Workflow

### 1. Create Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code with type hints
- Add comprehensive tests
- Update documentation
- Write clear commit messages

### 3. Pre-commit Checks

```bash
# Run all checks
ruff check dicee/ --line-length=200
mypy dicee/ --config-file=pyproject.toml
python -m pytest -p no:warnings -x
```

### 4. Commit and Push

```bash
git add -A
git commit -m "feat: add new feature X

- Detailed description of changes
- Added type hints to all new functions
- Added tests in tests/test_feature_x.py
- Updated docs/guides/feature_guide.md"

git push origin feature/my-new-feature
```

### 5. Create Pull Request

- Target branch: `develop` (not `main`)
- Clear description of changes
- Link to related issues
- Ensure CI passes

## Commit Message Convention

Use conventional commits format:

```
<type>: <subject>

<body>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks
- `perf:` Performance improvements

**Example:**

```
feat: add multi-hop query support for union operations

- Implemented 2u (two-way union) query pattern
- Added up (union + projection) query pattern
- Added comprehensive type hints to answer_multi_hop_query()
- Added tests in tests/test_answer_multi_hop_query.py
- Updated docs/guides/multi_hop_queries.md with examples

Closes #123
```

## Common Tasks

### Adding a New Model

See `.github/skills/add-model/SKILL.md` for the complete workflow.

### Adding a New Trainer

1. Extend `AbstractTrainer` in `dicee/abstracts.py`
2. Implement required methods: `fit()`, `configure_callbacks()`
3. Register in `dicee/trainer/__init__.py`
4. Add tests in `tests/test_trainers.py`

### Debugging Issues

1. Check `docs/guides/troubleshooting.md`
2. Run with verbose logging: `--verbose 1`
3. Enable debug mode if available
4. Add minimal reproduction in test file

## Getting Help

- 📖 **Documentation:** https://dice-embeddings.readthedocs.io/
- 💬 **Issues:** https://github.com/dice-group/dice-embeddings/issues
- 🧪 **Examples:** See `tests/test_*.py` files
- 📋 **Guides:** See `docs/guides/` directory

---

## Summary Checklist

Before submitting a pull request:

- [ ] Code follows line length limit (200 characters)
- [ ] Ruff linting passes: `ruff check dicee/ --line-length=200`
- [ ] Type hints added to new functions
- [ ] Tests added and passing: `pytest -p no:warnings -x`
- [ ] Docstrings written with examples
- [ ] Documentation updated (guides, README)
- [ ] Commit messages follow convention
- [ ] PR targets `develop` branch
- [ ] CI pipeline passes

Thank you for contributing! 🎉
