# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development
- Install with CUDA: `python install.py`
- Install (Meson): `pip install --editable .`
- Install (Legacy): `pip install --editable '.[test,plots,docs]'`
- JS components: `npm install && npm run build`

## Testing & Validation
- Run all tests: `pytest`
- Single test: `pytest tests/path/to/specific_test.py::TestClass::test_method`
- Type check: `mypy shap tests`
- Lint: `ruff check .`
- Format: `ruff format .`

## Code Style Guidelines
- Docstrings: NumPy convention
- Line length: 120 characters
- Imports order: Standard library → Third-party → Local modules
- Type hints: Used but not required for all files
- Organized by modules: explainers, maskers, plots, utils, models
- Error handling: Use shap.utils._exceptions for custom errors

## Documentation
- Build docs: `cd docs && make html`