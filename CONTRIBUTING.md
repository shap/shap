# Contributing guidelines

- [Introduction](#introduction)
- [Installing from the master branch](#installing-from-the-master-branch)
- [Setting up a local development environment](#setting-up-a-local-development-environment)
  - [Code checks with precommit](#code-checks-with-precommit)
  - [Unit tests with pytest](#unit-tests-with-pytest)
  - [Documentation](#documentation)
- [Pull Requests (PRs)](#pull-requests-prs)
  - [Etiquette for creating PRs](#etiquette-for-creating-prs)
  - [Checklist for publishing PRs](#checklist-for-publishing-prs)

## Introduction

Thank you for contributing to SHAP. SHAP is an open source collective effort,
and contributions of all forms are welcome!

You can contribute by:

- Submitting bug reports and features requests on the GitHub
  [issue tracker](https://github.com/slundberg/shap/issues), or
- Contributing fixes and improvements via
  [Pull Requests](https://github.com/slundberg/shap/pulls).

## Installing from the master branch

To get the very latest version of shap, you can pip-install the library directly
from the `master` branch:

```bash
pip install git+https://github.com/slundberg/shap.git@master
```

This can be useful to test if a particular issue or bug has been fixed since the
most recent release.

## Setting up a local development environment

To set up a local development environment

1. Fork the repository on Github to your user area.
2. Clone the repository to your local environment:

   ```bash
   # Clone with HTTPS
   git clone https://github.com/slundberg/shap.git

   # Or, clone with SSH
   git clone git@github.com:slundberg/shap.git
   ```

3. Create a new environment, e.g. with conda:

   ```bash
   conda create -n shap python=3.11
   conda activate shap
   ```

4. Install the project and dependencies, including the `test` extras:

   ```bash
   pip install --editable '.[test,plots]'
   ```

### Code checks with precommit

We use [pre-commit hooks](https://pre-commit.com/#install) to run code checks.
Enable `pre-commit` in your local environment with:

```bash
pip install pre-commit
pre-commit install
```

To run the checks on all files, use:

```bash
pre-commit install
pre-commit run --all-files
```

[Ruff](https://beta.ruff.rs/docs/) is used as a linter, and it is enabled as a
pre-commit hook. You can also run `ruff` locally with:

```bash
pip install ruff
ruff check .
```

### Unit tests with pytest

The unit test suite can be run locally with:

```bash
pytest
```

### Documentation

The documentation is built on CI, and is hosted by readthedocs. If you have
modified the docstrings or notebooks, please also check that the changes are
rendered properly in the generated HTML files.

To build the documentation locally:

1. Navigate to the `docs` directory
2. Run `make html`
3. Open "_build/html/index.html" in your browser

## Pull Requests (PRs)

### Etiquette for creating PRs

- Before starting on a PR, please make a proposal by opening an Issue, and check
  for any duplicates. This isn't necessary for trivial PRs such as fixing a
  typo.
- Work on a feature branch with a descriptive name such as
  `fix/lightgbm-warnings` or `doc/contributing`.
- Open a Draft PR as early as possible, do not wait until the feature is ready.
- Separate functional code changes (such as bug fixes) from refactoring changes.
  PRs should contain one or the other, but not both.

### Checklist for publishing PRs

Before marking your PR as "ready for review" (by removing the `Draft` status),
please ensure:

- Your feature branch is up-to-date with the master branch,
- All [pre-commit hooks](https://pre-commit.com/#install) pass, and
- Unit tests have been added (if your PR adds any new features or fixes a bug).
