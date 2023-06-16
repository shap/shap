# Contributing guidelines

- [Introduction](#introduction)
- [Pull Requests (PRs)](#pull-requests-prs)
  - [Etiquette for creating PRs](#etiquette-for-creating-prs)
  - [Checklist for publishing PRs](#checklist-for-publishing-prs)
- [Setting up a local development
  environment](#setting-up-a-local-development-environment)
  - [Code checks with pre-commit](#code-checks-with-pre-commit)
  - [Unit tests with pytest](#unit-tests-with-pytest)

## Introduction

Thank you for contributing to SHAP. SHAP is an open source collective effort,
and contributions of all forms are welcome!

You can contribute by:

- Submitting bug reports and features requests on the GitHub issue tracker
- Contributing fixes and improvements via Pull Requests

## Pull Requests (PRs)

### Etiquette for creating PRs

- Before starting on a PR, please make a proposal by opening an Issue, and check
  for any duplicates
- Work on a feature branch with a descriptive name
- Open a Draft PR as early as possible, do not wait until the feature is ready

### Checklist for publishing PRs

Before marking your PR as "ready for review" (by removing the `Draft` status),
please ensure:

- All [pre-commit hooks](https://pre-commit.com/#install) pass
- If your PR adds any new features or fixes a bug, ensure unit tests have been
  added
- Your feature branch is up-to-date with the master branch

## Setting up a local development environment

1. Fork the repository on Github to your user area
2. Clone the repository to your local environment
3. Create a new environment, e.g. with conda:

   ```bash
   conda create -n shap python=3.11
   conda activte shap
   ```

4. Install the project and dependencies, including the `test` extras:

   ```bash
   pip install --editable .[test,plots]
   ```

### Code checks with pre-commit

We use [pre-commit hooks](https://pre-commit.com/#install) to run code checks.
Enable pre-commit in your local environment with:

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
pre-commit hook. You can also run ruff locally with:

```bash
ruff check .
```

### Unit tests with pytest

The unit test suite can be run locally with:

```bash
pytest
```
