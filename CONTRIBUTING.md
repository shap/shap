# Contributing guidelines

- [Introduction](#introduction)
- [Writing helpful bug reports](#writing-helpful-bug-reports)
- [Installing the latest version](#installing-the-latest-version)
- [Setting up a local development environment](#setting-up-a-local-development-environment)
  - [Fork the repository](#fork-the-repository)
  - [Creating a python environment](#creating-a-python-environment)
  - [Installing from source](#installing-from-source)
  - [Code checks with precommit](#code-checks-with-precommit)
  - [Unit tests with pytest](#unit-tests-with-pytest)
- [Pull Requests (PRs)](#pull-requests-prs)
  - [Etiquette for creating PRs](#etiquette-for-creating-prs)
  - [Checklist for publishing PRs](#checklist-for-publishing-prs)
- [Documentation](#documentation)
  - [Previewing changes on Pull Requests](#previewing-changes-on-pull-requests)
  - [Building the docs locally](#building-the-docs-locally)
- [Jupyter notebook style guide](#jupyter-notebook-style-guide)
  - [General Jupyter guidelines](#general-jupyter-guidelines)
  - [Links / Cross-references](#links--cross-references)
  - [Notebook linting and formatting](#notebook-linting-and-formatting)

## Introduction

Thank you for contributing to SHAP. SHAP is an open source collective effort,
and contributions of all forms are welcome!

You can contribute by:

- Submitting bug reports and features requests on the GitHub [issue
  tracker][issues],
- Contributing fixes and improvements via [Pull Requests][pulls], or
- Discussing ideas and questions in the [Discussions forum][discussions].

If you are looking for a good place to get started, look for issues with the
[good first issue][goodfirstissue] label.

[issues]: https://github.com/shap/shap/issues
[pulls]: https://github.com/shap/shap/pulls
[discussions]: https://github.com/shap/shap/discussions
[goodfirstissue]:
    https://github.com/shap/shap/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22

## Writing helpful bug reports

When submitting bug reports on the [issue tracker][issues], it is very helpful
for the maintainers to include a good **Minimal Reproducible Example** (MRE).

An MRE should be:

- **Minimal**: Use as little code as possible that still produces the same
  problem.
- **Self-contained**: Include everything needed to reproduce your problem,
  including imports and input data.
- **Reproducible**: Test the code you're about to provide to make sure it
  reproduces the problem.

For more information, see [How To Craft Minimal Bug
Reports](https://matthewrocklin.com/minimal-bug-reports).

## Installing the latest version

To get the very latest version of shap, you can pip-install the library directly
from the `master` branch:

```bash
pip install git+https://github.com/shap/shap.git@master
```

This can be useful to test if a particular issue or bug has been fixed since the
most recent release.

Alternatively, if you are considering making changes to the code you can clone
the repository and install your local copy as described below.

## Setting up a local development environment

### Fork the repository

Click [this link](https://github.com/shap/shap/fork) to fork the repository on
GitHub to your user area.

Clone the repository to your local environment, using the URL provided by the
green `<> Code` button on your projects home page.

### Creating a python environment

Create a new isolated environment for the project, e.g. with conda:

```bash
conda create -n shap python=3.11
conda activate shap
```

### Installing from source

Pip-install the project with the `--editable` flag, which ensures that any
changes you make to the source code are immediately reflected in your
environment.

```bash
pip install --editable '.[test,plots,docs]'
```

The various pip extras are defined in [pyproject.toml](pyproject.toml):

- `test-core`: a minimal set of dependencies to run pytest.
- `test`: a wider set of 3rd party packages for the full test suite such as
  tensorflow, pytest, xgboost.
- `plots`: includes matplotlib.
- `docs`: dependencies for building the docs with Sphinx.

Note: When installing from source, shap will attempt to build the C extension
and the CUDA extension. If CUDA is not available, shap will retry the build
without CUDA support.

Consequently, is is quite normal to see warnings such as `WARNING: Could not
compile cuda extensions` when building from source if you do not have CUDA
available.

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

## Pull Requests (PRs)

### Etiquette for creating PRs

Before starting on a PR, please make a proposal by **opening an Issue**,
checking for any duplicates. This isn't necessary for trivial PRs such as fixing
a typo.

**Keep the scope small**. This makes PRs a lot easier to review. Separate
functional code changes (such as bug fixes) from refactoring changes (such as
style improvements). PRs should contain one or the other, but not both.

Open a **Draft PR** as early as possible, do not wait until the feature is
ready. Work on a feature branch with a descriptive name such as
`fix/lightgbm-warnings` or `doc/contributing`.

Use a descriptive title, such as:

- `FIX: Update parameters to remove DeprecationWarning in TreeExplainer`
- `ENH: Add support for python 3.11`
- `DOCS: Fix formatting of ExactExplainer docstring`

### Checklist for publishing PRs

Before marking your PR as "ready for review" (by removing the `Draft` status),
please ensure:

- Your feature branch is up-to-date with the master branch,
- All [pre-commit hooks](#code-checks-with-precommit) pass, and
- Unit tests have been added (if your PR adds any new features or fixes a bug).

## Documentation

The documentation is hosted at
[shap.readthedocs.io](https://shap.readthedocs.io/en/latest/). If you have
modified the docstrings or notebooks, please also check that the changes are
are rendered properly in the generated HTML files.

### Previewing changes on Pull Requests

The documentation is built automatically on each Pull Request, to facilitate
previewing how your changes will render. To see the preview:

1. Look for "All checks have passed", and click "Show all checks".
2. Browse to the check called "docs/readthedocs.org".
3. Click the `Details` hyperlink to open a preview of the docs.

The PR previews are typically hosted on a URL of the form below, replacing
`<pr-number>`:

```
https://shap--<pr-number>.org.readthedocs.build/en/<pr-number>
```

### Building the docs locally

To build the documentation locally:

1. Navigate to the `docs` directory.
2. Run `make html`.
3. Open "\_build/html/index.html" in your browser to inspect the documentation.

Note that `nbsphinx` currently requires the stand-alone program `pandoc`. If you
get an error "Pandoc wasn't found", install `pandoc` as described in
[nbsphinx installation
guide](https://nbsphinx.readthedocs.io/en/0.9.2/installation.html#pandoc).

## Jupyter notebook style guide

If you are contributing changes to the Jupyter notebooks in the documentation, please
adhere to the following style guidelines.

### General Jupyter guidelines

Before committing your notebook(s),

- Ensure that you "Restart Kernel and Run All Cells...", making
  sure that cells are executed in order, the notebook is reproducible and does not have any hidden states.
- Ensure that the notebook does not raise syntax warnings in the Sphinx build logs as a result of your
  changes.

### Links / Cross-references

You are advised to include links in the notebooks as much as possible if it provides the
reader with more background / context on the topic at hand.

Here's an example of how you would accomplish this in a Markdown cell in the notebook:

```markdown
# Force Plot Colors

The [scatter][scatter_doclink] plot create Python matplotlib plots that can be customized at will.

[scatter_doclink]: ../../../generated/shap.plots.scatter.rst#shap.plots.scatter
```

where the link specified is a relative path to the rst file generated by Sphinx.
Prefer relative links over absolute paths.

### Notebook linting and formatting

We use `ruff` and `black-jupyter` to perform code linting and auto-formatting on our notebooks.
Assuming you have set up `pre-commit` as described [above](#code-checks-with-precommit), these checks will
run automatically whenever you commit any changes.
To run the code-quality checks manually, you can do, e.g.:

```bash
pre-commit run --files notebook1.ipynb notebook2.ipynb
```

replacing `notebook1.ipynb` and `notebook2.ipynb` with any notebook(s) you have modified.
