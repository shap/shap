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
  - [AI Usage Policy](#ai-usage-policy)
- [Documentation](#documentation)
  - [Previewing changes on Pull Requests](#previewing-changes-on-pull-requests)
  - [Building the docs locally](#building-the-docs-locally)
- [Jupyter notebook style guide](#jupyter-notebook-style-guide)
  - [General Jupyter guidelines](#general-jupyter-guidelines)
  - [Links / Cross-references](#links--cross-references)
  - [Notebook linting and formatting](#notebook-linting-and-formatting)
- [Maintainer guide](#maintainer-guide)
  - [Issue triage](#issue-triage)
  - [PR triage](#pr-triage)
  - [Versioning](#versioning)
  - [Minimum supported dependencies](#minimum-supported-dependencies)
  - [Making releases](#making-releases)
  - [Release notes from PR labels](#release-notes-from-pr-labels)

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

To build from source, you need a compiler to build the C extension.

- On linux, you can install gcc with:

  ```bash
  sudo apt install build-essential
  ```

- Or on Windows, one way of getting a compiler is to [install
  mingw64](https://www.mingw-w64.org/downloads/).

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

For info about matplotlib tests, see `tests/plots/__init__.py`.

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

### AI Usage Policy

This repository accepts PRs written by LLMs in any capacity. When filing a PR, please keep the following in mind:
 - Disclose your AI usage. A disclosure skill is available at [.claude/skills/ai-disclosure](.claude/skills/ai-disclosure) — not mandatory, but a good guideline for what we expect. Explain what the agent did, where it acted autonomously, where it assisted, and where it advised.
 - Make sure you understand each and every line in your PR as if you've written it yourself. Explain what you understood during working on the issue in the PR description and the rationale behind your changes.
 - This repository is maintained by humans, so write your PR documentation in a clean, minimal and human-oriented way without AI bloat.
 - Most of the work on a feature happens after the merge, so maintainers need to understand the code thoroughly. As someone filing the PR it is your responsibility to help them so maintainers can confidently own the code.
 - LLMs can reproduce copyrighted code. As a contributor, you are responsible for ensuring your submission does not violate copyright.
 - PRs can be closed if any of these points are not met.
 - This policy is experimental and can change any time.

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

```text
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

The documentation dependencies are pinned in `docs/requirements-docs.txt`. These can be
updated by running the `uv` command specified in the top of that file, optionally with
the `--upgrade` flag.

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

We use `ruff` to perform code linting and auto-formatting on our notebooks.
Assuming you have set up `pre-commit` as described
[above](#code-checks-with-precommit), these checks will run automatically
whenever you commit any changes.

To run the code-quality checks manually, you can do, e.g.:

```bash
pre-commit run --files notebook1.ipynb notebook2.ipynb
```

replacing `notebook1.ipynb` and `notebook2.ipynb` with any notebook(s) you have modified.

## Maintainer guide

### Issue triage

Bug reports and feature requests are managed on the github issue tracker. We use
automation to help prioritise and organise the issues.

The `good first issue` label should be assigned to any issue that could be
suitable for new contributors.

The `awaiting feedback` label should be assigned if more information is required
from the author, such as a reproducible example.

The [stale bot](https://github.com/actions/stale) will mark issues and PRs that
have not had any activity for a long period of time with the `stale` label, and
comment to solicit feedback from our community. If there is still no activity,
the issue will be closed after a further period of time.

We value feedback from our users very highly, so the bot is configured with long
time periods before marking issues as stale.

Issues marked with the `todo` label will never be marked as stale, so this label
should be assigned to any issues that should be kept open such as long-running
feature requests.

### PR triage

Pull Requests should generally be assigned a category label such as `bug`,
`enhancement` or `BREAKING`. These labels are used to categorise the PR in the
release notes, as described [below](#release-notes-from-pr-labels).

All PRs should have at least one review before being merged. In particular,
maintainers should generally ensure that PRs have sufficient unit tests to cover
any fixed bugs or new features.

PRs are usually completed with "squash and merge" in order to maintain a clear
linear history and make it easier to debug any issues.

### Versioning

shap uses a PEP 440-compliant versioning scheme of `MAJOR.MINOR.PATCH`. Like
[numpy][numpy_versioning], shap does *not* use semantic versioning, and has
never made a `major` release. Most releases increment `minor`, typically made
every month or two. `patch` releases are sometimes made for any important
bugfixes.

[numpy_versioning]: https://numpy.org/doc/stable/dev/depending_on_numpy.html

Breaking changes are done with care, given that shap is a very popular package.
When breaking changes are made, the PR should be tagged with the `BREAKING`
label to ensure it is highlighted in the release notes. Deprecation cycles are
used to mitigate the impact on downstream users.

GitHub milestones can be used to track any actions that need to be completed for
a given release, such as those relating to deprecation cycles.

We use `setuptools-scm` to source the version number from the git history
automatically. At build time, the version number is determined from the git tag.

### Minimum supported dependencies

We aim to follow the [SPEC 0](https://scientific-python.org/specs/spec-0000/) convention
on minimum supported dependencies.

- Support for Python versions are dropped 3 years after their initial release.
- Support for core package dependencies are dropped 2 years after their initial release.

We may support python versions for slightly longer than this window where it does
not add any extra maintenance burden.

### Making releases

We try to use automation to make the release process reliable, transparent and
reproducible. This also helps us make releases more frequently.

A release is made by publishing a [GitHub
Release](https://github.com/shap/shap/releases), tagged with an appropriately
incremented version number.

When a release is published, the wheels will be built and published to PyPI
automatically by the `build_wheels` GitHub action. This workflow can also be
triggered manually at any time to do a dry-run of cibuildwheel.

In the run-up to a release, create a GitHub issue for the release such as [[Meta
issue] Release 0.43.0](https://github.com/shap/shap/issues/3289). This can be
used to co-ordinate with other maintainers and agree to make a release.

Suggested release checklist:

```markdown
- [ ] Dry-run cibuildwheel & test
- [ ] Make GitHub release & tag
- [ ] Confirm PyPI wheels published
- [ ] Conda forge published
```

The conda package is managed in a [separate
repo](https://github.com/conda-forge/shap-feedstock). The conda-forge bot will
automatically make a PR to this repo to update the conda package, typically
within a few hours of the PyPSA package being published.

### Release notes from PR labels

Release notes can be automatically drafted by Github using the titles and labels
of PRs that were merged since the previous release. See the GitHub docs on
[automatically generated release notes][auto_release_notes] for more
information.

The generated notes will follow the template defined in
[.github/release.yml](.github/release.yml), arranging PRs into subheadings by
label and excluding PRs made by bots. See the [docs][auto_release_notes] for the
available configuration options.

[auto_release_notes]:
    https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes

It's helpful to assign labels such as `BREAKING`, `bug`, `enhancement` or
`skip-changelog` to each PR, so that the change will show up in the notes under
the right section. It also helps to ensure each PR has a descriptive name.

The notes can be edited (both before and after release) to remove information
that is unlikely to be of high interest to users, such as maintenance updates.
