# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Added `n_points` parameter to all functions in `shap.datasets` (#39).
- Added the ruff linter (#25, #26, #27).

### Fixed

- Fixed failing unit tests (#29, #20, #24).
- Fixed `plot.waterfall` yticklabels with boolean features (#58).
- Fixed sampling in `shap.datasets` to sample without replacement (#36).
- Fixed deprecation warnings for `numpy>=1.24` from numpy types (#7).
- Fixed deprecation warnings for `Ipython>=8` from `Ipython.core.display` (#13).
- Fixed deprecation warnings for `tensorflow>=2.11` from `tf.optimisers` (#16).
- Fixed deprecation warnings for `sklearn>=1.2` from `sklearn.linear_model` (#22).
- Fixed installation of package via setuptools (#51).

### Changed


### Removed

- Deprecated the Boston house price dataset (#38).
- Removed the unused `mimic.py` file and `MimicExplainer` code (#53).

## [0.41.0] - 2022-06-16 (parent repo)

For details of previous changes, see the release page of the parent repository
[here](https://github.com/slundberg/shap/releases).
