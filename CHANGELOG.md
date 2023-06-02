# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Added `n_points` parameter to all functions in `shap.datasets`
  ([#39](https://github.com/dsgibbons/shap/pull/39)).
- Added the `ruff` linter
  ([#25](https://github.com/dsgibbons/shap/pull/25),
   [#26](https://github.com/dsgibbons/shap/pull/26),
   [#27](https://github.com/dsgibbons/shap/pull/27)).

### Fixed

- Fixed failing unit tests
  ([#29](https://github.com/dsgibbons/shap/pull/29),
  [#20](https://github.com/dsgibbons/shap/pull/20),
  [#24](https://github.com/dsgibbons/shap/pull/24)).
- Fixed `plot.waterfall` to support yticklabels with boolean features
  ([#58](https://github.com/dsgibbons/shap/pull/58)).
- Prevent `TreeExplainer.__call__` from throwing ValueError when passed a pandas DataFrame containing Categorical columns
  ([#88](https://github.com/dsgibbons/shap/pull/88)).
- Fixed sampling in `shap.datasets` to sample without replacement
  ([#36](https://github.com/dsgibbons/shap/pull/36)).
- Fixed deprecation warnings for `numpy>=1.24` from numpy types
  ([#7](https://github.com/dsgibbons/shap/pull/7)).
- Fixed deprecation warnings for `Ipython>=8` from `Ipython.core.display`
  ([#13](https://github.com/dsgibbons/shap/pull/13)).
- Fixed deprecation warnings for `tensorflow>=2.11` from `tf.optimisers`
  ([#16](https://github.com/dsgibbons/shap/pull/16)).
- Fixed deprecation warnings for `sklearn>=1.2` from `sklearn.linear_model`
  ([#22](https://github.com/dsgibbons/shap/pull/22)).
- Fixed installation of package via setuptools
  ([#51](https://github.com/dsgibbons/shap/pull/51)).

### Changed


### Removed

- Deprecated the Boston house price dataset
  ([#38](https://github.com/dsgibbons/shap/pull/38)).
- Removed the unused `mimic.py` file and `MimicExplainer` code
  ([#53](https://github.com/dsgibbons/shap/pull/53)).

## [0.41.0] - 2022-06-16 (parent repo)

For details of previous changes, see the release page of the parent repository
[here](https://github.com/slundberg/shap/releases).
