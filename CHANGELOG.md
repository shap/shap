# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
<!--Changes from new PRs should be put in this section-->

### Added

- Added `n_points` parameter to all functions in `shap.datasets`
  ([dsgibbons#39](https://github.com/dsgibbons/shap/pull/39)).
- Added `__call__` to `KernelExplainer`
  ([#2966](https://github.com/slundberg/shap/pull/2966)).
- Added the `ruff` linter
  ([dsgibbons#25](https://github.com/dsgibbons/shap/pull/25),
   [dsgibbons#26](https://github.com/dsgibbons/shap/pull/26),
   [dsgibbons#27](https://github.com/dsgibbons/shap/pull/27),
   [#2968](https://github.com/slundberg/shap/pull/2968),
   [#2973](https://github.com/slundberg/shap/pull/2973),
   [#2972](https://github.com/slundberg/shap/pull/2972),
   [#2976](https://github.com/slundberg/shap/pull/2976),
   [#2986](https://github.com/slundberg/shap/pull/2986)).

### Fixed

- Fixed failing unit tests
  ([dsgibbons#29](https://github.com/dsgibbons/shap/pull/29),
   [dsgibbons#20](https://github.com/dsgibbons/shap/pull/20),
   [dsgibbons#24](https://github.com/dsgibbons/shap/pull/24)).
- Fixed `plot.waterfall` to support yticklabels with boolean features
  ([dsgibbons#58](https://github.com/dsgibbons/shap/pull/58)).
- Prevent `TreeExplainer.__call__` from throwing ValueError when passed a pandas DataFrame containing Categorical columns
  ([dsgibbons#88](https://github.com/dsgibbons/shap/pull/88)).
- Fixed sampling in `shap.datasets` to sample without replacement
  ([dsgibbons#36](https://github.com/dsgibbons/shap/pull/36)).
- Fixed tensorflow import issue with Pyspark when using `Gradient`
  ([#2983](https://github.com/slundberg/shap/pull/2983)).
- Fixed deprecation warnings for `numpy>=1.24` from numpy types
  ([dsgibbons#7](https://github.com/dsgibbons/shap/pull/7)).
- Fixed deprecation warnings for `Ipython>=8` from `Ipython.core.display`
  ([dsgibbons#13](https://github.com/dsgibbons/shap/pull/13)).
- Fixed deprecation warnings for `tensorflow>=2.11` from `tf.optimisers`
  ([dsgibbons#16](https://github.com/dsgibbons/shap/pull/16)).
- Fixed deprecation warnings for `sklearn>=1.2` from `sklearn.linear_model`
  ([dsgibbons#22](https://github.com/dsgibbons/shap/pull/22)).
- Fixed deprecation warnings for `xgboost>=1.4` from `ntree_limit` in tree explainer
  ([#2987](https://github.com/slundberg/shap/pull/2987)).
- Fixed installation of package via setuptools
  ([dsgibbons#51](https://github.com/dsgibbons/shap/pull/51)).

### Changed


### Removed

- Deprecated the Boston house price dataset
  ([dsgibbons#38](https://github.com/dsgibbons/shap/pull/38)).
- Removed the unused `mimic.py` file and `MimicExplainer` code
  ([dsgibbons#53](https://github.com/dsgibbons/shap/pull/53)).

## [0.41.0] - 2022-06-16

For details of previous changes, see the release page on GitHub
[here](https://github.com/slundberg/shap/releases).
