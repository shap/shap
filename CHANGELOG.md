# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
<!--Changes from new PRs should be put in this section-->

### Added

- Added `n_points` parameter to all functions in `shap.datasets`
  ([dsgibbons#39](https://github.com/dsgibbons/shap/pull/39) by @thatlittleboy).
- Added `__call__` to `KernelExplainer`
  ([#2966](https://github.com/slundberg/shap/pull/2966) by @dwolfeu).
- Added [contributing guidelines](https://github.com/slundberg/shap/blob/master/CONTRIBUTING.md)
  ([#2996](https://github.com/slundberg/shap/pull/2996) by @connortann)

### Fixed

- Fixed `plot.waterfall` to support yticklabels with boolean features
  ([dsgibbons#58](https://github.com/dsgibbons/shap/pull/58) by @dwolfeu).
- Prevent `TreeExplainer.__call__` from throwing ValueError when passed a pandas DataFrame containing Categorical columns
  ([dsgibbons#88](https://github.com/dsgibbons/shap/pull/88) by @thatlittleboy).
- Fixed sampling in `shap.datasets` to sample without replacement
  ([dsgibbons#36](https://github.com/dsgibbons/shap/pull/36) by @thatlittleboy).
- Fixed an `UnboundLocalError` problem arising from passing a dictionary input to `shap.plots.bar`
  ([#3001](https://github.com/slundberg/shap/pull/3000) by @thatlittleboy).
- Fixed tensorflow import issue with Pyspark when using `Gradient`
  ([#2983](https://github.com/slundberg/shap/pull/2983) by @skamdar).
- Fixed the aspect ratio of the colorbar in `shap.plots.heatmap`, and use the `ax` matplotlib API internally
  for plotting
  ([#3040](https://github.com/slundberg/shap/pull/3040) by @thatlittleboy).
- Fixed deprecation warnings for `numpy>=1.24` from numpy types
  ([dsgibbons#7](https://github.com/dsgibbons/shap/pull/7) by @dsgibbons).
- Fixed deprecation warnings for `Ipython>=8` from `Ipython.core.display`
  ([dsgibbons#13](https://github.com/dsgibbons/shap/pull/13) by @thatlittleboy).
- Fixed deprecation warnings for `tensorflow>=2.11` from `tf.optimisers`
  ([dsgibbons#16](https://github.com/dsgibbons/shap/pull/16) by @simonangerbauer).
- Fixed deprecation warnings for `sklearn>=1.2` from `sklearn.linear_model`
  ([dsgibbons#22](https://github.com/dsgibbons/shap/pull/22) by @dsgibbons).
- Fixed deprecation warnings for `xgboost>=1.4` from `ntree_limit` in tree explainer
  ([#2987](https://github.com/slundberg/shap/pull/2987) by @adnene-guessoum).

### Changed

- Updates to docstrings of several `shap.plots` functions
  ([#3003](https://github.com/slundberg/shap/pull/3003),
   [#3005](https://github.com/slundberg/shap/pull/3005) by @thatlittleboy).

### Removed

- Deprecated the Boston house price dataset
  ([dsgibbons#38](https://github.com/dsgibbons/shap/pull/38) by @thatlittleboy).
- Removed the unused `mimic.py` file and `MimicExplainer` code
  ([dsgibbons#53](https://github.com/dsgibbons/shap/pull/53) by @thatlittleboy).

### Maintenance

- Fixed failing unit tests
  ([dsgibbons#29](https://github.com/dsgibbons/shap/pull/29) by @dsgibbons,
   [dsgibbons#20](https://github.com/dsgibbons/shap/pull/20) by @simonangerbauer,
   [#3044](https://github.com/slundberg/shap/pull/3044) by @connortann,
   [dsgibbons#24](https://github.com/dsgibbons/shap/pull/24) by @connortann).
- Include CUDA GPU C extension files in the source distribution
  ([#3009](https://github.com/slundberg/shap/pull/3009) by @jklaise).
- Fixed installation of package via setuptools
  ([dsgibbons#51](https://github.com/dsgibbons/shap/pull/51) by @thatlittleboy).
- Introduced a minimal set of `ruff` linting
  ([dsgibbons#25](https://github.com/dsgibbons/shap/pull/25),
   [dsgibbons#26](https://github.com/dsgibbons/shap/pull/26),
   [dsgibbons#27](https://github.com/dsgibbons/shap/pull/27),
   [#2973](https://github.com/slundberg/shap/pull/2973),
   [#2972](https://github.com/slundberg/shap/pull/2972) and
   [#2976](https://github.com/slundberg/shap/pull/2976) by @connortann,
   [#2968](https://github.com/slundberg/shap/pull/2968),
   [#2986](https://github.com/slundberg/shap/pull/2986) by @thatlittleboy).
- Fixed wheel packaging and updated project metadata to PEP 517
  ([#3022](https://github.com/slundberg/shap/pull/3022) by @connortann)
- Introduced more thorough testing on CI against newer dependencies
  ([dsgibbons#61](https://github.com/dsgibbons/shap/pull/61) and
   [#3017](https://github.com/slundberg/shap/pull/3017)
  by @connortann)


## [0.41.0] - 2022-06-16

For details of previous changes, see the release page on GitHub
[here](https://github.com/slundberg/shap/releases).
