# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased
<!--Changes from new PRs should be put in this section-->

### Added

- Added `show_values_in_legend` parameter to `summary_plot`
  ([#3062](https://github.com/slundberg/shap/pull/3062) by @101AlexMartin).

### Fixed

- Fixed segmentation fault errors on our MacOS test suite involving `lightgbm`
  ([#3093](https://github.com/slundberg/shap/pull/3093) by @thatlittleboy).
- Add support for LightGBM ensembles containing single leaf trees in `TreeExplainer`
  ([#3094](https://github.com/slundberg/shap/pull/3094) by @thatlittleboy).

### Changed

### Removed

- Dropped support for python 3.7 ([#3079](https://github.com/slundberg/shap/pull/3079) by @connortann)

## 0.42.1 - 2023-07-11

Patch release to provide wheels for a broader range of architectures.

- Added wheels for `linux:aarch64` and `macos:arm64`
  ([#3078](https://github.com/slundberg/shap/pull/3078) by @PrimozGodec and
  [#3083](https://github.com/slundberg/shap/pull/3083) by @connortann).
- Fixed circular import issues with `shap.benchmark`
  ([#3076](https://github.com/slundberg/shap/pull/3076) by @thatlittleboy).

## 0.42.0 - 2023-07-05

This release incorporates many changes that were originally contributed by the SHAP
community via @dsgibbons's [Community Fork][fork], which has now been merged
into the main shap repository. PRs from this origin are labelled here as `fork#123`.

[fork]: https://github.com/slundberg/shap/discussions/2942

### Added

- Added support for python 3.11
  ([fork#72](https://github.com/dsgibbons/shap/pull/72) by @connortann).
- Added `n_points` parameter to all functions in `shap.datasets`
  ([fork#39](https://github.com/dsgibbons/shap/pull/39) by @thatlittleboy).
- Added `__call__` to `KernelExplainer`
  ([#2966](https://github.com/slundberg/shap/pull/2966) by @dwolfeu).
- Added [contributing guidelines][contrib-guide]
  ([#2996](https://github.com/slundberg/shap/pull/2996) by @connortann).

[contrib-guide]: [https://github.com/slundberg/shap/blob/master/CONTRIBUTING.md]

### Fixed

- Fixed `plot.waterfall` to support yticklabels with boolean features
  ([fork#58](https://github.com/dsgibbons/shap/pull/58) by @dwolfeu).
- Prevent `TreeExplainer.__call__` from throwing ValueError when passed a pandas
  DataFrame containing Categorical columns
  ([fork#88](https://github.com/dsgibbons/shap/pull/88) by @thatlittleboy).
- Fixed sampling in `shap.datasets` to sample without replacement
  ([fork#36](https://github.com/dsgibbons/shap/pull/36) by @thatlittleboy).
- Fixed an `UnboundLocalError` problem arising from passing a dictionary input to `shap.plots.bar`
  ([#3001](https://github.com/slundberg/shap/pull/3001) by @thatlittleboy).
  ([fork#36](https://github.com/dsgibbons/shap/pull/36) by @thatlittleboy).
- Fixed an `UnboundLocalError` problem arising from passing a dictionary input
  to `shap.plots.bar`
  ([#3001](https://github.com/slundberg/shap/pull/3000) by @thatlittleboy).
- Fixed tensorflow import issue with Pyspark when using `Gradient`
  ([#2983](https://github.com/slundberg/shap/pull/2983) by @skamdar).
- Fixed the aspect ratio of the colorbar in `shap.plots.heatmap`, and use the
  `ax` matplotlib API internally for plotting
  ([#3040](https://github.com/slundberg/shap/pull/3040) by @thatlittleboy).
- Fixed deprecation warnings for `numba>=0.44`
  ([fork#9](https://github.com/dsgibbons/shap/pull/9) and
  [fork#68](https://github.com/dsgibbons/shap/pull/68) by @connortann).
- Fixed deprecation warnings for `numpy>=1.24` from numpy types
  ([fork#7](https://github.com/dsgibbons/shap/pull/7) by @dsgibbons).
- Fixed deprecation warnings for `Ipython>=8` from `Ipython.core.display`
  ([fork#13](https://github.com/dsgibbons/shap/pull/13) by @thatlittleboy).
- Fixed deprecation warnings for `tensorflow>=2.11` from `tf.optimisers`
  ([fork#16](https://github.com/dsgibbons/shap/pull/16) by @simonangerbauer).
- Fixed deprecation warnings for `sklearn>=1.2` from `sklearn.linear_model`
  ([fork#22](https://github.com/dsgibbons/shap/pull/22) by @dsgibbons).
- Fixed deprecation warnings for `xgboost>=1.4` from `ntree_limit` in tree explainer
  ([#2987](https://github.com/slundberg/shap/pull/2987) by @adnene-guessoum).
- Fixed build on Windows and MacOS
  ([#3015](https://github.com/slundberg/shap/pull/3015) by @PrimozGodec;
  [#3028](https://github.com/slundberg/shap/pull/3028),
  [#3029](https://github.com/slundberg/shap/pull/3029) and
  [#3031](https://github.com/slundberg/shap/pull/3031) by @connortann).
- Fixed creation of ragged arrays in `shap.explainers.Exact`
  ([#3064](https://github.com/slundberg/shap/pull/3064) by @connortann).

### Changed

- Updates to docstrings of several `shap.plots` functions
  ([#3003](https://github.com/slundberg/shap/pull/3003),
   [#3005](https://github.com/slundberg/shap/pull/3005) by @thatlittleboy).

### Removed

- Deprecated the Boston house price dataset
  ([fork#38](https://github.com/dsgibbons/shap/pull/38) by @thatlittleboy).
- Removed the unused `mimic.py` file and `MimicExplainer` code
  ([fork#53](https://github.com/dsgibbons/shap/pull/53) by @thatlittleboy).

### Maintenance

- Fixed failing unit tests
  ([fork#29](https://github.com/dsgibbons/shap/pull/29) by @dsgibbons,
   [fork#20](https://github.com/dsgibbons/shap/pull/20) by @simonangerbauer,
   [#3044](https://github.com/slundberg/shap/pull/3044) and
   [fork#24](https://github.com/dsgibbons/shap/pull/24) by @connortann).
- Include CUDA GPU C extension files in the source distribution
  ([#3009](https://github.com/slundberg/shap/pull/3009) by @jklaise).
- Fixed installation of package via setuptools
  ([fork#51](https://github.com/dsgibbons/shap/pull/51) by @thatlittleboy).
- Introduced a minimal set of `ruff` linting
  ([fork#25](https://github.com/dsgibbons/shap/pull/25),
   [fork#26](https://github.com/dsgibbons/shap/pull/26),
   [fork#27](https://github.com/dsgibbons/shap/pull/27),
   [#2973](https://github.com/slundberg/shap/pull/2973),
   [#2972](https://github.com/slundberg/shap/pull/2972) and
   [#2976](https://github.com/slundberg/shap/pull/2976) by @connortann;
   [#2968](https://github.com/slundberg/shap/pull/2968),
   [#2986](https://github.com/slundberg/shap/pull/2986) by @thatlittleboy).
- Updated project metadata to PEP 517
  ([#3022](https://github.com/slundberg/shap/pull/3022) by @connortann).
- Introduced more thorough testing on CI against newer dependencies
  ([fork#61](https://github.com/dsgibbons/shap/pull/61) and
   [#3017](https://github.com/slundberg/shap/pull/3017)
  by @connortann)
- Reduced unit test time by ~5 mins
  ([#3046](https://github.com/slundberg/shap/pull/3046) by @connortann).
- Introduced fixtures for reproducible fuzz testing
  ([#3048](https://github.com/slundberg/shap/pull/3048) by @connortann).


## [0.41.0] - 2022-06-16

For details of previous changes, see the release page on GitHub
[here](https://github.com/slundberg/shap/releases).
