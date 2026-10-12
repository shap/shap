CPU Quadrature TreeSHAP
======================

``TreeExplainer`` uses Quadrature TreeSHAP in its CPU extension for exact
``tree_path_dependent`` values and pairwise interactions. Its public arguments,
output shapes, split conventions, and training-cover/background semantics stay
the same. Interventional explanations and ``approximate=True`` use their
existing algorithms. ``GPUTreeExplainer`` retains its existing GPU backend.

Algorithm and precision
-----------------------

The implementation adapts XGBoost's return-edge quadrature recurrence from
``src/predictor/interpretability/shap.cc`` and ``quadrature.h``, pinned to
commit ``51b132efe13cb39e6cb424394980ac4610cbe8e9``. The source attribution and
Apache-2.0 license are distributed alongside the adapted header under
``shap/cext/quadrature_tree_shap.NOTICE`` and
``shap/cext/quadrature_tree_shap.LICENSE``.

For a path with :math:`m` distinct features, the additive SHAP integrand is a
polynomial of degree at most :math:`m-1`. An :math:`n`-point Gauss-Legendre rule
integrates polynomials through degree :math:`2n-1`. We therefore use
:math:`\lceil m_{\max}/2\rceil` points per tree, with a minimum of one for constant
trees. Repeated splits on a feature do not increase this count. Interactions
use the same rule because their diagonal entries require additive SHAP values.
This is polynomial exactness; floating-point rounding still applies.

Compared with upstream's fixed eight-point float32 implementation, the port:

* generates nodes and weights and accumulates results in float64;
* selects the sufficient quadrature order for each tree rather than fixing it
  at eight, including for trees with more than 16 distinct path features;
* retains small positive branch probabilities without applying a float32 floor;
* combines path weights with the polynomial basis during descent;
* derives the expected value from leaf values and cover ratios rather than
  relying on imported internal-node values;
* supports SHAP's numerical comparisons, missing/default branches, LightGBM
  categorical bitmasks, multiple outputs, tree limits, and interaction layout.

Pairwise off-diagonal contributions are symmetrized, and each diagonal is the
feature's SHAP value minus its off-diagonal row sum.

Dispatch and compatibility
--------------------------

Dense numerical XGBoost and LightGBM inputs use the CPU quadrature backend
when SHAP has a usable dense model adapter. The existing native implementations
remain fallbacks for native containers or sparse inputs, unsupported categorical
encodings, XGBoost custom missing sentinels, DART and boosted forests, LightGBM special
missing-value settings and native categorical encodings, and models with
incomplete cover information. CatBoost retains native dispatch because its
scale/bias and categorical metadata are not fully represented by the dense
adapter. These fallbacks preserve existing capabilities; they do not constitute
a replacement of the algorithms inside those external libraries.

For exact CPU calls, ``expected_value`` is refreshed from the computed bias
term, and additivity validation uses the requested tree limit. Approximate
explanations retain their existing baseline handling.

Validation and performance monitoring
-------------------------------------

``tests/explainers/test_quadrature_tree.py`` compares values and interactions
with exhaustive coalition enumeration on small trees, including repeated
categorical splits, missing values, multiple outputs, and tree limits. Analytic
conjunction tests cover paths with up to 129 distinct features, repeated-feature
paths, rare branches, and constant trees. Model-library tests verify CPU
dispatch, native parity, and fallback behavior.

``monitoring/tree_explainer.py`` provides the ASV ``TreeQuadratureSuite`` for
values and interactions at depths 4, 8, and 12, with one and three outputs. It
uses deterministic synthetic data and needs no dataset downloads. Run it in an
installed development environment with::

    asv run --environment=existing:python --bench=TreeQuadratureSuite

References
----------

* Wettenstein, Mitchell, and Yu, `Quadrature-TreeSHAP: Depth-Independent TreeSHAP
  and Shapley Interactions <https://arxiv.org/abs/2605.04497>`_.
* `Pinned XGBoost implementation
  <https://github.com/dmlc/xgboost/tree/51b132efe13cb39e6cb424394980ac4610cbe8e9/src/predictor/interpretability>`_.
