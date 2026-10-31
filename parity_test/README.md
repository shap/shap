# TreeExplainer parity checks

These checks target the Polars input work and guard against changing existing SHAP results.

## pandas vs Polars

Run from the repository root:

```bash
python -m pytest parity_test/pandas_vs_polars/test_tree_parity.py -q
```

Hypothesis generates deterministic random states. Each case trains one LightGBM model on pandas input, then explains the same rows using pandas and Polars inputs. It compares `shap_values` and every `Explanation` field, including values, baselines, data, names, bounds, interactions, clustering, and operation history. Exact `compute_time` values are excluded because runtimes naturally differ; whether the field is populated is checked. The generated frames share the mixed numeric and categorical schema used in the LightGBM regression case.

`test_tree.py` is a copy of `tests/explainers/test_tree.py` for this PR check. Its `_validate_shap_values` helper also checks pandas and Polars outputs for both `__call__` and `shap_values`, and a 1,000-example Hypothesis test exercises that comparison with sklearn decision trees. The original tree test suite remains available in the copied file for broader regression checks.

This test requires LightGBM, pandas, Polars, and Hypothesis. It is expected to expose LightGBM versions that cannot accept Polars categorical input yet.
Hypothesis is not a core SHAP dependency; install it in the test environment if needed (`uv pip install hypothesis`).

## Current branch vs main

The branch comparison uses two checkouts and two Python environments so each side imports its own SHAP code. It compares the full `Explanation` fields as well as `shap_values`. Set `SHAP_MAIN_ROOT` to a main checkout and `SHAP_MAIN_PYTHON` to the Python executable installed against that checkout. Run the test from the candidate branch:

```bash
SHAP_MAIN_ROOT=/path/to/shap-main \
SHAP_MAIN_PYTHON=/path/to/main-env/bin/python \
PARITY_ARTIFACT_DIR=/tmp/shap-parity-artifacts \
python -m pytest parity_test/branch_vs_main/test_branch_parity.py -q
```

The current side uses the Python executable running pytest and the current checkout. Hypothesis generates a deterministic `random_state` for each case; the test writes one input fixture, then both checkouts load that exact fixture and use the same model seed. It saves `.npz` artifacts for both checkouts and compares `shap_values`, `__call__` values, baselines, and feature names exactly. Set `PARITY_ARTIFACT_DIR` to retain input and output artifacts for inspection; without it, temporary artifacts are used.

The main checkout must have the test dependencies installed and its SHAP extension built. The candidate checkout should likewise be installed in the environment running pytest.
