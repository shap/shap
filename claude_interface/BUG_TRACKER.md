# SHAP Bug Tracker — Interventional Tree SHAP Improvements

> **Last updated**: 2026-02-11
> **Scope**: Bugs found during implementation of interventional SHAP interaction values and categorical split support

---

## Our Commits (master branch)

| Commit | Description |
|--------|-------------|
| `869ee4cf` | Implement interventional SHAP interaction values (Zern et al. AAAI 2023) |
| `36e30249` | Expand brute-force test to cover 3-6 feature counts |
| `9ef0e88b` | Add categorical split support to interventional SHAP (CPU) |
| `18983ca1` | Add categorical split support to GPU TreeSHAP |
| `ef45a45c` | Fix `category_in_threshold` 0-based encoding bug + XGBoost `dmatrix_props` for interaction values |

---

## Bug 1: Interventional interaction values return all zeros

| Field | Value |
|-------|-------|
| **GitHub** | [#1824](https://github.com/shap/shap/issues/1824) (open since Feb 2021, 5 upvotes, 10 comments) |
| **Severity** | CRITICAL — silent data corruption |
| **Affects** | ALL tree models (sklearn, LightGBM, XGBoost, CatBoost) with `feature_perturbation="interventional"` |
| **Root cause** | Interventional interaction values were **never implemented** in C++. A guard assertion was commented out during refactoring, so `shap_interaction_values()` silently returns zeros. |
| **Status** | **FIXED** by commit `869ee4cf` |
| **Verified** | sklearn GBR, RandomForest, LightGBM, XGBoost, CatBoost all produce correct non-zero interaction values with symmetry, row-sum additivity, and prediction additivity |

---

## Bug 2: `category_in_threshold` broken for category value 0

| Field | Value |
|-------|-------|
| **GitHub** | No issue filed (discovered during our work) |
| **Severity** | HIGH — wrong predictions and SHAP values for LightGBM models with category 0 |
| **Affects** | LightGBM and GPBoost models where any categorical feature has value 0 |
| **Root cause** | Two-part encoding mismatch: |
| | **Python** (`_tree.py:1952`): `threshold += 2 ** (cat - 1)` — for cat=0 produces 0.5, truncated to 0 by `int()` cast |
| | **C++** (`tree_shap.h:183`): `1 << (int(category) - 1)` — for category=0, `1 << -1` is undefined behavior |
| **Impact** | Empirically confirmed: 63% relative error on predictions for samples with category 0. Affects `tree_predict()`, `tree_shap_recursive()`, `tree_shap_indep()`, `tree_shap_indep_interactions()` — all 12 call sites. |
| **Fix** | Change both to 0-based encoding: C++ `1 << int(category)`, Python `2 ** cat`. Single function change + single Python line. |
| **Status** | **FIXED** by commit `ef45a45c` |
| **Verified** | Test uses 0-based categories (0-3), prediction diff dropped from 63% to 0%. All call sites fixed: CPU `category_in_threshold()`, GPU `EvaluateSplit()`, Python `SingleTree` threshold construction. |

---

## Bug 3: XGBoost `dmatrix_props` not propagated for interaction values

| Field | Value |
|-------|-------|
| **GitHub** | [#3510](https://github.com/shap/shap/issues/3510), [#4091](https://github.com/shap/shap/issues/4091) (PR open) |
| **Severity** | MEDIUM — error raised, not silent |
| **Affects** | XGBoost models with `enable_categorical=True` using path-dependent interaction values |
| **Root cause** | `shap_interaction_values()` line 787 creates `xgboost.DMatrix(X)` without passing `**dmatrix_props`, so XGBoost doesn't know about categorical features |
| **Fix** | Pass `dmatrix_props` like `shap_values()` does at line 589-590 |
| **Status** | **FIXED** by commit `ef45a45c` |
| **Verified** | `shap_interaction_values()` now passes `_xgb_dmatrix_props` when creating `xgboost.DMatrix` |

---

## Bug 4: Categorical splits ignored in interventional SHAP code path

| Field | Value |
|-------|-------|
| **GitHub** | No specific issue (pre-existing, blocked by Python guards) |
| **Severity** | HIGH — wrong SHAP values for categorical models |
| **Affects** | LightGBM, XGBoost, CatBoost with categorical splits using interventional mode |
| **Root cause** | `tree_shap_indep()` and `tree_shap_indep_interactions()` always used `x > threshold` comparisons, ignoring `threshold_types`. GPU code also used numeric-only `ShapSplitCondition`. |
| **Status** | **FIXED** by commits `9ef0e88b` (CPU) and `18983ca1` (GPU) |
| **Verified** | LightGBM categorical test passes with SHAP value additivity, interaction symmetry, row-sum, and prediction checks |

---

## Bug 5: LightGBM multiclass + interaction values error (path-dependent)

| Field | Value |
|-------|-------|
| **GitHub** | [#3574](https://github.com/shap/shap/issues/3574) (18 comments) |
| **Severity** | MEDIUM — error raised for path-dependent mode |
| **Affects** | LightGBM multiclass classification with `tree_path_dependent` interaction values |
| **Root cause** | Leaf coverage check fails for multiclass LightGBM trees in `fully_defined_weighting` |
| **Status** | **WORKAROUND**: Our interventional mode now works correctly for LightGBM multiclass interactions. Verified with Iris dataset (shape `(5, 4, 4, 3)`). |

---

## Bug 6: LightGBM categorical splits crash interaction values

| Field | Value |
|-------|-------|
| **GitHub** | [#292](https://github.com/shap/shap/issues/292), [#248](https://github.com/shap/shap/issues/248), [#1644](https://github.com/shap/shap/issues/1644) |
| **Severity** | HIGH — crash (`ValueError: could not convert string to float`) |
| **Affects** | LightGBM with categorical features, path-dependent interaction values |
| **Root cause** | LightGBM represents categorical thresholds as pipe-delimited strings (`'2\|\|4\|\|5'`). The old `SingleTree` parsing code tries `float()` on these strings. |
| **Status** | **WORKAROUND**: Our interventional mode correctly handles categorical splits via the `threshold_types` / `category_in_threshold` mechanism in the C++ backend. Users can switch to `feature_perturbation="interventional"` to avoid the crash. Root cause in path-dependent tree parsing code is NOT fixed. |

---

## Summary

| Bug | Severity | Status | Commit / Action |
|-----|----------|--------|----------------|
| #1 Interventional interactions zeros | CRITICAL | **FIXED** | `869ee4cf` |
| #2 category_in_threshold cat=0 | HIGH | **FIXED** | `ef45a45c` |
| #3 XGBoost dmatrix_props | MEDIUM | **FIXED** | `ef45a45c` |
| #4 Categorical in interventional | HIGH | **FIXED** | `9ef0e88b`, `18983ca1` |
| #5 LightGBM multiclass interactions | MEDIUM | **WORKAROUND** | Use interventional mode |
| #6 LightGBM categorical crash | HIGH | **WORKAROUND** | Use interventional mode |

---

## GitHub Issues Our Work Closes

When submitting PRs, reference these issues:

- **Closes #1824** — Interventional interaction values now work (were always zeros)
- **Fixes #3510** — XGBoost dmatrix_props propagated for interaction values
- **Addresses #3574** — Interventional mode works for LightGBM multiclass interactions
- **Addresses #292, #248, #1644** — Interventional mode supports LightGBM categorical splits
