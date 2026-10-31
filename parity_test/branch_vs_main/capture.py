"""Capture deterministic TreeExplainer outputs from the installed SHAP checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from parity_test import _utils


def capture(random_state: int, input_path: Path, output: Path) -> None:
    import shap

    with np.load(input_path, allow_pickle=False) as input_arrays:
        X_array = input_arrays["X"]
        regression_y = input_arrays["regression_y"]
        classification_y = input_arrays["classification_y"]
    n_features = X_array.shape[1]
    X = pd.DataFrame(X_array, columns=[f"feature_{i}" for i in range(n_features)])

    arrays: dict[str, np.ndarray] = {}
    for name, model, y in (
        ("regression", RandomForestRegressor, regression_y),
        ("classification", RandomForestClassifier, classification_y),
    ):
        estimator = model(n_estimators=9, max_depth=5, random_state=random_state, n_jobs=1)
        estimator.fit(X, y)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X.iloc[:12])
        explanation = explainer(X.iloc[:12])
        np.testing.assert_array_equal(np.asarray(shap_values), explanation.values)
        np.testing.assert_array_equal(np.asarray(explainer.expected_value), explanation.base_values[0])

        feature_names = [] if explanation.feature_names is None else list(explanation.feature_names)
        if feature_names != list(X.columns):
            raise AssertionError(f"{name} feature names differ: {feature_names}")

        arrays[f"{name}_shap_values"] = np.asarray(shap_values)
        arrays[f"{name}_call_values"] = np.asarray(explanation.values)
        arrays[f"{name}_call_base_values"] = np.asarray(explanation.base_values)
        arrays[f"{name}_expected_value"] = np.asarray(explainer.expected_value)
        arrays[f"{name}_feature_names"] = np.asarray(feature_names, dtype=str)
        arrays.update(_utils.explanation_to_arrays(f"{name}_explanation", explanation))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.random_state, args.input, args.output)


if __name__ == "__main__":
    main()
