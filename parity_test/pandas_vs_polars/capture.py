"""Capture pandas and Polars TreeExplainer outputs in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np

from parity_test import _utils


def capture(random_state: int, output: Path) -> None:
    import shap

    pandas_frame, polars_frame, labels = _utils.make_pandas_polars_frames(random_state)
    model = lgb.LGBMClassifier(
        n_estimators=7,
        n_jobs=1,
        random_state=random_state,
        verbosity=-1,
        min_child_samples=2,
    )
    model.fit(pandas_frame, labels)

    arrays: dict[str, np.ndarray] = {}
    explanations = {}
    for backend, frame in (("pandas", pandas_frame), ("polars", polars_frame)):
        shap_explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(shap_explainer.shap_values(frame))
        baseline = np.asarray(shap_explainer.expected_value)

        call_explainer = shap.TreeExplainer(model)
        explanation = call_explainer(frame)
        np.testing.assert_array_equal(shap_values, explanation.values)
        np.testing.assert_array_equal(baseline, explanation.base_values[0])
        feature_names = [] if explanation.feature_names is None else list(explanation.feature_names)
        if feature_names != _utils.FEATURE_NAMES:
            raise AssertionError(f"{backend} feature names differ: {feature_names}")
        explanations[backend] = explanation

        arrays[f"{backend}_shap_values"] = shap_values
        arrays[f"{backend}_baseline"] = baseline
        arrays[f"{backend}_call_values"] = np.asarray(explanation.values)
        arrays[f"{backend}_call_base_values"] = np.asarray(explanation.base_values)
        arrays[f"{backend}_feature_names"] = np.asarray(feature_names, dtype=str)
        arrays.update(_utils.explanation_to_arrays(f"{backend}_explanation", explanation))

    _utils.assert_explanations_equal(explanations["pandas"], explanations["polars"])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.random_state, args.output)


if __name__ == "__main__":
    main()
