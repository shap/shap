from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


FEATURE_NAMES = [f"f{i}" for i in range(13)]
EXPLANATION_FIELDS = (
    "values",
    "base_values",
    "data",
    "display_data",
    "instance_names",
    "feature_names",
    "output_names",
    "output_indexes",
    "lower_bounds",
    "upper_bounds",
    "error_std",
    "main_effects",
    "hierarchical_values",
    "clustering",
    "output_dims",
    "op_history",
    "compute_time",
)


def explanation_to_arrays(prefix: str, explanation: Any) -> dict[str, np.ndarray]:
    """Serialize every public Explanation field for cross-process comparisons."""
    arrays: dict[str, np.ndarray] = {}
    for field in EXPLANATION_FIELDS:
        value = getattr(explanation, field)
        arrays[f"{prefix}_{field}_is_none"] = np.asarray(value is None)
        if value is None:
            continue
        if field == "compute_time":
            # Execution time is expected to differ; retain whether it was populated.
            continue
        if field == "op_history":
            value = [repr(operation) for operation in value]

        array = np.asarray(value)
        arrays[f"{prefix}_{field}_dtype"] = np.asarray(str(array.dtype))
        if array.dtype == object:
            array = array.astype(str)
        arrays[f"{prefix}_{field}"] = array
    return arrays


def assert_explanations_equal(reference: Any, candidate: Any) -> None:
    """Compare all Explanation fields, ignoring only the exact runtime duration."""
    if type(reference) is not type(candidate):
        raise AssertionError(f"Explanation types differ: {type(reference)} != {type(candidate)}")
    for field in EXPLANATION_FIELDS:
        left = getattr(reference, field)
        right = getattr(candidate, field)
        if field == "compute_time":
            if (left is None) != (right is None):
                raise AssertionError("Explanation compute_time presence differs")
        elif left is None or right is None:
            if left is not right:
                raise AssertionError(f"Explanation field {field!r} differs: {left!r} != {right!r}")
        else:
            if field == "op_history":
                left = [repr(operation) for operation in left]
                right = [repr(operation) for operation in right]
            np.testing.assert_array_equal(
                np.asarray(left),
                np.asarray(right),
                err_msg=f"Explanation field {field!r} differs",
            )
            if field not in {"op_history"} and np.asarray(left).dtype != np.asarray(right).dtype:
                raise AssertionError(
                    f"Explanation field {field!r} dtypes differ: {np.asarray(left).dtype} != {np.asarray(right).dtype}"
                )


def make_pandas_polars_frames(random_state: int, n_rows: int = 96) -> tuple[Any, Any, np.ndarray]:
    """Create identical pandas and Polars frames with numeric and categorical dtypes."""
    import pandas as pd
    import polars as pl

    rng = np.random.default_rng(random_state)
    columns = {
        "f0": rng.normal(size=n_rows).astype(np.float32),
        "f1": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
        "f2": rng.normal(size=n_rows).astype(np.float32),
        "f3": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
        "f4": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
        "f5": rng.integers(0, 1000, size=n_rows, dtype=np.int64),
        "f6": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
        "f7": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
        "f8": rng.normal(size=n_rows).astype(np.float32),
        "f9": rng.normal(size=n_rows).astype(np.float32),
        "f10": rng.normal(size=n_rows).astype(np.float32),
        "f11": rng.integers(-5, 6, size=n_rows, dtype=np.int8),
    }
    categories = rng.choice(np.array(["M", "F"]), size=n_rows)
    pandas_frame = pd.DataFrame(columns)
    pandas_frame["f12"] = pd.Categorical(categories, categories=["F", "M"], ordered=False)
    polars_frame = pl.DataFrame(columns).with_columns(pl.Series("f12", categories, dtype=pl.Categorical))

    # Make the category affect the target so the fitted model must learn its split.
    score = columns["f0"] + (categories == "M") * 0.7 + columns["f1"] * 0.05
    labels = (score > np.median(score)).astype(np.int32)
    return pandas_frame, polars_frame, labels


def make_numeric_pandas_polars_frames(random_state: int = 0, n_rows: int = 32) -> tuple[Any, Any, np.ndarray]:
    """Create paired numeric frames for explainer and plotting parity checks."""
    import pandas as pd
    import polars as pl

    rng = np.random.default_rng(random_state)
    values = rng.normal(size=(n_rows, 3))
    names = ["a", "b", "c"]
    pandas_frame = pd.DataFrame(values, columns=names)
    polars_frame = pl.DataFrame({name: values[:, i] for i, name in enumerate(names)})
    target = values[:, 0] - 0.5 * values[:, 1] + 0.25 * values[:, 2]
    return pandas_frame, polars_frame, target


def assert_plot_frames_equal(plot_function: Any, pandas_frame: Any, polars_frame: Any) -> None:
    """Compare rendered plot pixels after passing each backend frame to a plot function."""
    import matplotlib.pyplot as plt

    images = []
    for frame in (pandas_frame, polars_frame):
        plt.close("all")
        plot_function(frame)
        figure = plt.gcf()
        figure.canvas.draw()
        images.append(np.asarray(figure.canvas.buffer_rgba()).copy())
    plt.close("all")
    np.testing.assert_array_equal(images[0], images[1])


def assert_npz_equal(reference_path: str | Path, candidate_path: str | Path) -> None:
    """Require the same keys, shapes, and exact values in two saved output bundles."""
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        if set(reference.files) != set(candidate.files):
            raise AssertionError(f"Artifact keys differ: {reference.files} != {candidate.files}")
        for key in reference.files:
            np.testing.assert_array_equal(reference[key], candidate[key], err_msg=f"Mismatch in {key}")


def assert_npz_groups_equal(path: str | Path, left: str, right: str) -> None:
    """Compare matching suffixes from two named groups in one artifact bundle."""
    with np.load(path, allow_pickle=False) as artifacts:
        left_keys = {key.removeprefix(f"{left}_") for key in artifacts.files if key.startswith(f"{left}_")}
        right_keys = {key.removeprefix(f"{right}_") for key in artifacts.files if key.startswith(f"{right}_")}
        if left_keys != right_keys:
            raise AssertionError(f"Artifact groups differ: {sorted(left_keys)} != {sorted(right_keys)}")
        for suffix in sorted(left_keys):
            np.testing.assert_array_equal(
                artifacts[f"{left}_{suffix}"],
                artifacts[f"{right}_{suffix}"],
                err_msg=f"Mismatch for {left}/{right} {suffix}",
            )
