"""Direct pandas/Polars parity checks for dataframe-based SHAP inputs."""

import narwhals as nw
import numpy as np
import pytest

import shap


@pytest.mark.parametrize("masker_type", [shap.maskers.Independent, shap.maskers.Partition])
def test_tabular_masker_dataframe_inputs_match(masker_type):
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")

    values = np.array(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
        dtype=np.float64,
    )
    columns = ["first", "second", "third"]
    pandas_frame = pandas.DataFrame(values, columns=columns)
    polars_frame = polars.DataFrame({name: values[:, i] for i, name in enumerate(columns)})
    partition = np.array([[0, 1, 1, 2], [2, 3, 1, 2]], dtype=np.float64)
    kwargs = {"partition": partition} if masker_type is shap.maskers.Partition else {}
    pandas_masker = masker_type(pandas_frame, **kwargs)
    polars_masker = masker_type(polars_frame, **kwargs)

    mask = np.array([True, False, True])
    sample = np.array([9.0, 90.0, 900.0])
    pandas_result = pandas_masker(mask, sample)
    polars_result = polars_masker(mask, sample)
    np.testing.assert_array_equal(
        nw.from_native(pandas_result).to_numpy(),
        nw.from_native(polars_result).to_numpy(),
    )
    assert list(nw.from_native(pandas_result).columns) == columns
    assert list(nw.from_native(polars_result).columns) == columns

    delta_masks = np.array([shap.utils.MaskedModel.delta_mask_noop_value, 1, -2], dtype=np.int32)
    (pandas_delta,), pandas_varying = pandas_masker(delta_masks, sample)
    (polars_delta,), polars_varying = polars_masker(delta_masks, sample)
    np.testing.assert_array_equal(
        nw.from_native(pandas_delta).to_numpy(),
        nw.from_native(polars_delta).to_numpy(),
    )
    np.testing.assert_array_equal(pandas_varying, polars_varying)


def test_sample_dataframe_inputs_match():
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")

    values = np.arange(60, dtype=np.float64).reshape(20, 3)
    columns = ["first", "second", "third"]
    pandas_frame = pandas.DataFrame(values, columns=columns)
    polars_frame = polars.DataFrame({name: values[:, i] for i, name in enumerate(columns)})

    pandas_sample = shap.utils.sample(pandas_frame, nsamples=7, random_state=0)
    polars_sample = shap.utils.sample(polars_frame, nsamples=7, random_state=0)
    np.testing.assert_array_equal(
        nw.from_native(pandas_sample).to_numpy(),
        nw.from_native(polars_sample).to_numpy(),
    )
    assert list(nw.from_native(pandas_sample).columns) == columns
    assert list(nw.from_native(polars_sample).columns) == columns


def test_hclust_dataframe_inputs_match():
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")

    values = np.arange(60, dtype=np.float64).reshape(20, 3)
    columns = ["first", "second", "third"]
    pandas_frame = pandas.DataFrame(values, columns=columns)
    polars_frame = polars.DataFrame({name: values[:, i] for i, name in enumerate(columns)})

    pandas_tree = shap.utils.hclust(pandas_frame)
    polars_tree = shap.utils.hclust(polars_frame)
    np.testing.assert_array_equal(pandas_tree, polars_tree)
