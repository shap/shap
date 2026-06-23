"""Tests for shap/utils/_legacy.py - legacy data structures and utilities."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from shap.utils._legacy import (
    DenseData,
    DenseDataWithIndex,
    IdentityLink,
    Instance,
    InstanceWithIndex,
    LogitLink,
    Model,
    SparseData,
    convert_to_data,
    convert_to_instance,
    convert_to_instance_with_index,
    convert_to_link,
    convert_to_model,
    kmeans,
    match_instance_to_data,
    match_model_to_data,
)

# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------


class TestKmeans:
    def test_numpy_array(self):
        X = np.random.default_rng(0).standard_normal((50, 4))
        result = kmeans(X, k=5)
        assert isinstance(result, DenseData)
        assert result.data.shape == (5, 4)

    def test_dataframe_input(self):
        X = pd.DataFrame(
            np.random.default_rng(1).standard_normal((50, 3)),
            columns=["a", "b", "c"],
        )
        result = kmeans(X, k=3)
        assert isinstance(result, DenseData)
        assert list(result.group_names) == ["a", "b", "c"]

    def test_round_values_true(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        result = kmeans(X, k=2, round_values=True)
        # Each centre value should be one of the original values
        for centre in result.data:
            for j, val in enumerate(centre):
                assert val in X[:, j]

    def test_round_values_false(self):
        X = np.random.default_rng(2).standard_normal((30, 2))
        result = kmeans(X, k=3, round_values=False)
        assert result.data.shape == (3, 2)

    def test_weights_sum_to_one(self):
        X = np.random.default_rng(3).standard_normal((40, 3))
        result = kmeans(X, k=4)
        assert pytest.approx(result.weights.sum()) == 1.0


# ---------------------------------------------------------------------------
# Instance / InstanceWithIndex
# ---------------------------------------------------------------------------


class TestInstance:
    def test_basic_creation(self):
        x = np.array([[1.0, 2.0]])
        inst = Instance(x, None)
        assert np.array_equal(inst.x, x)
        assert inst.group_display_values is None

    def test_convert_to_instance_passthrough(self):
        inst = Instance(np.zeros((1, 2)), None)
        assert convert_to_instance(inst) is inst

    def test_convert_to_instance_from_array(self):
        arr = np.array([[3.0, 4.0]])
        inst = convert_to_instance(arr)
        assert isinstance(inst, Instance)
        assert np.array_equal(inst.x, arr)

    def test_instance_with_index_creation(self):
        x = np.array([[1.0, 2.0]])
        inst = InstanceWithIndex(x, ["a", "b"], [0], "idx", None)
        assert inst.index_name == "idx"
        assert inst.column_name == ["a", "b"]

    def test_instance_with_index_convert_to_df(self):
        x = np.array([[1.0, 2.0]])
        inst = InstanceWithIndex(x, ["a", "b"], [42], "row_id", None)
        df = inst.convert_to_df()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert df.index.name == "row_id"

    def test_convert_to_instance_with_index(self):
        val = np.array([[5.0, 6.0]])
        inst = convert_to_instance_with_index(val, ["x", "y"], [0], "id")
        assert isinstance(inst, InstanceWithIndex)
        assert inst.index_name == "id"


# ---------------------------------------------------------------------------
# DenseData / DenseDataWithIndex / SparseData
# ---------------------------------------------------------------------------


class TestDenseData:
    def test_basic_creation(self):
        data = np.ones((10, 3))
        dd = DenseData(data, ["a", "b", "c"])
        assert dd.data is data
        assert dd.group_names == ["a", "b", "c"]
        assert dd.groups_size == 3

    def test_weights_normalised(self):
        data = np.ones((5, 2))
        dd = DenseData(data, ["a", "b"])
        assert pytest.approx(dd.weights.sum()) == 1.0

    def test_custom_weights(self):
        data = np.ones((4, 2))
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        dd = DenseData(data, ["a", "b"], None, weights)
        assert pytest.approx(dd.weights.sum()) == 1.0

    def test_mismatched_names_raises(self):
        data = np.ones((5, 3))
        with pytest.raises(ValueError, match="# of names must match data matrix"):
            DenseData(data, ["a", "b"])  # only 2 names for 3 columns

    def test_mismatched_weights_raises(self):
        data = np.ones((5, 2))
        with pytest.raises(ValueError, match="# of weights must match data matrix"):
            DenseData(data, ["a", "b"], None, np.ones(3))  # 3 weights for 5 rows


class TestDenseDataWithIndex:
    def test_creation_and_convert_to_df(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        dd = DenseDataWithIndex(data, ["x", "y"], [10, 20], "sample_id")
        df = dd.convert_to_df()
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "sample_id"
        assert list(df.columns) == ["x", "y"]
        assert list(df.index) == [10, 20]


class TestSparseData:
    def test_from_csr_matrix(self):
        sp = scipy.sparse.csr_matrix(np.eye(4))
        sd = SparseData(sp)
        assert sd.groups_size == 4
        assert pytest.approx(sd.weights.sum()) == 1.0
        assert not sd.transposed


# ---------------------------------------------------------------------------
# convert_to_data
# ---------------------------------------------------------------------------


class TestConvertToData:
    def test_numpy_array(self):
        arr = np.ones((5, 3))
        result = convert_to_data(arr)
        assert isinstance(result, DenseData)
        assert result.data is arr

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = convert_to_data(df)
        assert isinstance(result, DenseData)
        assert list(result.group_names) == ["a", "b"]

    def test_pandas_dataframe_keep_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="row"))
        result = convert_to_data(df, keep_index=True)
        assert isinstance(result, DenseDataWithIndex)
        assert result.index_name == "row"

    def test_pandas_series(self):
        s = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
        result = convert_to_data(s)
        assert isinstance(result, DenseData)

    def test_sparse_matrix(self):
        sp = scipy.sparse.csr_matrix(np.eye(3))
        result = convert_to_data(sp)
        assert isinstance(result, SparseData)

    def test_sparse_matrix_non_csr_converted(self):
        sp = scipy.sparse.csc_matrix(np.eye(3))
        result = convert_to_data(sp)
        assert isinstance(result, SparseData)

    def test_densedata_passthrough(self):
        dd = DenseData(np.ones((3, 2)), ["a", "b"])
        assert convert_to_data(dd) is dd

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unknown type passed as data object"):
            convert_to_data("not a valid type")


# ---------------------------------------------------------------------------
# Link / IdentityLink / LogitLink / convert_to_link
# ---------------------------------------------------------------------------


class TestIdentityLink:
    def test_str(self):
        assert str(IdentityLink()) == "identity"

    def test_f_is_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(IdentityLink.f(x), x)

    def test_finv_is_identity(self):
        x = np.array([0.1, 0.5, 0.9])
        assert np.array_equal(IdentityLink.finv(x), x)


class TestLogitLink:
    def test_str(self):
        assert str(LogitLink()) == "logit"

    def test_f_and_finv_are_inverses(self):
        x = np.array([0.1, 0.5, 0.9])
        # LogitLink.finv is the inverse of LogitLink.f
        assert np.allclose(LogitLink.finv(LogitLink.f(x)), x, atol=1e-10)

    def test_finv(self):
        # finv(0) should be 0.5
        assert pytest.approx(LogitLink.finv(0.0)) == 0.5

    def test_f_clips_extremes(self):
        # Should not raise or produce inf for values at boundary
        result = LogitLink.f(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(result))


class TestConvertToLink:
    def test_identity_string(self):
        link = convert_to_link("identity")
        assert isinstance(link, IdentityLink)

    def test_logit_string(self):
        link = convert_to_link("logit")
        assert isinstance(link, LogitLink)

    def test_link_passthrough(self):
        link = IdentityLink()
        assert convert_to_link(link) is link

    def test_invalid_string_raises(self):
        with pytest.raises(TypeError):
            convert_to_link("unknown")


# ---------------------------------------------------------------------------
# Model / convert_to_model / match_model_to_data
# ---------------------------------------------------------------------------


class TestModel:
    def test_basic_creation(self):
        m = Model(lambda x: x, ["output"])
        assert m.out_names == ["output"]

    def test_convert_to_model_from_function(self):
        fn = lambda x: x.sum(axis=1)  # noqa: E731
        result = convert_to_model(fn)
        assert isinstance(result, Model)
        assert result.f is fn

    def test_convert_to_model_passthrough(self):
        m = Model(lambda x: x, None)
        assert convert_to_model(m) is m

    def test_match_model_to_data_non_model_raises(self):
        data = DenseData(np.ones((3, 2)), ["a", "b"])
        with pytest.raises(TypeError, match="model must be of type Model"):
            match_model_to_data("not a model", data)

    def test_match_model_to_data_1d_output(self):
        data = DenseData(np.ones((4, 2)), ["a", "b"])
        model = convert_to_model(lambda x: x.sum(axis=1))
        match_model_to_data(model, data)
        assert model.out_names == ["output value"]

    def test_match_model_to_data_2d_output(self):
        # Model outputs 2 columns → out_names should have 2 entries (named by shape[0])
        data = DenseData(np.ones((4, 2)), ["a", "b"])
        model = convert_to_model(lambda x: np.column_stack([x.sum(axis=1), x.sum(axis=1)]))
        out_val = match_model_to_data(model, data)
        # out_val has shape (4, 2); out_names count equals shape[0] for 2-D output
        assert len(model.out_names) == out_val.shape[0]


# ---------------------------------------------------------------------------
# match_instance_to_data
# ---------------------------------------------------------------------------


class TestMatchInstanceToData:
    def test_non_instance_raises(self):
        data = DenseData(np.ones((3, 2)), ["a", "b"])
        with pytest.raises(TypeError, match="instance must be of type Instance"):
            match_instance_to_data("not an instance", data)

    def test_fills_group_display_values(self):
        data = DenseData(np.ones((3, 2)), ["a", "b"])
        inst = Instance(np.array([[1.0, 2.0]]), None)
        match_instance_to_data(inst, data)
        assert len(inst.group_display_values) == len(data.groups)
