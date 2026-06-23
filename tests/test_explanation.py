"""This file contains tests for the `shap._explanation` module."""

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from pytest import param
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

import shap
from shap._explanation import OpHistoryItem


def test_explanation_repr():
    exp = shap.Explanation(values=np.arange(5))
    assert (
        exp.__repr__()
        == dedent(
            """
            .values =
            array([0, 1, 2, 3, 4])
            """
        ).strip()
    )

    exp = shap.Explanation(values=np.arange(5), base_values=0.5, data=np.ones(5))
    assert (
        exp.__repr__()
        == dedent(
            """
            .values =
            array([0, 1, 2, 3, 4])

            .base_values =
            0.5

            .data =
            array([1., 1., 1., 1., 1.])
            """
        ).strip()
    )


def test_explanation_hstack(random_seed):
    """Checks that `hstack` works as expected with two valid Explanation objects.
    And that it returns an Explanation object.
    """
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(random_seed)
    base_vals = np.ones(20) * 0.123
    exp1 = shap.Explanation(
        values=rs.randn(20, 7),
        base_values=base_vals,
    )
    exp2 = shap.Explanation(
        values=rs.randn(20, 5),
        base_values=base_vals,
    )
    new_exp = exp1.hstack(exp2)

    assert isinstance(new_exp, shap.Explanation)
    assert new_exp.values.shape == (20, 12)


def test_explanation_hstack_errors(random_seed):
    """Checks that `hstack` throws errors on invalid input."""
    # generate 2 Explanation objects for stacking
    rs = np.random.RandomState(random_seed)
    base_vals = np.ones(20) * 0.123
    base_exp = shap.Explanation(
        values=rs.randn(20, 5),
        base_values=base_vals,
    )

    with pytest.raises(
        AssertionError,
        match="Can't hstack explanations with different numbers of rows",
    ):
        exp2 = shap.Explanation(
            values=rs.randn(7, 5),
            base_values=np.ones(7),
        )
        _ = base_exp.hstack(exp2)

    with pytest.raises(
        ValueError,
        match="Can't hstack explanations with different base values",
    ):
        exp2 = shap.Explanation(
            values=rs.randn(20, 5),
            base_values=np.ones(20) * 0.987,
        )
        _ = base_exp.hstack(exp2)


@pytest.mark.parametrize("N", [4, 5, 6])
def test_feature_names_slicing_for_square_arrays(random_seed, N):
    """Checks that feature names in Explanations are properly sliced with "square"
    arrays (N==k).

    For 2D arrays, there is an ambiguity in how to assign the feature names to the
    slicer index. E.g. if feature_names is a list of 5 elements, and the shap_values is
    a (5,5) array, it's ambiguous whether the axis=0 or axis=1 refers to the "feature
    columns".

    This test ensures that we give higher priority to axis=1 for the feature_names for
    square arrays. Since most of the time, the 2D shap values arrays are assembled as
    (# samples, # features).

    cf. GH #2722, GH #2699.
    """
    rs = np.random.RandomState(random_seed)
    featnames = list("abcde")

    exp = shap.Explanation(
        # an array of this shape typically arises as the shap values of N samples, k=5 features
        values=rs.rand(N, 5),
        feature_names=featnames,
        output_names=featnames,
    )
    first_sample = exp[0]
    # exp[0] used to return "a" incorrectly when N=5 here, instead of ["a","b","c","d","e"]
    assert first_sample.feature_names == first_sample.output_names == featnames
    column_e = exp[..., "e"]
    assert column_e.feature_names == "e"


def test_populating_op_history():
    """Tests whether the Explanation.op_history attribute is populated properly after operations have been applied."""
    values = np.arange(-18, 17).reshape(7, 5)

    # apply some operations
    exp = shap.Explanation(values=values).abs.sample(5, random_state=0).flip[..., :3].mean(axis=0)
    exp += 2
    expected_op_names = [
        "abs",
        "sample",
        "flip",
        "__getitem__",
        "mean",
        "__add__",
    ]

    op_history = exp.op_history
    # sanity check for op_history
    assert len(op_history) == 6
    assert all(isinstance(op, OpHistoryItem) for op in op_history)
    assert [op.name for op in op_history] == expected_op_names

    # check that operations have been applied and produce the correct output
    assert np.allclose(exp.values, [10.8, 11.0, 11.6])


@pytest.mark.parametrize(
    "inp",
    [
        param(None, id="None"),
        param([1, 2, 3], id="list[int]"),
        param({"a": 10}, id="dict[int]"),
    ],
)
def test_cohorts_invalid_input(inp):
    with pytest.raises(TypeError):
        _ = shap.Cohorts(test_grp=inp)

    with pytest.raises(TypeError):
        ch = shap.Cohorts()
        ch.cohorts = inp


def test_cohorts_magic_methods(random_seed):
    rs = np.random.RandomState(random_seed)
    e_size = (1_000, 5)
    exp = shap.Explanation(
        values=rs.uniform(low=-1, high=1, size=e_size),
        data=rs.normal(loc=1, scale=3, size=e_size),
        feature_names=list("abcde"),
    )

    exp_neg = exp[exp[:, "a"].data < 0]
    exp_pos = exp[exp[:, "a"].data >= 0]
    ch = shap.Cohorts(col_a_neg=exp_neg, col_a_pos=exp_pos)

    # normal attribute access AND method access -> should be dispatched to Explanation objects
    new_ch = ch.abs.mean(axis=0)
    assert isinstance(new_ch, shap.Cohorts)
    assert np.allclose(
        new_ch.cohorts["col_a_neg"].values,
        exp_neg.abs.mean(axis=0).values,
    )
    assert np.allclose(
        new_ch.cohorts["col_a_pos"].values,
        exp_pos.abs.mean(axis=0).values,
    )

    # getitem access -> should be dispatched to Explanation objects
    new_ch = ch[..., "a"]
    assert isinstance(new_ch, shap.Cohorts)
    assert np.allclose(
        new_ch.cohorts["col_a_neg"].values,
        exp_neg[..., "a"].values,
    )
    assert np.allclose(
        new_ch.cohorts["col_a_pos"].values,
        exp_pos[..., "a"].values,
    )


def test_cohorts_magic_methods_errors():
    """We don't support dispatching __call__ to the Explanation objects in Cohorts.
    The only valid use case for cohorts.__call__() is for invoking Explanation methods (see above test).
    """
    ch = shap.Cohorts()
    with pytest.raises(ValueError, match=r"No methods"):
        ch(axis=0)


def test_cohorts_multi_class():
    # Load dataset
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    Y = data.target
    model = RandomForestClassifier(random_state=42)
    model.fit(X, Y)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X[:100])

    with pytest.raises(ValueError, match="Cohorts cannot be calculated on multiple outputs at once."):
        shap_values.cohorts(2)

    cohorts = shap_values[..., 0].cohorts(2)
    isinstance(cohorts, shap.Cohorts)


def test_cohorts_generation_with_one_feature():
    exp = shap.Explanation(
        values=np.random.uniform(low=-1, high=1, size=(500, 1)),
        data=np.random.normal(loc=1, scale=3, size=(500, 1)),
        feature_names=list("a"),
    )
    cohorts = exp.cohorts(3)
    assert isinstance(cohorts, shap.Cohorts)
    assert len(cohorts.cohorts) == 3


def test_cohorts_with_array_and_repr():
    """Test cohorts created from an array of labels, and that repr works."""
    exp = shap.Explanation(
        values=np.random.RandomState(0).randn(10, 3),
        data=np.random.RandomState(0).randn(10, 3),
        feature_names=list("abc"),
    )
    labels = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"])
    ch = exp.cohorts(labels)
    assert isinstance(ch, shap.Cohorts)
    assert len(ch.cohorts) == 3

    repr_str = repr(ch)
    assert "Cohorts" in repr_str
    assert "3 cohorts" in repr_str

    with pytest.raises(TypeError, match="not recognized"):
        exp.cohorts("invalid")


def test_explanation_init_from_explanation():
    """Test initializing an Explanation from another Explanation."""
    exp1 = shap.Explanation(
        values=np.array([1.0, 2.0, 3.0]),
        base_values=0.5,
        data=np.array([10.0, 20.0, 30.0]),
    )
    exp2 = shap.Explanation(exp1)
    np.testing.assert_array_equal(exp2.values, exp1.values)
    assert exp2.base_values == exp1.base_values
    np.testing.assert_array_equal(exp2.data, exp1.data)


@pytest.mark.parametrize(
    ("op", "other", "expected_values"),
    [
        param(lambda e, o: e + o, 1, [3.0, 5.0, 7.0], id="add_scalar"),
        param(lambda e, o: o + e, 1, [3.0, 5.0, 7.0], id="radd_scalar"),
        param(lambda e, o: e - o, 1, [1.0, 3.0, 5.0], id="sub_scalar"),
        param(lambda e, o: e * o, 2, [4.0, 8.0, 12.0], id="mul_scalar"),
        param(lambda e, o: e / o, 2, [1.0, 2.0, 3.0], id="truediv_scalar"),
    ],
)
def test_explanation_binary_operators_scalar(op, other, expected_values):
    """Test arithmetic operators with scalars and reverse variants."""
    exp = shap.Explanation(values=np.array([2.0, 4.0, 6.0]), base_values=1.0, data=np.array([1.0, 2.0, 3.0]))
    result = op(exp, other)
    np.testing.assert_array_equal(result.values, expected_values)


def test_explanation_binary_operators_between_explanations():
    """Arithmetic between two Explanations must propagate to base_values and data."""
    vals = np.array([1.0, 2.0, 3.0])
    data = np.array([10.0, 20.0, 30.0])
    exp1 = shap.Explanation(values=vals.copy(), base_values=1.0, data=data.copy())
    exp2 = shap.Explanation(values=vals.copy(), base_values=2.0, data=data.copy())

    result = exp1 + exp2
    np.testing.assert_array_equal(result.values, [2.0, 4.0, 6.0])
    assert result.base_values == 3.0
    np.testing.assert_array_equal(result.data, [20.0, 40.0, 60.0])


@pytest.mark.parametrize(
    ("prop", "expected"),
    [
        param("argsort", [1, 2, 0], id="argsort"),
        param("flip", [2.0, 1.0, 3.0], id="flip"),
    ],
)
def test_explanation_argsort_and_flip(prop, expected):
    """Test the argsort and flip properties."""
    exp = shap.Explanation(values=np.array([3.0, 1.0, 2.0]))
    np.testing.assert_array_equal(getattr(exp, prop).values, expected)


@pytest.mark.parametrize(
    ("method", "args", "expected"),
    [
        param("max", (), [4.0, 6.0], id="max"),
        param("min", (), [1.0, 2.0], id="min"),
        param("mean", (), [2.6666666666666665, 4.333333333333333], id="mean"),
        param("percentile", (50,), [3.0, 5.0], id="percentile_50"),
    ],
)
def test_explanation_reduction_ops(method, args, expected):
    """Reduction ops (min/max/mean/percentile) along axis=0."""
    vals = np.array([[1.0, 5.0], [3.0, 2.0], [4.0, 6.0]])
    exp = shap.Explanation(values=vals, data=vals.copy())
    result = getattr(exp, method)(*args, axis=0)
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize(
    ("values", "axis", "expected_shape"),
    [
        param(np.array([[1.0, 2.0, 3.0, 4.0]]), 1, (1, 2), id="rank2"),
        param(np.array([1.0, 2.0, 3.0, 4.0]), None, (2,), id="rank1"),
    ],
)
def test_explanation_sum_with_grouping(values, axis, expected_shape):
    """Sum with feature grouping on rank-1 and rank-2 explanations."""
    exp = shap.Explanation(
        values=values,
        data=values.copy(),
        feature_names=["a", "b", "c", "d"],
    )
    grouping = {"a": "group1", "b": "group1", "c": "group2", "d": "group2"}
    result = exp.sum(axis=axis, grouping=grouping)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.values.flatten(), [3.0, 7.0])


def test_explanation_sum_grouping_invalid_axis():
    """Sum with grouping on an invalid axis must raise DimensionError."""
    from shap.utils._exceptions import DimensionError

    exp = shap.Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        feature_names=["a", "b"],
    )
    with pytest.raises(DimensionError, match="Only axis = 1"):
        exp.sum(axis=0, grouping={"a": "group1", "b": "group1"})


@pytest.mark.parametrize(
    ("max_samples", "expected_rows"),
    [
        param(10, 10, id="subsample"),
        param(200, 100, id="cap_at_len"),
    ],
)
def test_explanation_sample(max_samples, expected_rows):
    """Test the sample method with subsampling and capping."""
    exp = shap.Explanation(values=np.random.RandomState(0).randn(100, 5))
    sampled = exp.sample(max_samples, random_state=42)
    assert sampled.shape == (expected_rows, 5)


@pytest.mark.parametrize("axis", [0, 1])
def test_explanation_hclust_method(axis):
    """Test the hclust method on Explanation along both axes."""
    vals = np.random.RandomState(0).randn(20, 5)
    exp = shap.Explanation(values=vals)
    order = exp.hclust(axis=axis)
    expected_len = 20 if axis == 0 else 5
    assert len(order) == expected_len
    assert set(order) == set(range(expected_len))


def test_explanation_hclust_dimension_error():
    """Test that hclust raises DimensionError for non-2D arrays."""
    from shap.utils._exceptions import DimensionError

    exp = shap.Explanation(values=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(DimensionError, match="2D"):
        exp.hclust()


def test_explanation_getitem_with_ellipsis_and_explanation():
    """Test slicing with Ellipsis and indexing with another Explanation."""
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    exp = shap.Explanation(values=vals, feature_names=list("abc"))
    assert exp[..., :2].shape == (3, 2)

    idx_exp = shap.Explanation(values=np.array([0, 2]))
    assert exp[idx_exp].shape == (2, 3)


def test_explanation_numpy_func_edge_cases():
    """Test _numpy_func with base_values, non-reducible data, and 3D clustering collapse."""
    # base_values reduction
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    exp = shap.Explanation(values=vals, base_values=np.array([0.5, 0.6]), data=vals.copy())
    result = exp.mean(axis=0)
    np.testing.assert_allclose(result.values, [2.0, 3.0])

    # string data can't be reduced — should set data to None
    exp2 = shap.Explanation(values=vals, data=[["a", "b"], ["c", "d"]])
    assert exp2.mean(axis=0).data is None

    # 3D clustering with no variance across axis 0 should collapse
    clustering = np.array([[[0.0, 1.0, 0.5, 2.0]], [[0.0, 1.0, 0.5, 2.0]]])
    exp3 = shap.Explanation(values=vals, clustering=clustering)
    result3 = exp3.mean(axis=0)
    assert result3.clustering is not None
    assert result3.clustering.shape == (1, 4)


def test_explanation_from_tree_explainer_operations():
    """Test Explanation operations using real TreeExplainer output.

    Exercises _compute_shape, list_wrap, and is_1d through the public
    Explanation interface rather than testing private functions directly.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = shap.datasets.california(n_points=50)
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X[:10])

    # shape and len work on real explainer output
    assert shap_values.shape == (10, 8)
    assert len(shap_values) == 10

    # reduction ops on real output
    mean_exp = shap_values.mean(axis=0)
    assert mean_exp.shape == (8,)

    # sample on real output
    sampled = shap_values.sample(5, random_state=0)
    assert sampled.shape == (5, 8)

    # display_data setter with DataFrame
    shap_values.display_data = X[:10]
    assert isinstance(shap_values.display_data, np.ndarray)
