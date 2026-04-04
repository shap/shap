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
        "__getitem__",
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


def test_cohorts_with_array():
    """Test cohorts created from an array of cohort labels."""
    exp = shap.Explanation(
        values=np.random.RandomState(0).randn(10, 3),
        data=np.random.RandomState(0).randn(10, 3),
        feature_names=list("abc"),
    )
    labels = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"])
    ch = exp.cohorts(labels)
    assert isinstance(ch, shap.Cohorts)
    assert len(ch.cohorts) == 3


def test_cohorts_repr():
    """Test the __repr__ method of Cohorts."""
    exp = shap.Explanation(
        values=np.random.RandomState(0).randn(10, 3),
        data=np.random.RandomState(0).randn(10, 3),
        feature_names=list("abc"),
    )
    ch = exp.cohorts(np.array(["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"]))
    repr_str = repr(ch)
    assert "Cohorts" in repr_str
    assert "2 cohorts" in repr_str


def test_cohorts_invalid_type():
    """Test that cohorts raises TypeError for invalid input types."""
    exp = shap.Explanation(
        values=np.random.RandomState(0).randn(10, 3),
        data=np.random.RandomState(0).randn(10, 3),
        feature_names=list("abc"),
    )
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


def test_explanation_display_data_setter_with_dataframe():
    """Test that setting display_data with a DataFrame converts to values."""
    exp = shap.Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [30.0, 40.0]})
    exp.display_data = df
    assert isinstance(exp.display_data, np.ndarray)
    np.testing.assert_array_equal(exp.display_data, df.values)


def test_explanation_binary_operators():
    """Test arithmetic operations between Explanation objects."""
    vals = np.array([1.0, 2.0, 3.0])
    data = np.array([10.0, 20.0, 30.0])
    exp1 = shap.Explanation(values=vals.copy(), base_values=1.0, data=data.copy())
    exp2 = shap.Explanation(values=vals.copy(), base_values=2.0, data=data.copy())

    # add two explanations
    result = exp1 + exp2
    np.testing.assert_array_equal(result.values, [2.0, 4.0, 6.0])
    assert result.base_values == 3.0

    # subtract
    result = exp1 - exp2
    np.testing.assert_array_equal(result.values, [0.0, 0.0, 0.0])

    # multiply
    result = exp1 * exp2
    np.testing.assert_array_equal(result.values, [1.0, 4.0, 9.0])

    # divide
    result = exp1 / exp2
    np.testing.assert_array_equal(result.values, [1.0, 1.0, 1.0])


def test_explanation_binary_operators_with_scalar():
    """Test arithmetic operations with scalars."""
    exp = shap.Explanation(values=np.array([2.0, 4.0, 6.0]), base_values=1.0, data=np.array([1.0, 2.0, 3.0]))

    result = exp * 2
    np.testing.assert_array_equal(result.values, [4.0, 8.0, 12.0])

    result = exp / 2
    np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])

    # radd
    result = 1 + exp
    np.testing.assert_array_equal(result.values, [3.0, 5.0, 7.0])

    # rsub — shap's __rsub__ uses operator.sub(self.values, other), same as __sub__
    result = 10 - exp
    np.testing.assert_array_equal(result.values, [2.0 - 10, 4.0 - 10, 6.0 - 10])

    # rmul
    result = 3 * exp
    np.testing.assert_array_equal(result.values, [6.0, 12.0, 18.0])


def test_explanation_argsort_and_flip():
    """Test the argsort and flip properties."""
    exp = shap.Explanation(values=np.array([3.0, 1.0, 2.0]))

    argsorted = exp.argsort
    np.testing.assert_array_equal(argsorted.values, [1, 2, 0])

    flipped = exp.flip
    np.testing.assert_array_equal(flipped.values, [2.0, 1.0, 3.0])


def test_explanation_min_max():
    """Test min and max operations."""
    vals = np.array([[1.0, 5.0], [3.0, 2.0], [4.0, 6.0]])
    exp = shap.Explanation(values=vals, data=vals.copy())

    max_exp = exp.max(axis=0)
    np.testing.assert_array_equal(max_exp.values, [4.0, 6.0])

    min_exp = exp.min(axis=0)
    np.testing.assert_array_equal(min_exp.values, [1.0, 2.0])


def test_explanation_sum_with_grouping():
    """Test sum with feature grouping."""
    exp = shap.Explanation(
        values=np.array([[1.0, 2.0, 3.0, 4.0]]),
        data=np.array([[10.0, 20.0, 30.0, 40.0]]),
        feature_names=["a", "b", "c", "d"],
    )
    grouping = {"a": "group1", "b": "group1", "c": "group2", "d": "group2"}
    result = exp.sum(axis=1, grouping=grouping)
    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result.values, [[3.0, 7.0]])


def test_explanation_sum_grouping_rank1():
    """Test sum with grouping on a 1D explanation."""
    exp = shap.Explanation(
        values=np.array([1.0, 2.0, 3.0, 4.0]),
        data=np.array([10.0, 20.0, 30.0, 40.0]),
        feature_names=["a", "b", "c", "d"],
    )
    grouping = {"a": "group1", "b": "group1", "c": "group2", "d": "group2"}
    result = exp.sum(grouping=grouping)
    np.testing.assert_array_equal(result.values, [3.0, 7.0])


def test_explanation_sum_grouping_invalid_axis():
    """Test sum with grouping on invalid axis raises DimensionError."""
    from shap.utils._exceptions import DimensionError

    exp = shap.Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        feature_names=["a", "b"],
    )
    grouping = {"a": "group1", "b": "group1"}
    with pytest.raises(DimensionError, match="Only axis = 1"):
        exp.sum(axis=0, grouping=grouping)


def test_explanation_percentile():
    """Test percentile operation."""
    vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    exp = shap.Explanation(values=vals, data=vals.copy())

    p50 = exp.percentile(50, axis=0)
    np.testing.assert_array_equal(p50.values, [3.0, 4.0])


def test_explanation_sample():
    """Test the sample method."""
    vals = np.random.RandomState(0).randn(100, 5)
    exp = shap.Explanation(values=vals)

    sampled = exp.sample(10, random_state=42)
    assert sampled.shape == (10, 5)

    # sampling more than available without replace should cap at len
    sampled = exp.sample(200, random_state=42)
    assert sampled.shape == (100, 5)


def test_explanation_hclust_method():
    """Test the hclust method on Explanation."""
    vals = np.random.RandomState(0).randn(20, 5)
    exp = shap.Explanation(values=vals)

    order = exp.hclust()
    assert len(order) == 20
    assert set(order) == set(range(20))

    # test axis=1
    order = exp.hclust(axis=1)
    assert len(order) == 5
    assert set(order) == set(range(5))


def test_explanation_hclust_dimension_error():
    """Test that hclust raises DimensionError for non-2D arrays."""
    from shap.utils._exceptions import DimensionError

    exp = shap.Explanation(values=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(DimensionError, match="2D"):
        exp.hclust()


def test_explanation_shape_and_len():
    """Test shape and len properties."""
    exp = shap.Explanation(values=np.zeros((10, 5)))
    assert exp.shape == (10, 5)
    assert len(exp) == 10


def test_explanation_copy():
    """Test that __copy__ creates a proper copy."""
    import copy

    exp = shap.Explanation(
        values=np.array([1.0, 2.0]),
        base_values=0.5,
        data=np.array([10.0, 20.0]),
        feature_names=["a", "b"],
    )
    exp_copy = copy.copy(exp)
    assert isinstance(exp_copy, shap.Explanation)
    np.testing.assert_array_equal(exp_copy.values, exp.values)


def test_explanation_getitem_with_ellipsis():
    """Test slicing with Ellipsis."""
    exp = shap.Explanation(
        values=np.random.RandomState(0).randn(10, 5),
        feature_names=list("abcde"),
    )
    sliced = exp[..., :3]
    assert sliced.shape == (10, 3)


def test_explanation_getitem_with_explanation():
    """Test indexing with another Explanation's values."""
    vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    exp = shap.Explanation(values=vals)
    idx_exp = shap.Explanation(values=np.array([0, 2]))
    result = exp[idx_exp]
    assert result.shape == (2, 2)


def test_explanation_numpy_func_with_base_values():
    """Test _numpy_func handles base_values correctly."""
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    base = np.array([0.5, 0.6])
    exp = shap.Explanation(values=vals, base_values=base, data=vals.copy())

    result = exp.mean(axis=0)
    np.testing.assert_allclose(result.values, [2.0, 3.0])


def test_explanation_numpy_func_data_exception():
    """Test _numpy_func handles data that can't be reduced gracefully."""
    # string data can't be summed — should set data to None
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    exp = shap.Explanation(values=vals, data=[["a", "b"], ["c", "d"]])
    result = exp.mean(axis=0)
    assert result.data is None


def test_explanation_clustering_collapse():
    """Test that clustering is collapsed when axis=0 and clustering is 3D with no variance."""
    vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    # 3D clustering with same values across axis 0
    clustering = np.array([[[0.0, 1.0, 0.5, 2.0]], [[0.0, 1.0, 0.5, 2.0]]])
    exp = shap.Explanation(values=vals, clustering=clustering)
    result = exp.mean(axis=0)
    assert result.clustering is not None
    assert result.clustering.shape == (1, 4)


def test_metaclass_properties():
    """Test that MetaExplanation class properties return OpChain objects."""
    from shap.utils._general import OpChain

    assert isinstance(shap.Explanation.abs, OpChain)
    assert isinstance(shap.Explanation.identity, OpChain)
    assert isinstance(shap.Explanation.argsort, OpChain)
    assert isinstance(shap.Explanation.flip, OpChain)
    assert isinstance(shap.Explanation.sum, OpChain)
    assert isinstance(shap.Explanation.max, OpChain)
    assert isinstance(shap.Explanation.min, OpChain)
    assert isinstance(shap.Explanation.mean, OpChain)
    assert isinstance(shap.Explanation.sample, OpChain)
    assert isinstance(shap.Explanation.hclust, OpChain)


def test_metaclass_getitem():
    """Test that MetaExplanation __getitem__ returns OpChain."""
    from shap.utils._general import OpChain

    result = shap.Explanation[0]
    assert isinstance(result, OpChain)


def test_compute_shape_edge_cases():
    """Test _compute_shape with various edge cases."""
    from shap._explanation import _compute_shape

    # scalar
    assert _compute_shape(5) == ()

    # string
    assert _compute_shape("hello") == ()

    # list of strings
    assert _compute_shape(["a", "b", "c"]) == (None,)

    # empty list
    assert _compute_shape([]) == (0,)

    # single-element list
    assert _compute_shape([np.array([1, 2])]) == (1, 2)

    # nested list of arrays with inconsistent inner shapes
    assert _compute_shape([np.array([1, 2]), np.array([3, 4, 5])]) == (2, None)


def test_is_1d():
    """Test the is_1d utility."""
    from shap._explanation import is_1d

    assert is_1d(["a", "b", "c"]) is True
    assert is_1d([np.array([1, 2]), np.array([3, 4])]) is False


def test_list_wrap():
    """Test list_wrap utility function."""
    from shap._explanation import list_wrap

    # regular array passes through
    x = np.array([1.0, 2.0, 3.0])
    assert list_wrap(x) is x

    # 1D array of arrays gets wrapped into a list
    inner = [np.array([1, 2]), np.array([3, 4])]
    x = np.empty(2, dtype=object)
    x[0] = inner[0]
    x[1] = inner[1]
    result = list_wrap(x)
    assert isinstance(result, list)
    assert len(result) == 2

    # None passes through
    assert list_wrap(None) is None


def test_explanation_property_setters():
    """Test various property setters on Explanation."""
    exp = shap.Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        base_values=np.array([0.5, 0.5]),
        data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        feature_names=["a", "b"],
    )

    # test values setter
    new_vals = np.array([[5.0, 6.0], [7.0, 8.0]])
    exp.values = new_vals
    np.testing.assert_array_equal(exp.values, new_vals)

    # test base_values setter
    exp.base_values = np.array([1.0, 1.0])
    np.testing.assert_array_equal(exp.base_values, [1.0, 1.0])

    # test data setter
    new_data = np.array([[100.0, 200.0], [300.0, 400.0]])
    exp.data = new_data
    np.testing.assert_array_equal(exp.data, new_data)

    # test output_names setter
    exp.output_names = ["out1", "out2"]

    # test feature_names setter
    exp.feature_names = ["x", "y"]

    # test main_effects setter
    exp.main_effects = np.zeros((2, 2))
    np.testing.assert_array_equal(exp.main_effects, np.zeros((2, 2)))

    # test hierarchical_values setter
    exp.hierarchical_values = np.ones((2, 2))
    np.testing.assert_array_equal(exp.hierarchical_values, np.ones((2, 2)))

    # test clustering setter
    exp.clustering = np.array([[0, 1, 0.5, 2]])
    assert exp.clustering is not None
