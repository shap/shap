import numpy as np
import pytest

from shap.utils import hclust, partition_tree_shuffle
from shap.utils._exceptions import DimensionError


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_hclust_distance_random_state(linkage, metric):
    X = np.zeros((20, 5))
    first = hclust(X, linkage=linkage, metric=metric, random_state=42)
    second = hclust(X, linkage=linkage, metric=metric, random_state=42)
    np.testing.assert_array_equal(first, second)


def test_hclust_distance_preserves_global_random_state():
    old_state = np.random.get_state()
    try:
        np.random.seed(17)
        state = np.random.get_state()
        hclust(np.zeros((20, 5)), random_state=42)
        actual = np.random.get_state()
        assert actual[0] == state[0]
        np.testing.assert_array_equal(actual[1], state[1])
        assert actual[2:] == state[2:]
    finally:
        np.random.set_state(old_state)


def test_hclust_distance_random_state_object():
    X = np.zeros((20, 5))
    expected = hclust(X, random_state=42)
    rng = np.random.RandomState(42)
    actual = hclust(X, random_state=rng)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(rng.get_state()[1], np.random.RandomState(42).get_state()[1])


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_hclust_runs(linkage):
    # GH #3290
    pytest.importorskip("xgboost")
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    # just check if clustered ran successfully (using xgboost_distances_r2)
    clustered = hclust(X, y, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)

    # Check clustering runs if y=None (using scipy metrics)
    clustered = hclust(X, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)


@pytest.mark.parametrize(
    "X",
    [
        np.arange(1, 10),
        list(range(1, 10)),
    ],
)
def test_hclust_errors_on_input_shapes(X):
    # hclust only accepts 2-d arrays for X
    with pytest.raises(DimensionError):
        hclust(X, random_state=0)


def test_hclust_errors_on_unknown_linkages():
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    with pytest.raises(ValueError, match=r"Unknown linkage type:"):
        hclust(X, linkage="random-string", random_state=0)  # type: ignore


def test_partition_tree_shuffle_respects_tree_and_mask():
    partition_tree = np.array(
        [
            [0, 1, 0, 2],
            [2, 3, 0, 2],
            [4, 5, 0, 4],
        ],
        dtype=np.float64,
    )
    index_mask = np.array([True, False, True, True])
    indexes = np.empty(index_mask.sum(), dtype=np.int64)

    np.random.seed(0)
    partition_tree_shuffle(indexes, index_mask, partition_tree)

    assert set(indexes) == {0, 2, 3}
    assert abs(np.where(indexes == 2)[0][0] - np.where(indexes == 3)[0][0]) == 1
