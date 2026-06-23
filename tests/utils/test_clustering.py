import numpy as np
import pytest

from shap.utils import hclust
from shap.utils._exceptions import DimensionError


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


def test_hclust_abs_correlation_groups_anti_correlated_features():
    """Regression test for https://github.com/shap/shap/issues/4134.

    With ``metric="abs_correlation"``, a perfectly anti-correlated pair of
    features clusters at near-zero distance — the same as a perfectly
    correlated pair — because knowing one determines the other. The standard
    ``"correlation"`` metric treats them as maximally distant (d = 2.0).
    """
    rs = np.random.RandomState(0)
    n = 200
    base = rs.randn(n)
    # Features 0, 1, 2 are all mutually (anti-)correlated with magnitude ~1:
    #   0 and 1 are (near) perfectly correlated,
    #   2 is perfectly anti-correlated to 0 and 1.
    # Feature 3 is independent noise.
    X = np.column_stack([base, base + rs.randn(n) * 1e-6, -base, rs.randn(n)])

    np.random.seed(0)
    clustered_abs = hclust(X, metric="abs_correlation")
    np.random.seed(0)
    clustered_corr = hclust(X, metric="correlation")

    # Column 2 of the linkage matrix is the cluster merge distance.
    # Under abs_correlation, features {0, 1, 2} form a tight cluster: both
    # of the first two merges should happen at distance ~0.
    first_two_abs = clustered_abs[:2, 2]
    assert np.all(first_two_abs < 0.05), (
        f"abs_correlation: expected the first two merges at distance ~0 "
        f"(three mutually (anti-)correlated features should cluster tightly). "
        f"Got {first_two_abs}."
    )

    # Under plain correlation, only feature 1 merges tightly with feature 0;
    # feature 2 sits at distance ~2 and cannot merge until far later.
    first_two_corr = clustered_corr[:2, 2]
    assert first_two_corr[0] < 0.05, (
        f"correlation: first merge should be at distance ~0 (features 0, 1 "
        f"are near-identical). Got {first_two_corr[0]}."
    )
    assert first_two_corr[1] > 0.5, (
        f"correlation: second merge should be far — the anti-correlated feature "
        f"is treated as maximally distant. Got {first_two_corr[1]}. If this fails, "
        f"the premise of the abs_correlation test is undermined."
    )
