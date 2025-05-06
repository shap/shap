import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import shap


def test_summary_plot_wrong_features_shape():
    """Checks that ValueError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """

    rs = np.random.RandomState(42)

    emsg = (
        r"The shape of the shap_values matrix does not match the shape of the provided data matrix\. "
        r"Perhaps the extra column in the shap_values matrix is the constant offset\? Of so just pass shap_values\[:,:-1\]\."
    )
    with pytest.raises(ValueError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 4), show=False)

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(AssertionError, match=emsg):
        shap.summary_plot(rs.randn(20, 5), rs.randn(20, 1), show=False)


@pytest.mark.mpl_image_compare
def test_summary_plot(explainer):
    """Check a beeswarm chart renders correctly with shap_values as an Explanation
    object (default settings).
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.parametrize(
    "rng",
    [
        np.random.default_rng(167089660),
        17,
        np.random.SeedSequence(entropy=60767),
    ],
)
def test_summary_plot_seed_insulated(explainer, rng):
    # ensure that it is possible for downstream
    # projects to avoid mutating global NumPy
    # random state
    # see i.e., https://scientific-python.org/specs/spec-0007/
    shap_values = explainer(explainer.data)
    state_before = np.random.get_state()[1]  # type: ignore[index]
    shap.summary_plot(shap_values, show=False, rng=rng)
    state_after = np.random.get_state()[1]  # type: ignore[index]
    assert_array_equal(state_after, state_before)


def test_summary_plot_warning(explainer):
    # enforce FutureWarning for usage of global random
    # state as we prepare for SPEC 7 adoption
    shap_values = explainer(explainer.data)
    with pytest.warns(FutureWarning, match="NumPy global RNG"):
        shap.summary_plot(shap_values, show=False)
