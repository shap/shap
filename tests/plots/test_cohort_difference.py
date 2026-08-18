import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import shap
from shap.utils._exceptions import DimensionError


@pytest.mark.mpl_image_compare
def test_cohort_difference():
    """Check that the cohort difference plot is rendered consistently."""
    np.random.seed(0)

    X, y = shap.datasets.iris()
    model = RandomForestClassifier(random_state=0, n_estimators=25).fit(X, y)
    shap_values = shap.TreeExplainer(model)(X)[..., 0]
    group_mask = np.random.randint(2, size=shap_values.shape[0]).astype(bool)
    cohorts = shap.Cohorts(baseline=shap_values[~group_mask], recent=shap_values[group_mask])

    fig, ax = plt.subplots()
    shap.plots.cohort_difference(cohorts, show=False, ax=ax, random_state=0)
    plt.tight_layout()
    return fig


def test_cohort_difference_raises_error_for_shape_mismatch():
    X, y = shap.datasets.iris()
    model = RandomForestClassifier(random_state=0, n_estimators=25).fit(X, y)
    shap_values = shap.TreeExplainer(model)(X)[..., 0]
    with pytest.raises(DimensionError, match="same number of feature columns"):
        shap.plots.cohort_difference(
            shap_values[:10],
            shap_values[:10, :-1],
            show=False,
        )
