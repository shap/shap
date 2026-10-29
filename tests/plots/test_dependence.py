import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap

# The following tests use shap.dependence_plot,
# which currently points to shap.plots._scatter.dependence_legacy


def test_random_dependence():
    """Make sure a dependence plot does not crash."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)


def test_random_dependence_no_interaction():
    """Make sure a dependence plot does not crash when we are not showing interactions."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)


def test_dependence_use_line_collection_bug():
    """Make sure a dependence plot does not crash."""
    # GH 3368
    sklearn = pytest.importorskip("sklearn")

    X, y = shap.datasets.california(n_points=10)

    X2 = shap.utils.sample(X, 2)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    explainer = shap.Explainer(model.predict, X2)
    shap_values = explainer(X2)
    shap.partial_dependence_plot(
        "MedInc",
        model.predict,
        X2,
        model_expected_value=True,
        feature_expected_value=True,
        ice=False,
        shap_values=shap_values[:1, :],  # type: ignore[call-overload]
        show=False,
    )


def test_dependence_polars_dataframe():
    """dependence_plot accepts a polars DataFrame and uses its column names."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20), "c": rng.normal(size=20)})
    shap_values = rng.normal(size=(20, 3))

    shap.dependence_plot("b", shap_values, X, interaction_index=None, show=False)

    assert plt.gca().get_xlabel() == "b"
    plt.close("all")


def test_dependence_polars_string_feature():
    """A polars DataFrame with a string column works like the pandas equivalent."""
    pl = pytest.importorskip("polars")
    X = pl.DataFrame(
        {
            "Sex": ["Male", "Female", "Male", "Male", "Female", "Female"],
            "Age": [10, 15, 28, 3, 84, 56],
        }
    )
    shap_values = np.random.randn(*X.shape)

    shap.dependence_plot("Sex", shap_values, X, interaction_index=None, show=False)

    plt.close("all")
