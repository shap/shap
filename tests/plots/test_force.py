import matplotlib
import pytest
matplotlib.use('Agg')
import shap # pylint: disable=wrong-import-position

def test_random_force_plot_mpl_with_data():
    """ Test if force plot with matplotlib works.
    """

    RandomForestRegressor = pytest.importorskip('sklearn.ensemble').RandomForestRegressor

    # train model
    X, y = shap.datasets.boston()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)

def test_random_force_plot_mpl_text_rotation_with_data():
    """ Test if force plot with matplotlib works when supplied with text_rotation.
    """

    RandomForestRegressor = pytest.importorskip('sklearn.ensemble').RandomForestRegressor

    # train model
    X, y = shap.datasets.boston()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False)
