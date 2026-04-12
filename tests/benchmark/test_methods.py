import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import shap.benchmark.methods as methods


# ---------- helpers ----------

@pytest.fixture
def dummy_data():
    return np.random.randn(300, 5)


@pytest.fixture
def dummy_X():
    return np.random.randn(10, 5)


# ---------- simple functions ----------

@patch("shap.benchmark.methods.LinearExplainer")
def test_linear_shap_corr(mock_explainer, dummy_data):
    mock_instance = MagicMock()
    mock_instance.shap_values = "ok"
    mock_explainer.return_value = mock_instance

    result = methods.linear_shap_corr("model", dummy_data)
    assert result == "ok"


@patch.object(methods.other, "CoefficentExplainer", create=True)
def test_coef(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions = "coef"
    mock_explainer.return_value = mock_instance

    result = methods.coef("model", None)
    assert result == "coef"


# ---------- lambda-return functions ----------

@patch("shap.benchmark.methods.kmeans")
@patch("shap.benchmark.methods.KernelExplainer")
def test_kernel_shap_lambda(mock_explainer, mock_kmeans, dummy_data, dummy_X):
    mock_kmeans.return_value = MagicMock(data=dummy_data)

    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = "val"
    mock_explainer.return_value = mock_instance

    f = methods.kernel_shap_1000_meanref(MagicMock(), dummy_data)
    assert callable(f)
    assert f(dummy_X) == "val"


# ---------- tree_shap_independent_200 ----------

@patch("shap.benchmark.methods.TreeExplainer")
def test_tree_shap_independent_subsample(mock_explainer, dummy_data):
    mock_instance = MagicMock()
    mock_instance.shap_values = "tree"
    mock_explainer.return_value = mock_instance

    result = methods.tree_shap_independent_200("model", dummy_data)
    mock_explainer.assert_called_once()
    assert result == "tree"


# ---------- mean_abs_tree_shap ----------

@patch("shap.benchmark.methods.TreeExplainer")
def test_mean_abs_tree_shap_array(mock_explainer, dummy_X):
    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = np.ones((10, 5))
    mock_explainer.return_value = mock_instance

    f = methods.mean_abs_tree_shap("model", None)
    out = f(dummy_X)

    assert out.shape == (10, 5)


@patch("shap.benchmark.methods.TreeExplainer")
def test_mean_abs_tree_shap_list(mock_explainer, dummy_X):
    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = [np.ones((10, 5))]
    mock_explainer.return_value = mock_instance

    f = methods.mean_abs_tree_shap("model", None)
    out = f(dummy_X)

    assert isinstance(out, list)
    assert len(out) == 1


# ---------- deep_shap ----------

@patch("shap.benchmark.methods.kmeans")
@patch("shap.benchmark.methods.DeepExplainer")
def test_deep_shap_single_output(mock_explainer, mock_kmeans, dummy_X):
    mock_kmeans.return_value = MagicMock(data=dummy_X)

    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = [np.ones((10, 5))]
    mock_explainer.return_value = mock_instance

    f = methods.deep_shap("model", np.ones((10, 5)))
    out = f(dummy_X)

    assert isinstance(out, np.ndarray)


@patch("shap.benchmark.methods.kmeans")
@patch("shap.benchmark.methods.DeepExplainer")
def test_deep_shap_multi_output(mock_explainer, mock_kmeans, dummy_X):
    mock_kmeans.return_value = MagicMock(data=dummy_X)

    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = [1, 2]
    mock_explainer.return_value = mock_instance

    f = methods.deep_shap("model", np.ones((10, 5)))
    out = f(dummy_X)

    assert isinstance(out, list)


# ---------- expected_gradients ----------

@patch("shap.benchmark.methods.GradientExplainer")
def test_expected_gradients(mock_explainer, dummy_X):
    mock_instance = MagicMock()
    mock_instance.shap_values.return_value = [np.ones((10, 5))]
    mock_explainer.return_value = mock_instance

    f = methods.expected_gradients("model", np.ones((10, 5)))
    out = f(dummy_X)

    assert isinstance(out, np.ndarray)


# ---------- KerasWrap handling ----------

def test_keras_wrap_unwrap():
    class DummyWrap:
        def __init__(self):
            self.model = "real_model"

    model = DummyWrap()

    with patch("shap.benchmark.methods.kmeans") as mock_kmeans, \
         patch("shap.benchmark.methods.DeepExplainer") as mock_explainer:

        mock_kmeans.return_value = MagicMock(data=np.ones((10, 5)))

        mock_instance = MagicMock()
        mock_instance.shap_values.return_value = [np.ones((10, 5))]
        mock_explainer.return_value = mock_instance

        f = methods.deep_shap(model, np.ones((10, 5)))
        assert callable(f)