import pytest
import numpy as np
import pandas as pd
import shap
from shap import Explanation
from shap.maskers import Independent
from shap.benchmark._result import BenchmarkResult

from shap.benchmark._sequential import SequentialMasker, SequentialPerturbation


@pytest.fixture
def dummy_data():
    """Provides basic dummy data for testing."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return X


@pytest.fixture
def dummy_model():
    """Provides a simple dummy model function."""
    def model(X):
        return np.sum(X, axis=1)
    return model


@pytest.fixture
def dummy_masker(dummy_data):
    """Provides a basic SHAP tabular masker."""
    return Independent(dummy_data)


@pytest.fixture
def dummy_attributions():
    """Provides dummy SHAP values."""
    return np.array([[0.5, -0.2, 0.1], [0.8, 0.0, -0.5]])


class TestSequentialMasker:
    def test_dataframe_argument_raises_error(self, dummy_model, dummy_masker):
        """Test that passing a pandas DataFrame raises a TypeError."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        
        with pytest.raises(TypeError, match="DataFrame arguments dont iterate correctly"):
            SequentialMasker("keep", "positive", dummy_masker, dummy_model, df)

    def test_initialization_success(self, dummy_model, dummy_masker, dummy_data):
        """Test successful initialization with valid numpy arrays."""
        masker = SequentialMasker("keep", "positive", dummy_masker, dummy_model, dummy_data)
        
        assert isinstance(masker.inner, SequentialPerturbation)
        assert masker.batch_size == 500
        assert len(masker.model_args) == 1
        assert np.array_equal(masker.model_args[0], dummy_data)


class TestSequentialPerturbation:
    def test_invalid_sort_order(self, dummy_model, dummy_masker):
        """Test that an invalid sort_order raises a ValueError."""
        with pytest.raises(ValueError, match='sort_order must be either "positive", "negative", or "absolute"!'):
            SequentialPerturbation(dummy_model, dummy_masker, "invalid_order", "keep")

    def test_invalid_explanation_type(self, dummy_model, dummy_masker, dummy_data):
        """Test that passing a list instead of ndarray/Explanation raises an error."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "positive", "keep")
        invalid_explanation = [[0.1, 0.2, 0.3]]
        
        with pytest.raises(ValueError, match="The passed explanation must be either of type numpy.ndarray or shap.Explanation!"):
            perturbation("test_name", invalid_explanation, dummy_data)

    def test_explanation_length_mismatch(self, dummy_model, dummy_masker, dummy_data):
        """Test that explanation length matching model_args length is enforced."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "positive", "keep")
        mismatched_attributions = np.array([[0.1, 0.2, 0.3]])
        
        with pytest.raises(AssertionError, match="The explanation passed must have the same number of rows"):
            perturbation("test_name", mismatched_attributions, dummy_data)

    def test_call_debug_mode_with_ndarray(self, dummy_model, dummy_masker, dummy_data, dummy_attributions):
        """Test a full run in debug_mode returning raw metrics using an ndarray explanation."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "positive", "keep")
        
        mask_vals, curves, aucs = perturbation(
            "test_run",
            dummy_attributions,
            dummy_data,  
            percent=0.5,
            debug_mode=True,
            silent=True
        )
        
        assert isinstance(mask_vals, list)
        assert len(mask_vals) == len(dummy_data)
        assert isinstance(curves, np.ndarray)
        assert curves.shape == (2, 100)
        assert len(aucs) == 2

    def test_call_standard_mode_with_shap_explanation(self, dummy_model, dummy_masker, dummy_data, dummy_attributions):
        """Test a full run returning a BenchmarkResult using a shap.Explanation object."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "absolute", "remove")
        explanation_obj = Explanation(values=dummy_attributions, data=dummy_data)
        
        result = perturbation(
            "test_explanation_run",
            explanation_obj,
            dummy_data,
            percent=0.2,
            debug_mode=False,
            silent=True
        )

        assert isinstance(result, BenchmarkResult)
        assert len(result.curve_x) == 100
        assert len(result.curve_y) == 100
        assert len(result.curve_y_std) == 100

    @pytest.mark.parametrize("sort_order", ["positive", "negative", "absolute"])
    @pytest.mark.parametrize("perturbation_type", ["keep", "remove"])
    def test_sort_orders_and_perturbations(self, dummy_model, dummy_masker, dummy_data, dummy_attributions, sort_order, perturbation_type):
        """Parameterized test to ensure all combinations of sort_order and perturbation run without crashing."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, sort_order, perturbation_type)
        
        mask_vals, curves, aucs = perturbation(
            f"test_{sort_order}_{perturbation_type}",
            dummy_attributions,
            dummy_data,
            percent=0.1,
            debug_mode=True,
            silent=True
        )
        
        assert len(aucs) == len(dummy_data)

class TestSequentialPerturbationLegacyMethods:

    def test_score_standard_mode(self, dummy_model, dummy_masker, dummy_data, dummy_attributions):
        """Test the legacy score method returns xs, ys, and auc."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "positive", "keep")
        perturbation.f = lambda masked, x, index: np.sum(masked[0], axis=1) if len(masked[0].shape)>1 else np.array([np.sum(masked[0])])
        
        xs, ys, auc = perturbation.score(
            explanation=dummy_attributions, 
            X=dummy_data, 
            percent=0.5, 
            silent=True
        )
        
        assert len(xs) == 100
        assert len(ys) == 100
        assert isinstance(auc, float)
        assert len(perturbation.score_values) == 1

    def test_score_debug_mode(self, dummy_model, dummy_masker, dummy_data, dummy_attributions):
        """Test the legacy score method in debug mode returns raw structures."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "absolute", "remove")
        perturbation.f = lambda masked, x, index: np.sum(masked[0], axis=1) if len(masked[0].shape)>1 else np.array([np.sum(masked[0])])
        
        mask_vals, curves, aucs = perturbation.score(
            explanation=dummy_attributions, 
            X=dummy_data, 
            percent=0.5, 
            silent=True,
            debug_mode=True
        )
        
        assert isinstance(mask_vals, list)
        assert isinstance(curves, np.ndarray)
        assert curves.shape[1] == 100
        assert len(aucs) == len(dummy_data)

    def test_score_dataframe_conversion(self, dummy_model, dummy_masker, dummy_attributions):
        """Test that passing a pandas DataFrame correctly converts to numpy arrays."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "negative", "keep")
        perturbation.f = lambda masked, x, index: np.sum(masked[0], axis=1) if len(masked[0].shape)>1 else np.array([np.sum(masked[0])])
        
        df_data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=['f1', 'f2', 'f3'])
        xs, ys, auc = perturbation.score(
            explanation=dummy_attributions, 
            X=df_data, 
            silent=True
        )
        
        assert len(xs) == 100

    def test_plot(self, dummy_model, dummy_masker, monkeypatch):
        """Test the plot method triggers matplotlib calls using pytest's monkeypatch."""
        perturbation = SequentialPerturbation(dummy_model, dummy_masker, "positive", "keep")
        plot_called_with = {}
        show_called = False

        def mock_plot(x, y, label):
            plot_called_with['x'] = x
            plot_called_with['y'] = y
            plot_called_with['label'] = label

        def mock_show():
            nonlocal show_called
            show_called = True

        monkeypatch.setattr("matplotlib.pyplot.plot", mock_plot)
        monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

        xs = np.linspace(0, 1, 100)
        ys = np.random.rand(100)
        auc = 0.95
        
        perturbation.plot(xs, ys, auc)
        
        assert np.array_equal(plot_called_with['x'], xs)
        assert np.array_equal(plot_called_with['y'], ys)
        assert plot_called_with['label'] == "AUC 0.9500"
        assert show_called is True