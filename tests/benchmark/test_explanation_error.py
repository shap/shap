import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from shap import Explanation
from shap.benchmark import ExplanationError
from shap.maskers import FixedComposite, Image, Text
from shap.utils._exceptions import DimensionError


def test_explanation_error_init_data_types():
    """Test that the init function correctly identifies tabular, image, and text maskers."""
    # Tabular (fallback default)
    ee_tab = ExplanationError(MagicMock(), MagicMock())
    assert ee_tab.data_type == "tabular"

    # Image
    mock_image_masker = MagicMock()
    mock_image_masker.__class__ = Image
    ee_img = ExplanationError(mock_image_masker, MagicMock())
    assert ee_img.data_type == "image"

    # Text inside FixedComposite
    mock_text_masker = MagicMock()
    mock_text_masker.__class__ = Text
    mock_composite = MagicMock(spec=FixedComposite)
    mock_composite.masker = mock_text_masker
    ee_txt = ExplanationError(mock_composite, MagicMock())
    assert ee_txt.data_type == "text"


@patch("shap.benchmark._explanation_error.MaskedModel")
def test_explanation_error_call_exceptions(mock_masked_model):
    """Test that bad inputs properly raise ValueErrors and DimensionErrors."""
    masker = MagicMock()
    masker.clustering = None  # Prevent Numba compilation crashes
    model = MagicMock()
    model_args = (np.array([[1.0, 2.0]]),)
    ee = ExplanationError(masker, model, *model_args)

    # 1. Invalid explanation type
    with pytest.raises(ValueError, match="must be either of type numpy.ndarray or shap.Explanation"):
        ee([1.0, 2.0], "test")

    # 2. Row dimension mismatch
    with pytest.raises(DimensionError, match="same number of rows as the self.model_args"):
        ee(np.array([[1.0, 2.0], [3.0, 4.0]]), "test")

    # 3. Shape dimension mismatch (Force a 3D array to mismatch the 1D model_args row)
    with pytest.raises(ValueError, match="same dim as the model_args"):
        ee(np.array([[[1.0]]]), "test")


@patch("shap.benchmark._explanation_error.MaskedModel")
@patch("shap.benchmark._explanation_error.partition_tree_shuffle")
def test_explanation_error_clustering(mock_shuffle, mock_masked_model):
    """Test the different row_clustering branches inside the loop."""
    model = MagicMock()
    model_args = (np.array([[1.0, 2.0]]),)
    explanation = np.array([[0.5, 0.5]])

    # Mock MaskedModel to return an array of ones so the math doesn't crash
    mock_model_instance = MagicMock()
    mock_model_instance.side_effect = lambda masks: np.ones(len(masks))
    mock_masked_model.return_value = mock_model_instance

    # 1. Array Clustering
    masker_arr = MagicMock()
    masker_arr.clustering = np.array([[1, 2, 3]])
    ee_arr = ExplanationError(masker_arr, model, *model_args, num_permutations=1)
    ee_arr(explanation, "test")
    mock_shuffle.assert_called()

    # 2. Callable Clustering
    masker_call = MagicMock()
    masker_call.clustering = lambda x: np.array([[1, 2, 3]])
    ee_call = ExplanationError(masker_call, model, *model_args, num_permutations=1)
    ee_call(explanation, "test")  # Should execute smoothly

    # 3. Invalid Clustering raises NotImplementedError
    masker_invalid = MagicMock()
    masker_invalid.clustering = "invalid_string"
    ee_invalid = ExplanationError(masker_invalid, model, *model_args, num_permutations=1)
    with pytest.raises(NotImplementedError, match="not yet supported"):
        ee_invalid(explanation, "test")


@patch("time.time")
@patch("shap.benchmark._explanation_error.tqdm")
@patch("shap.benchmark._explanation_error.MaskedModel")
def test_explanation_error_tqdm_and_success(mock_masked_model, mock_tqdm, mock_time):
    """Test a full successful run using an Explanation object and trigger tqdm."""
    # Setup the math mock
    mock_model_instance = MagicMock()
    mock_model_instance.side_effect = lambda masks: np.ones(len(masks))
    mock_masked_model.return_value = mock_model_instance

    # Provide multiple rows to trigger the loop
    model_args = (np.array([[1.0], [2.0]]),)
    exp_obj = Explanation(values=np.array([[0.5], [0.5]]))
    
    masker = MagicMock()
    masker.clustering = None
    ee = ExplanationError(masker, MagicMock(), *model_args, num_permutations=1)
    
    # Mock time.time() to jump forward 10 seconds to force the progress bar to render
    mock_time.side_effect = [0, 10, 10, 10]
    mock_pbar = MagicMock()
    mock_tqdm.return_value = mock_pbar
    
    result = ee(exp_obj, "test_tqdm")
    
    # Verify standard results
    assert result.metric == "explanation error"
    assert result.method == "test_tqdm"
    
    # Verify tqdm was initialized, updated, and closed
    mock_tqdm.assert_called_once()
    assert mock_pbar.update.call_count >= 1
    mock_pbar.close.assert_called_once()