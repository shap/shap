import numpy as np
import pandas as pd
import pytest

import shap
from shap.benchmark import BenchmarkResult, SequentialMasker


def _dummy_sum_model(x):
    return np.sum(x, axis=1)


def test_dataframe_argument_error_message():
    with pytest.raises(TypeError, match="DataFrame arguments don't iterate correctly"):
        SequentialMasker("keep", "positive", None, None, pd.DataFrame({"a": [1, 2]}))


def test_invalid_sort_order():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    masker = shap.maskers.Independent(X)
    with pytest.raises(ValueError, match="sort_order must be either"):
        SequentialMasker("keep", "random_order", masker, _dummy_sum_model, X)


def test_invalid_explanation_type():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, _dummy_sum_model, X)
    with pytest.raises(ValueError, match="must be either of type numpy.ndarray or shap.Explanation"):
        seq([[1.0, 2.0], [3.0, 4.0]], name="test")


def test_row_count_mismatch():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, _dummy_sum_model, X)
    attributions = np.zeros((2, 2))  # 2 rows instead of 3
    with pytest.raises(AssertionError, match="same number of rows as the model_args"):
        seq(attributions, name="test")


def test_sequential_masker_tabular_independent_keep_positive():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, _dummy_sum_model, X, batch_size=2)

    attributions = np.array([[0.1, 0.5, -0.2], [0.3, 0.2, 0.1], [-0.5, 0.2, 0.4], [0.1, -0.1, 0.2]])
    res = seq(attributions, name="linear_model", silent=True)

    assert isinstance(res, BenchmarkResult)
    assert res.metric == "keep positive"
    assert res.method == "linear_model"
    assert res.value_sign == 1
    assert len(res.curve_x) == 100
    assert len(res.curve_y) == 100
    assert isinstance(res.value, float)
    assert not np.isnan(res.value)


def test_sequential_masker_tabular_partition_remove_negative():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    masker = shap.maskers.Partition(X)
    seq = SequentialMasker("remove", "negative", masker, _dummy_sum_model, X, batch_size=2)

    attributions = np.array([[-0.2, 0.5, -0.1], [-0.3, -0.2, 0.1], [-0.5, 0.2, -0.4]])
    res = seq(attributions, name="partition_model", silent=True)

    assert isinstance(res, BenchmarkResult)
    assert res.metric == "remove negative"
    assert res.method == "partition_model"
    assert res.value_sign == 1
    assert isinstance(res.value, float)
    assert not np.isnan(res.value)


def test_sequential_masker_sort_order_absolute():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "absolute", masker, _dummy_sum_model, X, batch_size=2)

    attributions = np.array([[-0.5, 0.2], [0.1, -0.9]])
    res = seq(attributions, name="abs_model", silent=True)

    assert isinstance(res, BenchmarkResult)
    assert res.metric == "keep absolute"
    assert res.method == "abs_model"
    assert res.value_sign == -1
    assert isinstance(res.value, float)


def test_sequential_masker_with_explanation_object():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, _dummy_sum_model, X)

    attributions = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    explanation = shap.Explanation(values=attributions, data=X)

    res = seq(explanation, name="explanation_obj", silent=True)

    assert isinstance(res, BenchmarkResult)
    assert res.metric == "keep positive"
    assert res.method == "explanation_obj"


def test_sequential_masker_debug_mode():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, _dummy_sum_model, X)

    attributions = np.array([[0.5, 0.2], [0.1, 0.9]])
    mask_vals, curves, aucs = seq(attributions, name="debug_test", debug_mode=True, silent=True)

    assert len(mask_vals) == len(X)
    assert len(curves) == len(X)
    assert len(aucs) == len(X)
    for auc in aucs:
        assert isinstance(auc, float)


def test_sequential_masker_object_with_predict_method():
    class CustomEstimator:
        def predict(self, x):
            return np.sum(x, axis=1)

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = CustomEstimator()
    masker = shap.maskers.Independent(X)
    seq = SequentialMasker("keep", "positive", masker, model, X)

    attributions = np.array([[0.3, 0.7], [0.2, 0.8]])
    res = seq(attributions, name="custom_estimator", silent=True)

    assert isinstance(res, BenchmarkResult)
    assert res.method == "custom_estimator"
