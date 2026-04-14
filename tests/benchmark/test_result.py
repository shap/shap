import numpy as np
from shap.benchmark._result import BenchmarkResult


def test_benchmark_result_initialization():
    """Test standard initialization and sign defaults."""
    # Test 1: Value and sign are explicitly provided
    res_explicit = BenchmarkResult("keep positive", "method_A", value=0.85, value_sign=-1)
    assert res_explicit.value == 0.85
    assert res_explicit.value_sign == -1
    assert res_explicit.full_name == "method_A keep positive"

    # Test 2: Sign is auto-populated from the sign_defaults dictionary
    res_default = BenchmarkResult("keep positive", "method_B", value=0.90)
    assert res_default.value == 0.90
    assert res_default.value_sign == 1  # 1 is the default for "keep positive"


def test_benchmark_result_auc_calculation():
    """Test that AUC is calculated properly if no value is explicitly passed."""
    curve_x = np.array([0.0, 1.0, 2.0])
    # The class subtracts curve_y[0] from all elements before calculating AUC.
    # curve_y becomes: [0, 5, 10]. 
    # Trapezoidal AUC for x=[0, 1, 2] and y=[0, 5, 10] is 10.0
    curve_y = np.array([10.0, 15.0, 20.0])
    
    res_auc = BenchmarkResult("unknown_metric", "method_C", curve_x=curve_x, curve_y=curve_y)
    
    assert res_auc.value == 10.0
    assert res_auc.value_sign is None  # "unknown_metric" is not in sign_defaults