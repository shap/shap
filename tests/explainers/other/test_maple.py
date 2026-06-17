import numpy as np

from shap.explainers.other._maple import MAPLE


def test_maple_small():
    """Test a small MAPLE model."""
    rs = np.random.RandomState(0)
    X_train = rs.randn(10, 4)
    y_train = X_train[:, 0] * 2 + rs.randn(10) * 0.1
    X_val = rs.randn(5, 4)
    y_val = X_val[:, 0] * 2 + rs.randn(5) * 0.1

    maple = MAPLE(X_train, y_train, X_val, y_val)

    preds = maple.predict(X_val)
    assert preds.shape == (5,)
