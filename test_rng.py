import warnings

import numpy as np

import shap


def run_legacy():
    np.random.seed(0)
    X = np.random.randn(20, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shap.summary_plot(X, show=False)
        print(f"Legacy triggered {len(w)} warnings.")


def run_modern():
    rs = np.random.RandomState(0)
    X = rs.randn(20, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shap.summary_plot(X, show=False, rng=rs)
        print(f"Modern triggered {len(w)} warnings.")


if __name__ == "__main__":
    run_legacy()
    run_modern()
