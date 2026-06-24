import os

import pytest

# PR #4951: Modernize tutorial notebooks (Execution Test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "notebook_path",
    [
        "notebooks/api_examples/plots/heatmap.ipynb",
        "notebooks/api_examples/plots/violin.ipynb",
        "notebooks/tabular_examples/model_agnostic/Simple Kernel SHAP.ipynb",
        "notebooks/tabular_examples/tree_based_models/Catboost tutorial.ipynb",
        "notebooks/tabular_examples/tree_based_models/Census income classification with LightGBM.ipynb",
        "notebooks/tabular_examples/tree_based_models/Census income classification with XGBoost.ipynb",
    ],
)
def test_modernized_notebook_execution(notebook_path):
    """
    Execute the modernized notebooks to ensure the new Explanation API
    usage doesn't cause runtime errors.
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        pytest.skip("nbformat or nbconvert not installed")

    # Path to tutorials
    if not os.path.exists(notebook_path):
        pytest.skip(f"Notebook not found at {notebook_path}")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed execution: {str(e)}")
