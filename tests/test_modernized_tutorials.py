import os
import pytest

# PR #4951: Modernize tutorial notebooks (Execution Test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "notebook_name",
    [
        "census_income_xgboost.ipynb",
        "linear_regression_xgboost.ipynb",
        "front_page_xgboost.ipynb",
        "catboost.ipynb",
        "lightgbm.ipynb",
        "sentiment_analysis_lightgbm.ipynb",
    ],
)
def test_modernized_notebook_execution(notebook_name):
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
    # Note: Adjusting path relative to repo root
    nb_path = os.path.join("notebooks", "tabular_examples", "model_agnostic", notebook_name)
    if not os.path.exists(nb_path):
        # Fallback for different repo structures
        nb_path = os.path.join("notebooks", notebook_name)

    if not os.path.exists(nb_path):
        pytest.skip(f"Notebook {notebook_name} not found at {nb_path}")

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nb_path)}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_name} failed execution: {str(e)}")
