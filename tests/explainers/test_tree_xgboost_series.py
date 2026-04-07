import pandas as pd

from shap.explainers._tree import _xgboost_prepare_input


def test_xgboost_prepare_input_series():
    series = pd.Series([65, 150, 236], index=["Age", "RestingBP", "Cholesterol"], name=371)

    prepared = _xgboost_prepare_input(series)

    assert isinstance(prepared, pd.DataFrame)
    assert prepared.shape == (1, 3)
    assert list(prepared.columns) == ["Age", "RestingBP", "Cholesterol"]
    assert prepared.iloc[0].tolist() == [65, 150, 236]
