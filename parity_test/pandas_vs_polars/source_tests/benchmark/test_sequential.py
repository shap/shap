import pandas as pd
import pytest

from shap.benchmark import SequentialMasker


def test_dataframe_argument_error_message():
    with pytest.raises(TypeError, match="DataFrame arguments don't iterate correctly"):
        SequentialMasker("keep", "positive", None, None, pd.DataFrame({"a": [1, 2]}))
