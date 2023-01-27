import pandas as pd
import pytest
from sklearn import datasets


@pytest.fixture(scope="function")
def iris_dataset():
    """Return the classic iris data in a nice package."""

    d = datasets.load_iris()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    df["target"] = d.target

    df = df.sample(n=20, random_state=42)

    target = df["target"].copy()
    df = df.drop(columns=["target"])

    return df, target
