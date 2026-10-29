"""This file contains tests for the `shap.datasets` module."""

import os

import numpy as np
import pandas as pd
import pytest

import shap


@pytest.mark.parametrize("n_points", [None, 12])
def test_imagenet50(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.imagenet50(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 50 if n_points is None else n_points
    assert X.shape == (n_points, 224, 224, 3)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_california(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.california(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 20_640 if n_points is None else n_points
    assert X.shape == (n_points, 8)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_linnerud(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.linnerud(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 20 if n_points is None else n_points
    assert X.shape == (n_points, 3)
    assert y.shape == (n_points, 3)


@pytest.mark.parametrize("n_points", [None, 12])
def test_imdb(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.imdb(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 25_000 if n_points is None else n_points
    assert len(X) == n_points
    assert len(y) == n_points


@pytest.mark.parametrize("n_points", [None, 12])
def test_diabetes(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.diabetes(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 442 if n_points is None else n_points
    assert X.shape == (n_points, 10)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_iris(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.iris(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 150 if n_points is None else n_points
    assert X.shape == (n_points, 4)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_adult(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.adult(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 32_561 if n_points is None else n_points
    assert X.shape == (n_points, 12)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_nhanesi(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.nhanesi(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 14_264 if n_points is None else n_points
    assert X.shape == (n_points, 79)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [100, 2_000])
def test_corrgroups60(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.corrgroups60(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    assert X.shape == (n_points, 60)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [100, 2_000])
def test_independentlinear60(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.independentlinear60(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    assert X.shape == (n_points, 60)
    assert y.shape == (n_points,)


@pytest.mark.parametrize("n_points", [None, 12])
def test_a1a(n_points):
    # test that fetch/download works fine
    X, y = shap.datasets.a1a(n_points=n_points)

    # check the shape of the result
    # check that the n_points parameter samples the dataset
    n_points = 1_605 if n_points is None else n_points
    assert X.shape == (n_points, 119)
    assert y.shape == (n_points,)


def test_rank():
    # test that fetch/download works fine
    X1, y1, X2, y2, q1, q2 = shap.datasets.rank()

    # check the shape of the result
    assert X1.shape == (3_005, 300)
    assert y1.shape == (3_005,)
    assert X2.shape == (768, 300)
    assert y2.shape == (768,)
    assert q1.shape == (201,)
    assert q2.shape == (50,)


# ---------------------------------------------------------------------------
# communitiesandcrime — previously untested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [None, 15])
def test_communitiesandcrime(n_points):
    X, y = shap.datasets.communitiesandcrime(n_points=n_points)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    # rows of X and y must align
    assert X.shape[0] == y.shape[0]
    if n_points is not None:
        assert X.shape[0] == n_points
    else:
        assert X.shape[0] > 0
    # target contains finite floats (violent crimes per 100K)
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# display=True paths for iris / adult / nhanesi
# ---------------------------------------------------------------------------


def test_iris_display_true_returns_string_labels():
    X, y = shap.datasets.iris(display=True)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, list)
    assert len(y) == 150
    assert all(isinstance(label, str) for label in y)
    assert set(y) == {"setosa", "versicolor", "virginica"}


def test_iris_display_true_n_points():
    X, y = shap.datasets.iris(display=True, n_points=10)

    assert isinstance(y, list)
    assert len(y) == 10
    assert X.shape[0] == 10


def test_iris_display_false_returns_integer_labels():
    X, y = shap.datasets.iris(display=False)

    assert isinstance(y, np.ndarray)
    assert y.dtype.kind in ("i", "u")  # integer dtype


def test_adult_display_true_drops_education_target_fnlwgt():
    X, y = shap.datasets.adult(display=True, n_points=50)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert "Education" not in X.columns
    assert "Target" not in X.columns
    assert "fnlwgt" not in X.columns
    # raw display keeps the Age column
    assert "Age" in X.columns
    assert X.shape[0] == 50


def test_adult_display_true_keeps_categorical_dtypes():
    # display=True returns raw data with categorical columns intact
    X_display, _ = shap.datasets.adult(display=True, n_points=30)
    X_model, _ = shap.datasets.adult(display=False, n_points=30)

    # display=True keeps Workclass as categorical; display=False encodes it as integers
    assert X_display["Workclass"].dtype.name == "category"
    assert X_model["Workclass"].dtype.kind in ("i", "u", "f")


def test_nhanesi_display_true_shape():
    X, y = shap.datasets.nhanesi(display=True, n_points=50)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 50
    assert y.shape[0] == 50


def test_nhanesi_display_true_same_columns_as_false():
    # The sex-column modification in display=True is currently commented out,
    # so columns should be identical between display modes.
    X_display, _ = shap.datasets.nhanesi(display=True, n_points=20)
    X_model, _ = shap.datasets.nhanesi(display=False, n_points=20)

    assert list(X_display.columns) == list(X_model.columns)


# ---------------------------------------------------------------------------
# Return-type and column assertions for tabular datasets
# ---------------------------------------------------------------------------


def test_california_returns_dataframe_with_expected_columns():
    X, y = shap.datasets.california(n_points=10)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    expected_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    assert list(X.columns) == expected_cols


def test_linnerud_returns_dataframes_with_expected_columns():
    X, y = shap.datasets.linnerud()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert list(X.columns) == ["Chins", "Situps", "Jumps"]
    assert list(y.columns) == ["Weight", "Waist", "Pulse"]


def test_independentlinear60_returns_sixty_columns():
    X, y = shap.datasets.independentlinear60(n_points=50)

    assert isinstance(X, pd.DataFrame)
    assert X.shape[1] == 60
    assert isinstance(y, np.ndarray)


# ---------------------------------------------------------------------------
# cache helper
# ---------------------------------------------------------------------------


def test_cache_returns_path_to_existing_file():
    # adult.data is already downloaded by test_adult above
    from shap.datasets import cache

    url = "https://github.com/shap/shap/raw/master/data/adult.data"
    path = cache(url)

    assert isinstance(path, str)
    assert os.path.isfile(path)
    assert path.endswith("adult.data")


def test_cache_custom_file_name(tmp_path, monkeypatch):
    # Redirect cached_data dir to tmp_path so we don't pollute the repo
    monkeypatch.setattr(
        "shap.datasets.cache",
        lambda url, file_name=None: str(tmp_path / (file_name or os.path.basename(url))),
    )
    from shap.datasets import cache

    result = cache("https://example.com/data.csv", file_name="renamed.csv")
    # monkeypatched version just returns the path string
    assert result.endswith("renamed.csv")
