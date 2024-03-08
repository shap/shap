"""This file contains tests for the `shap.datasets` module."""

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
