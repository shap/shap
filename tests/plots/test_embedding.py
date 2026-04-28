import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture
def embedding_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    shap_values = np.array(
        [
            [0.3, -0.2, 0.1],
            [0.1, 0.4, -0.3],
            [-0.5, 0.2, 0.2],
            [0.6, -0.1, -0.4],
            [-0.2, -0.5, 0.3],
            [0.4, 0.1, -0.2],
        ]
    )
    embedding_values = np.array(
        [
            [-1.0, -0.6],
            [-0.4, 0.8],
            [0.1, -0.9],
            [0.5, 0.5],
            [0.9, -0.2],
            [1.2, 0.7],
        ]
    )
    feature_names = ["Feature A", "Feature B", "Feature C"]
    return shap_values, embedding_values, feature_names


def test_embedding_returns_axes(embedding_data):
    shap_values, embedding_values, feature_names = embedding_data

    returned_ax = shap.plots.embedding(
        0,
        shap_values,
        feature_names=feature_names,
        method=embedding_values,
        show=False,
    )

    assert isinstance(returned_ax, plt.Axes)


def test_embedding_returns_user_axes(embedding_data):
    shap_values, embedding_values, feature_names = embedding_data
    _, ax = plt.subplots()

    returned_ax = shap.plots.embedding(
        0,
        shap_values,
        feature_names=feature_names,
        method=embedding_values,
        show=False,
        ax=ax,
    )

    assert returned_ax is ax


def test_embedding_supports_pca_method(embedding_data):
    shap_values, _, feature_names = embedding_data

    returned_ax = shap.plots.embedding(
        "sum()",
        shap_values,
        feature_names=feature_names,
        method="pca",
        show=False,
    )

    assert isinstance(returned_ax, plt.Axes)


def test_embedding_raises_for_unsupported_method(embedding_data):
    shap_values, _, feature_names = embedding_data

    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(
            0,
            shap_values,
            feature_names=feature_names,
            method="tsne",
            show=False,
        )


@pytest.mark.mpl_image_compare
def test_embedding_custom_ax(embedding_data):
    shap_values, embedding_values, feature_names = embedding_data
    fig, ax = plt.subplots()

    shap.plots.embedding(
        0,
        shap_values,
        feature_names=feature_names,
        method=embedding_values,
        show=False,
        ax=ax,
    )
    plt.tight_layout()
    return fig
