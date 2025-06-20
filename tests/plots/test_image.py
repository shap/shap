import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


def set_reproducible_mpl_rcparams() -> None:
    """Set some matplotlib rcParams to ensure consistency between versions.

    In matplotlib 3.10, the default value of "image.interpolation_stage" changed from "data" to "auto"
    which can lead to slighly different results.

    Careful: the @pytest.mark.mpl_image_compare decorator will override rcParams,
    so this change must be done *after* the fixtures are called.
    """
    plt.rcParams["image.interpolation"] = "bilinear"
    plt.rcParams["image.interpolation_stage"] = "data"


@pytest.fixture
def imagenet50_example() -> tuple[np.ndarray, np.ndarray]:
    # Return a subset of the imagenet50 dataset, normalised for plotting
    images, labels = shap.datasets.imagenet50()
    images = images / 255
    return images, labels


@pytest.fixture
def explanation_multi_example(imagenet50_example) -> shap.Explanation:
    # Return an explanation example
    images, _ = imagenet50_example
    n_images = 2
    n_classes = 4

    images = images[:n_images]
    shap_values_single = (images - images.mean()) / images.max(keepdims=True)
    assert shap_values_single.shape == images.shape

    # Just repeat the same SHAP values for each class
    shap_values_multi = np.stack([shap_values_single for _ in range(n_classes)], axis=-1)
    assert shap_values_multi.shape[-1] == n_classes

    explanation = shap.Explanation(values=shap_values_multi, data=images, output_names=[x for x in range(n_classes)])
    return explanation


@pytest.mark.mpl_image_compare
def test_image_single(imagenet50_example):
    set_reproducible_mpl_rcparams()
    images, _ = imagenet50_example
    images = images[0]
    shap_values = (images - images.mean()) / images.max(keepdims=True)
    explanation = shap.Explanation(values=shap_values, data=images)
    shap.image_plot(explanation, show=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_image_multi_no_labels(explanation_multi_example):
    """Multiple images, multiple classes, labels taken from explanation object"""
    set_reproducible_mpl_rcparams()
    shap.image_plot(explanation_multi_example, show=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_image_multi(explanation_multi_example):
    """Multiple images, multiple classes, a common set of labels for all rows"""
    set_reproducible_mpl_rcparams()
    *_, n_classes = explanation_multi_example.shape
    labels = [f"Class {x + 1}" for x in range(n_classes)]
    shap.image_plot(explanation_multi_example, labels=labels, show=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_image_multi_labels_per_row_list(explanation_multi_example):
    """Multiple images, multiple classes, a set of labels per row as list of lists"""
    set_reproducible_mpl_rcparams()
    n_images, *_, n_classes = explanation_multi_example.shape
    labels = [[f"Class {x + 1 + y * n_classes}" for x in range(n_classes)] for y in range(n_images)]
    shap.image_plot(explanation_multi_example, labels=labels, show=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_image_multi_labels_per_row_ndarray(explanation_multi_example):
    """Multiple images, multiple classes, a set of labels per row as np.array"""
    set_reproducible_mpl_rcparams()
    n_images, *_, n_classes = explanation_multi_example.shape
    labels = np.array([[f"Class {x + 1 + y * n_classes}" for x in range(n_classes)] for y in range(n_images)])
    shap.image_plot(explanation_multi_example, labels=labels, show=False)
    return plt.gcf()


def test_random_single_image():
    """Just make sure the image_plot function doesn't crash."""
    shap.image_plot(np.random.randn(3, 20, 20), np.random.randn(3, 20, 20), show=False)


def test_random_multi_image():
    """Just make sure the image_plot function doesn't crash."""
    shap.image_plot([np.random.randn(3, 20, 20) for i in range(3)], np.random.randn(3, 20, 20), show=False)


def test_image_to_text_single():
    """Just make sure the image_to_text function doesn't crash."""

    class MockImageExplanation:
        """Fake explanation object."""

        def __init__(self, data, values, output_names):
            self.data = data
            self.values = values
            self.output_names = output_names

    test_image_height = 500
    test_image_width = 500
    test_word_length = 4

    test_data = np.ones((test_image_height, test_image_width, 3)) * 50
    test_values = np.random.rand(test_image_height, test_image_width, 3, test_word_length)
    test_output_names = np.array([str(i) for i in range(test_word_length)])

    shap_values_test = MockImageExplanation(test_data, test_values, test_output_names)
    shap.plots.image_to_text(shap_values_test)
