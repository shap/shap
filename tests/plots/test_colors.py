import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import param

from shap.plots import colors


def test_colors():
    assert_allclose(colors.blue_rgb, np.array([0.0, 0.54337757, 0.98337906]))
    assert_allclose(colors.gray_rgb, np.array([0.51615537, 0.51615111, 0.5161729]))
    assert_allclose(colors.light_blue_rgb, np.array([0.49803922, 0.76862745, 0.98823529]))
    assert_allclose(colors.light_red_rgb, np.array([1.0, 0.49803922, 0.65490196]))
    assert_allclose(colors.red_rgb, np.array([1.0, 0.0, 0.31796406]))


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(
    "cmap",
    [
        param(colors.red_blue, id="red_blue"),
        param(colors.red_blue_circle, id="red_blue_circle"),
        param(colors.red_blue_no_bounds, id="red_blue_no_bounds"),
        param(colors.red_blue_transparent, id="red_blue_transparent"),
        param(colors.red_transparent_blue, id="red_transparent_blue"),
        param(colors.red_white_blue, id="red_white_blue"),
        param(colors.transparent_blue, id="transparent_blue"),
        param(colors.transparent_red, id="transparent_red"),
    ],
)
def test_colormaps(cmap):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    fig = plt.figure(figsize=(6, 0.6))
    plt.imshow(gradient, aspect="auto", cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    return fig
