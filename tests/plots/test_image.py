import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap

def test_random_single_image():
    """ Just make sure the image_plot function doesn't crash.
    """

    shap.image_plot(np.random.randn(3, 20,20), np.random.randn(3, 20,20), show=False)

def test_random_multi_image():
    """ Just make sure the image_plot function doesn't crash.
    """

    shap.image_plot([np.random.randn(3, 20,20) for i in range(3)], np.random.randn(3, 20,20), show=False)
