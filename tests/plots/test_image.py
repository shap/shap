import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap # pylint: disable=wrong-import-position

def test_random_single_image():
    """ Just make sure the image_plot function doesn't crash.
    """

    shap.image_plot(np.random.randn(3, 20, 20), np.random.randn(3, 20, 20), show=False)

def test_random_multi_image():
    """ Just make sure the image_plot function doesn't crash.
    """

    shap.image_plot([np.random.randn(3, 20, 20) for i in range(3)], np.random.randn(3, 20, 20), show=False)

def test_image_to_text_single():
    """ Just make sure the image_to_text function doesn't crash.
    """

    class MockImageExplanation: # pylint: disable=too-few-public-methods
        """ Fake explanation object.
        """
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
