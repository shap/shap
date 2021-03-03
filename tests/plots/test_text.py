import numpy as np
import shap

def test_single_text_to_text():
    """ Just make sure the test_plot function doesn't crash.
    """

    class MockTextExplanation: # pylint: disable=too-few-public-methods
        """ Fake explanation object.
        """
        def __init__(self, data, values, output_names, base_values, clustering, hierarchical_values):
            self.data = data
            self.values = values
            self.output_names = output_names
            self.base_values = base_values
            self.clustering = clustering
            self.hierarchical_values = hierarchical_values
            self.shape = (values.shape[0], values.shape[1])


    test_values = np.array([
        [10.61284012, 3.28389317],
        [-3.77245945, 10.76889759],
        [0., 0.]
    ])

    test_base_values = np.array([-6.12535715, -12.87049389])

    test_data = np.array(['▁Hello ', '▁world ', ' '], dtype='<U7')

    test_output_names = np.array(['▁Hola', '▁mundo'], dtype='<U6')

    test_clustering = np.array([
        [0., 1., 12., 2.],
        [3., 2., 13., 3.]
    ])

    test_hierarchical_values = np.array([
        [13.91739416, 7.09603131],
        [-0.4679054, 14.58103573],
        [0., 0.],
        [-6.60910809, -7.62427628],
        [0., 0.]
    ])


    shap_values_test = MockTextExplanation(test_data, test_values, test_output_names, test_base_values, test_clustering, test_hierarchical_values)
    shap.plots.text(shap_values_test)
