import numpy as np
import sklearn

sign_defaults = {
    "keep positive": 1,
    "keep negative": -1,
    "remove positive": -1,
    "remove negative": 1,
    "compute time": -1,
    "keep absolute": -1, # the absolute signs are defaults that make sense when scoring losses
    "remove absolute": 1,
    "explanation error": -1
}

class BenchmarkResult:
    """ The result of a benchmark run.
    """

    def __init__(self, metric, method, value=None, curve_x=None, curve_y=None, curve_y_std=None, value_sign=None):
        self.metric = metric
        self.method = method
        self.value = value
        self.curve_x = curve_x
        self.curve_y = curve_y
        self.curve_y_std = curve_y_std
        self.value_sign = value_sign
        if self.value_sign is None and self.metric in sign_defaults:
            self.value_sign = sign_defaults[self.metric]
        if self.value is None:
            self.value = sklearn.metrics.auc(curve_x, (np.array(curve_y) - curve_y[0]))

    @property
    def full_name(self):
        return self.method + " " + self.metric
