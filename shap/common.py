import numpy as np

class shaparray(np.ndarray):
    def __new__(cls, expected_value, *args):
        self = np.ndarray.__new__(cls, *args)
        self.expected_value = expected_value
        return self

class shapinteractionarray(np.ndarray):
    def __new__(cls, expected_value, *args):
        self = np.ndarray.__new__(cls, *args)
        self.expected_value = expected_value
        return self
