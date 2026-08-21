import numpy as np

from shap.links import identity, logit


class LinkSuite:
    params = ([1, 2], [1, 100, 10_000])
    param_names = ["ndim", "element_count"]

    def setup(self, ndim, element_count):
        if ndim == 1:
            shape = (element_count,)
        else:
            side_length = int(np.sqrt(element_count))
            shape = (side_length, side_length)

        self.probabilities = np.linspace(0.01, 0.99, element_count, dtype=np.float64).reshape(shape)
        self.log_odds = np.linspace(-5.0, 5.0, element_count, dtype=np.float64).reshape(shape)

        # Dry runs exclude Numba compilation time when comparing against older revisions.
        _ = identity(self.probabilities)
        _ = identity.inverse(self.probabilities)
        _ = logit(self.probabilities)
        _ = logit.inverse(self.log_odds)

    def time_identity(self, ndim, element_count):
        _ = identity(self.probabilities)

    def time_identity_inverse(self, ndim, element_count):
        _ = identity.inverse(self.probabilities)

    def time_logit(self, ndim, element_count):
        _ = logit(self.probabilities)

    def time_logit_inverse(self, ndim, element_count):
        _ = logit.inverse(self.log_odds)
