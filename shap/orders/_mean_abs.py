import numpy as np
from ._order import Order

class MeanAbs(Order):
    def __call__(self, values, axis):
        """ Return an ordering for the given dimension based on the mean absolute value over all other dimensions.
        """
        inds = np.argsort(-np.mean(np.abs(values), axis=self._other_axes(values, axis)))
        return self._postprocess_inds(inds)