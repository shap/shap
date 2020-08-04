import copy
import numpy as np


class Order():
    """ The abstract parent class for all order objects.
    """

    def __init__(self):
        """ Build a new order object without any slice or reverse parameters.
        """

        self._slice_op = None
        self._reversed = False
    
    def __call__(self, values, axis):
        """ Return an ordering for the given dimension.
        """

        raise Exception("The call method must be overidden by a child class!")
        
    def __getitem__(self, item):
        """ Return a new ordering object that returns indexes limited by the given slice operation.
        """
        new_self = copy.copy(self)
        new_self._slice_op = item
        return new_self
    
    @property
    def reverse(self):
        """ Return a new ordering object that returns reversed indexes.
        """
        new_self = copy.copy(self)
        new_self._reversed = not self._reversed
        return new_self

    def _other_axes(self, values, axis):
        other_axes = list(range(len(values.shape)))
        other_axes.remove(axis)
        return tuple(other_axes)

    def _postprocess_inds(self, inds):
        if self._reversed:
            inds = np.flip(inds)
        if self._slice_op is None:
            return inds
        else:
            return inds[self._slice_op]