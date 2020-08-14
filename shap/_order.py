import copy
import numpy as np
import scipy as sp


class Order():
    """ An ordering object that defines an ordering when applied to any Explanation or array.
    """

    def __init__(self):
        """ Build a new order object without any ops.
        """
        self._ops = []
    
    def apply(self, values, axis, return_values=False):
        """ Return an ordering for the given dimension.
        """
        
        in_values_stack = True
        for o in self._ops:
            op_type,op,args,kwargs = o
            if op_type == "value":
                assert in_values_stack
                values = op(values, axis, *args, **kwargs)
            elif op_type == "value_to_index":
                in_values_stack = False
                inds = op(values, axis, *args, **kwargs)
            elif op_type == "index":
                if in_values_stack:
                    in_values_stack = False
                    inds = np.argsort(-values)
                inds = op(inds, axis, *args, **kwargs)
            else:
                assert False, "Invalid op_type!"
        
        if in_values_stack:
            inds = np.argsort(-values)
        
        if return_values:
            return inds, values
        else:
            return inds
    
    def __call__(self, *args, **kwargs):
        """ Update the args for the previous ordering operation.
        """
        new_self = copy.deepcopy(self)
        new_self._ops[-1][2] = args
        new_self._ops[-1][3] = kwargs
        return new_self
    
    def _other_axes(self, values, axis):
        other_axes = list(range(len(values.shape)))
        other_axes.remove(axis)
        return tuple(other_axes)
        
    def __getitem__(self, item):
        """ Return a new ordering object that returns indexes limited by the given slice operation.
        """
        new_self = copy.deepcopy(self)
        def __getitem__(inds, axis):
            return inds[item]
        new_self._ops.append(["index", __getitem__, [], {}])
        return new_self
    
    def _add_op(self, op, op_type):
        new_self = copy.deepcopy(self)
        new_self._ops.append([op_type, op, [], {}])
        return new_self

    @property
    def abs(self):
        def abs(values, axis):
            return np.abs(values)
        return self._add_op(abs, "value")
    
    @property
    def mean(self):
        def mean(values, axis):
            return np.mean(values, axis=self._other_axes(values, axis))
        return self._add_op(mean, "value")

    @property
    def sum(self):
        def sum(values, axis):
            return np.sum(values, axis=self._other_axes(values, axis))
        return self._add_op(sum, "value")

    @property
    def min(self):
        def min(values, axis):
            return np.min(values, axis=self._other_axes(values, axis))
        return self._add_op(min, "value")

    @property
    def max(self):
        def max(values, axis):
            return np.max(values, axis=self._other_axes(values, axis))
        return self._add_op(max, "value")

    @property
    def inv(self):
        def inv(values, axis):
            return -values
        return self._add_op(inv, "value")

    @property
    def percentile(self):
        def percentile(values, axis, q):
            return np.nanpercentile(values, q, axis=self._other_axes(values, axis))
        return self._add_op(percentile, "value")
    
    @property
    def hclust(self):
        """ Sorts by a hclustering.
        
        hclust(metric="sqeuclidean")
        
        Parameters
        ----------
        metric : string
            A metric supported by scipy clustering.
        """
        def hclust(values, axis, metric="sqeuclidean"):
            if len(values.shape) != 2:
                raise Exception("The hclust order only supports 2D arrays right now!")

            if axis == 1:
                values = values.T

            # compute a hierarchical clustering and return the optimal leaf ordering
            D = sp.spatial.distance.pdist(values, metric)
            cluster_matrix = sp.cluster.hierarchy.complete(D)
            inds = sp.cluster.hierarchy.leaves_list(sp.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, D))
            return inds
        return self._add_op(hclust, "value_to_index")
