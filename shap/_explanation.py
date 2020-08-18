
import pandas as pd
import numpy as np
import scipy as sp
import sys
import warnings
import copy
from slicer.interpretapi.explanation import AttributionExplanation
from slicer import Slicer
# from ._order import Order
from .utils._general import OpChain

# slicer confuses pylint...
# pylint: disable=no-member


op_chain_root = OpChain()
class MetaExplanation(type):
    """ This metaclass exposes the Explanation object's methods for creating template op chains.
    """

    def __getitem__(cls, item):
        return op_chain_root.__getitem__(item)

    @property
    def abs(cls):
        return op_chain_root.abs

    @property
    def argsort(cls):
        return op_chain_root.argsort

    @property
    def sum(cls):
        return op_chain_root.sum

    @property
    def max(cls):
        return op_chain_root.max

    @property
    def min(cls):
        return op_chain_root.min

    @property
    def mean(cls):
        return op_chain_root.mean
    
    @property
    def sample(cls):
        return op_chain_root.sample

    @property
    def hclust(cls):
        return op_chain_root.hclust


class Explanation(AttributionExplanation, metaclass=MetaExplanation):
    """ This is currently an experimental feature don't depend on this object yet! :)
    """
    def __init__(
        self,
        values,
        expected_value = None,
        data = None,
        output_shape = tuple(),
        interaction_order = 0,
        instance_names = None,
        input_names = None,
        output_names = None,
        output_indexes = None,
        feature_types = None,
        lower_bounds = None,
        upper_bounds = None,
        main_effects = None,
        hierarchical_values = None,
        original_rows = None,
        clustering = None
    ):
        self.transform_history = []

        if issubclass(type(values), Explanation):
            e = values
            values = e.values
            expected_value = e.expected_value
            data = e.data
            # TODO: better cloning :)

        if data is not None:
            input_shape = _compute_shape(data)
        else:
            input_shape = _compute_shape(values)

        # trim any trailing None shapes since we don't want slicer to try and use those
        if len(input_shape) > 0 and input_shape[-1] is None:
            input_shape = input_shape[:-1]
        
        values_dims = list(
            range(len(input_shape) + interaction_order + len(output_shape))
        )
        output_dims = range(len(input_shape) + interaction_order, values_dims[-1])
        
        #main_effects_inds = values_dims[0:len(input_shape)] + values_dims[len(input_shape) + interaction_order:]
        self.output_names = output_names # TODO: needs to tracked after slicing still
        
        kwargs_dict = {}
        if lower_bounds is not None:
            kwargs_dict["lower_bounds"] = (values_dims, Slicer(lower_bounds))
        if upper_bounds is not None:
            kwargs_dict["upper_bounds"] = (values_dims, Slicer(upper_bounds))
        if main_effects is not None:
            kwargs_dict["main_effects"] = (values_dims, Slicer(main_effects))
        if output_indexes is not None:
            kwargs_dict["output_indexes"] = (output_dims, Slicer(output_indexes))
        if output_names is not None:
            kwargs_dict["output_names"] = (output_dims, Slicer(output_names))
        if hierarchical_values is not None:
            kwargs_dict["hierarchical_values"] = (values_dims, Slicer(hierarchical_values))
        if input_names is not None:
            if not is_1d(input_names):
                input_name_dims = values_dims
            else:
                input_name_dims = values_dims[1:]
            kwargs_dict["input_names"] = (input_name_dims, Slicer(input_names))
        else:
            self.input_names = None
        if original_rows is not None:
            kwargs_dict["original_rows"] = (values_dims[1:], Slicer(original_rows))
        if clustering is not None:
            kwargs_dict["clustering"] = ([0], Slicer(clustering))
        if expected_value is not None:
            ndims = len(getattr(expected_value, "shape", []))
            if ndims == len(values_dims):
                kwargs_dict["expected_value"] = (values_dims, Slicer(expected_value))
            elif ndims == len(values_dims)-1:
                kwargs_dict["expected_value"] = (values_dims[1:], Slicer(expected_value))
            else:
                raise Exception("The shape of the passed expected_value does not match the shape of the passed values!")
        else:
            self.expected_value = None
        # if clustering is not None:
        #     self.clustering = clustering

        super().__init__(
            data,
            values,
            input_shape,
            output_shape,
            expected_value,
            interaction_order,
            instance_names,
            input_names,
            feature_types,
            **kwargs_dict
        )
        
    def get_shape(self):
        return _compute_shape(self.values)
    shape = property(get_shape)
    
    # def get_expected_value(self):
    #     return self.expected_value
    # expected_value = property(get_expected_value)
        
    def __repr__(self):
        out = ".values =\n"+self.values.__repr__()
        if self.expected_value is not None:
            out += "\n\n.expected_value =\n"+self.expected_value.__repr__()
        if self.data is not None:
            out += "\n\n.data =\n"+self.data.__repr__()
        return out
    
    def __getitem__(self, item):
        """ This adds support for magic string indexes like "rank(0)".
        """
        if not isinstance(item, tuple):
            item = (item,)
        
        # convert any magic strings
        for i,t in enumerate(item):
            if issubclass(type(t), OpChain):
                tmp = list(item)
                tmp[i] = t.apply(self)
                if issubclass(type(tmp[i]), (np.int64, np.int32)): # because slicer does not like numpy indexes
                    tmp[i] = int(tmp[i])
                elif issubclass(type(tmp[i]), np.ndarray):
                    tmp[i] = [int(v) for v in tmp[i]] # slicer wants lists not numpy arrays for indexing
                item = tuple(tmp)
            elif type(t) is str:
                if is_1d(self.input_names):
                    ind = np.where(np.array(self.input_names) == t)[0][0]
                    tmp = list(item)
                    tmp[i] = int(ind)
                    item = tuple(tmp)
                else:
                    new_values = []
                    new_data = []
                    for i in range(len(self.values)):
                        for s,v,d in zip(self.input_names[i], self.values[i], self.data[i]):
                            if s == t:
                                new_values.append(v)
                                new_data.append(d)
                    new_self = copy.deepcopy(self)
                    new_self.values = new_values
                    new_self.data = new_data
                    new_self.input_names = t
                    new_self.clustering = None
                    return new_self
            elif issubclass(type(t), np.ndarray):
                tmp = list(item)
                tmp[i] = [int(j) for j in t]
                item = tuple(tmp)
        
        out = super().__getitem__(item)
        # if getattr(self, "clustering", None) is not None:
        #     out.clustering = self.clustering
        out.transform_history = self.transform_history
        out.transform_history.append(("__getitem__", (item,)))
        return out

    def __len__(self):
        return self.shape[0]

    @property
    def abs(self):
        new_self = copy.deepcopy(self)
        new_self.values = np.abs(new_self.values)
        new_self.transform_history.append(("abs", None))
        return new_self

    def mean(self, dims):
        new_self = copy.deepcopy(self)

        if self.input_names is not None and not is_1d(self.input_names) and dims == 0:
            new_values = self._flatten_input_names()
            new_self.feature_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.mean(v) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = new_self.values.mean(dims)

        if dims == 0 and len(getattr(self, "clustering", np.zeros(0)).shape) == 3:
            if self.clustering.std(0).sum() < 1e-8:
                new_self.clustering = self.clustering[0]
            else:
                new_self.clustering = None

        new_self.data = None
        new_self.transform_history.append(("mean", (dims,)))
        
        return new_self

    @property
    def argsort(self):
        return np.argsort(-self.values)


    def hclust(self, metric="sqeuclidean", axis=0):
        """ Computes an optimal leaf ordering sort order using hclustering.
        
        hclust(metric="sqeuclidean")
        
        Parameters
        ----------
        metric : string
            A metric supported by scipy clustering.

        axis : int
            The axis to cluster along.
        """
        values = self.values

        if len(values.shape) != 2:
            raise Exception("The hclust order only supports 2D arrays right now!")

        if axis == 1:
            values = values.T

        # compute a hierarchical clustering and return the optimal leaf ordering
        D = sp.spatial.distance.pdist(values, metric)
        cluster_matrix = sp.cluster.hierarchy.complete(D)
        inds = sp.cluster.hierarchy.leaves_list(sp.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, D))
        return inds

    def sample(self, max_samples, replace=False, random_state=0):
        """ Randomly samples the instances (rows) of the Explanation object.

        Parameters
        ----------
        max_samples : int
            The number of rows to sample. Note that if replace=False then less than
            fewer than max_samples will be drawn if explanation.shape[0] < max_samples.
        
        replace : bool
            Sample with or without replacement.
        """
        prev_seed = np.random.seed(random_state)
        inds = np.random.choice(self.shape[0], min(max_samples, self.shape[0]), replace=replace)
        np.random.seed(prev_seed)
        return self[list(inds)]

    def _flatten_input_names(self):
        new_values = {}
        for i in range(len(self.values)):
            for s,v in zip(self.input_names[i], self.values[i]):
                if s not in new_values:
                    new_values[s] = []
                new_values[s].append(v)
        return new_values

    def _use_data_as_feature_names(self):
        new_values = {}
        for i in range(len(self.values)):
            for s,v in zip(self.data[i], self.values[i]):
                if s not in new_values:
                    new_values[s] = []
                new_values[s].append(v)
        return new_values

    def sum(self, dims):
        new_self = copy.deepcopy(self)
        if self.input_names is not None and not is_1d(self.input_names) and dims == 0:
            new_values = self._flatten_input_names()
            new_self.input_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.sum(v) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = new_self.values.sum(dims)
        new_self.data = None
        new_self.transform_history.append(("sum", (dims,)))
        return new_self

    def max(self, dims):
        new_self = copy.deepcopy(self)
        if self.input_names is not None and not is_1d(self.input_names) and dims == 0:
            new_values = self._flatten_input_names()
            new_self.input_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.max(v) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = new_self.values.max(dims)
        new_self.data = None
        new_self.transform_history.append(("max", (dims,)))
        return new_self

    def min(self, dims):
        new_self = copy.deepcopy(self)
        if self.input_names is not None and not is_1d(self.input_names) and dims == 0:
            new_values = self._flatten_input_names()
            new_self.input_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.min(v) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = new_self.values.min(dims)
        new_self.data = None
        new_self.transform_history.append(("min", (dims,)))
        return new_self

    def percentile(self, q, dims):
        new_self = copy.deepcopy(self)
        if self.input_names is not None and not is_1d(self.input_names) and dims == 0:
            new_values = self._flatten_input_names()
            new_self.input_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.percentile(v, q) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = np.percentile(new_self.values, q, dims)
        new_self.data = None
        new_self.transform_history.append(("percentile", (dims,)))
        return new_self

def is_1d(val):
    return not (issubclass(type(val[0]), list) or issubclass(type(val[0]), np.ndarray))

class Op():
    pass

class Percentile(Op):
    def __init__(self, percentile):
        self.percentile = percentile

    def add_repr(self, s, verbose=False):
        return "percentile("+s+", "+str(self.percentile)+")"

    


def _compute_shape(x):
    if not hasattr(x, "__len__"):
        return tuple()
    elif len(x) > 0 and type(x[0]) is str:
        return (None,)
    else:
        if type(x) == dict:
            return (len(x),) + _compute_shape(x[next(iter(x))])

        # 2D arrays we just take their shape as-is
        if len(getattr(x, "shape", tuple())) > 1:
            return x.shape

        # 1D arrays we need to look inside
        if len(x) == 0:
            return (0,)
        elif len(x) == 1:
            return (len(x),) + _compute_shape(x[0])
        else:
            first_shape = _compute_shape(x[0])
            for i in range(1,len(x)):
                shape = _compute_shape(x[i])
                if shape != first_shape:
                    return (len(x), None)
            return (len(x),) + first_shape