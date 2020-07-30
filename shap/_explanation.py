
import pandas as pd
import numpy as np
import sys
import warnings
from slicer.interpretapi.explanation import AttributionExplanation
from slicer import Slicer

# slicer confuses pylint...
# pylint: disable=no-member


class Explanation(AttributionExplanation):
    """ This is currently an experimental feature don't depend on this object yet! :)
    """
    def __init__(
        self,
        expected_value,
        values,
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
        clustering = None
    ):
        
        input_shape = _compute_shape(data)
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
            kwargs_dict["hierarchical_values"] = (hierarchical_values, Slicer(hierarchical_values))
        if clustering is not None:
            self.clustering = clustering

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
    
    def get_expected_value(self):
        return self.base_value
    expected_value = property(get_expected_value)
        
    def __repr__(self):
        out  = ".expected_value =\n"+self.expected_value.__repr__()
        out += "\n\n.values =\n"+self.values.__repr__()
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
            if type(t) is str:
                if t.startswith("rank("):
                    t = "abs_rank(" + t[5:] # convert rank() to abs_rank()
                if t.startswith("abs_rank("):
                    rank = int(t[9:-1])
                    ranks = np.argsort(-np.sum(np.abs(self.values), tuple(j for j in range(len(self.values.shape)) if j != i)))
                    tmp = list(item)
                    tmp[i] = ranks[rank]
                    item = tuple(tmp)
                elif t.startswith("pos_rank("):
                    rank = int(t[9:-1])
                    ranks = np.argsort(-np.sum(self.values, tuple(j for j in range(len(self.values.shape)) if j != i)))
                    tmp = list(item)
                    tmp[i] = ranks[rank]
                    item = tuple(tmp)
                elif t.startswith("neg_rank("):
                    rank = int(t[9:-1])
                    ranks = np.argsort(np.sum(self.values, tuple(j for j in range(len(self.values.shape)) if j != i)))
                    tmp = list(item)
                    tmp[i] = ranks[rank]
                    item = tuple(tmp)
                else:
                    ind = np.where(np.array(self.feature_names) == t)[0][0]
                    tmp = list(item)
                    tmp[i] = int(ind)
                    item = tuple(tmp)
        
        out = super().__getitem__(item)
        if getattr(self, "clustering", None) is not None:
            out.clustering = self.clustering
        return out


def _compute_shape(x):
    if not hasattr(x, "__len__"):
        return tuple()
    else:
        if type(x) == list:
            return (len(x),) + _compute_shape(x[0])
        if type(x) == dict:
            return (len(x),) + _compute_shape(x[next(iter(x))])
        return x.shape