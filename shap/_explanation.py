
import pandas as pd
import numpy as np
import scipy as sp
import sys
import warnings
import copy
import operator
import sklearn
from slicer import Slicer, Alias, Obj
# from ._order import Order
from .utils._general import OpChain

# slicer confuses pylint...
# pylint: disable=no-member


op_chain_root = OpChain("shap.Explanation")
class MetaExplanation(type):
    """ This metaclass exposes the Explanation object's methods for creating template op chains.
    """

    def __getitem__(cls, item):
        return op_chain_root.__getitem__(item)

    @property
    def abs(cls):
        return op_chain_root.abs

    @property
    def identity(cls):
        return op_chain_root.identity

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


class Explanation(object, metaclass=MetaExplanation):
    """ A slicable set of parallel arrays representing a SHAP explanation.
    """
    def __init__(
        self,
        values,
        base_values=None,
        data=None,
        display_data=None,
        instance_names=None,
        feature_names=None,
        output_names=None,
        output_indexes=None,
        lower_bounds=None,
        upper_bounds=None,
        main_effects=None,
        hierarchical_values=None,
        clustering=None
    ):
        self.op_history = []

        # cloning. TODO: better cloning :)
        if issubclass(type(values), Explanation):
            e = values
            values = e.values
            base_values = e.base_values
            data = e.data

        output_dims = compute_output_dims(values, base_values, data)

        if len(_compute_shape(feature_names)) == 1: # TODO: should always be an alias once slicer supports per-row aliases
            values_shape = _compute_shape(values)
            if len(values_shape) >= 1 and len(feature_names) == values_shape[0]:
                feature_names = Alias(list(feature_names), 0)
            elif len(values_shape) >= 2 and len(feature_names) == values_shape[1]:
                feature_names = Alias(list(feature_names), 1)

        if len(_compute_shape(output_names)) == 1: # TODO: should always be an alias once slicer supports per-row aliases
            values_shape = _compute_shape(values)
            output_names = Alias(list(output_names), output_dims[0])
            # if len(values_shape) >= 1 and len(output_names) == values_shape[0]:
            #     output_names = Alias(list(output_names), 0)
            # elif len(values_shape) >= 2 and len(output_names) == values_shape[1]:
            #     output_names = Alias(list(output_names), 1)

        if output_names is not None and not isinstance(output_names, Alias):
            l = len(_compute_shape(output_names))
            if l == 0:
                pass
            elif l == 1:
                output_names = Obj(output_names, output_dims)
            elif l == 2:
                output_names = Obj(output_names, [0] + list(output_dims))
            else:
                raise ValueError("shap.Explanation does not yet support output_names of order greater than 3!")

        self._s = Slicer(
            values=values,
            base_values=None if base_values is None else Obj(base_values, [0] + list(output_dims)),
            data=data,
            display_data=display_data,
            instance_names=None if instance_names is None else Alias(instance_names, 0),
            feature_names=feature_names,
            output_names=output_names,
            output_indexes=None if output_indexes is None else (output_dims, output_indexes),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            main_effects=main_effects,
            hierarchical_values=hierarchical_values,
            clustering=None if clustering is None else Obj(clustering, [0])
        )

    @property
    def shape(self):
        return _compute_shape(self._s.values)

    @property
    def values(self):
        return self._s.values
    @values.setter
    def values(self, new_values):
        self._s.values = new_values

    @property
    def base_values(self):
        return self._s.base_values
    @base_values.setter
    def base_values(self, new_base_values):
        self._s.base_values = new_base_values

    @property
    def data(self):
        return self._s.data
    @data.setter
    def data(self, new_data):
        self._s.data = new_data

    @property
    def display_data(self):
        return self._s.display_data
    @display_data.setter
    def display_data(self, new_display_data):
        if issubclass(type(new_display_data), pd.DataFrame):
            new_display_data = new_display_data.values
        self._s.display_data = new_display_data

    @property
    def instance_names(self):
        return self._s.instance_names

    @property
    def output_names(self):
        return self._s.output_names
    @output_names.setter 
    def output_names(self, new_output_names):
        self._s.output_names = new_output_names

    @property
    def output_indexes(self):
        return self._s.output_indexes

    @property
    def feature_names(self):
        return self._s.feature_names
    @feature_names.setter
    def feature_names(self, new_feature_names):
        self._s.feature_names = new_feature_names

    @property
    def lower_bounds(self):
        return self._s.lower_bounds

    @property
    def upper_bounds(self):
        return self._s.upper_bounds

    @property
    def main_effects(self):
        return self._s.main_effects
    @main_effects.setter
    def main_effects(self, new_main_effects):
        self._s.main_effects = new_main_effects

    @property
    def hierarchical_values(self):
        return self._s.hierarchical_values
    @hierarchical_values.setter
    def hierarchical_values(self, new_hierarchical_values):
        self._s.hierarchical_values = new_hierarchical_values

    @property
    def clustering(self):
        return self._s.clustering
    @clustering.setter
    def clustering(self, new_clustering):
        self._s.clustering = new_clustering

    def cohorts(self, cohorts):
        """ Split this explanation into several cohorts.

        Parameters
        ----------
        cohorts : int or array
            If this is an integer then we auto build that many cohorts using a decision tree. If this is
            an array then we treat that as an array of cohort names/ids for each instance.
        """

        if isinstance(cohorts, int):
            return _auto_cohorts(self, max_cohorts=cohorts)
        elif isinstance(cohorts, (list, tuple, np.ndarray)):
            cohorts = np.array(cohorts)
            return Cohorts(**{name: self[cohorts == name] for name in np.unique(cohorts)})
        else:
            raise TypeError("The given set of cohort indicators is not recognized! Please give an array or int.")

    def __repr__(self):
        out = ".values =\n"+self.values.__repr__()
        if self.base_values is not None:
            out += "\n\n.base_values =\n"+self.base_values.__repr__()
        if self.data is not None:
            out += "\n\n.data =\n"+self.data.__repr__()
        return out

    def __getitem__(self, item):
        """ This adds support for OpChain indexing.
        """
        if not isinstance(item, tuple):
            item = (item,)

        # convert any OpChains or magic strings
        for pos, t in enumerate(item):
            orig_t = t
            if issubclass(type(t), OpChain):
                t = t.apply(self)
                if issubclass(type(t), (np.int64, np.int32)): # because slicer does not like numpy indexes
                    t = int(t)
                elif issubclass(type(t), np.ndarray):
                    t = [int(v) for v in t] # slicer wants lists not numpy arrays for indexing
            elif issubclass(type(t), Explanation):
                t = t.values
            elif isinstance(t, str):

                # work around for 2D output_names since they are not yet slicer supported
                output_names_dims = []
                if "output_names" in self._s._objects:
                    output_names_dims = self._s._objects["output_names"].dim
                if pos != 0 and pos in output_names_dims and len(output_names_dims) == 2:
                    new_values = []
                    new_base_values = []
                    new_data = []
                    new_self = copy.deepcopy(self)
                    for i in range(len(self.values)):
                        for j in range(len(self.output_names[i])):
                            s = self.output_names[i][j]
                            if s == t:
                                new_values.append(np.array(self.values[i][:,j]))
                                new_data.append(np.array(self.data[i]))
                                new_base_values.append(self.base_values[i][j])
                    new_self = copy.deepcopy(self)
                    new_self.values = np.array(new_values)
                    new_self.base_values = np.array(new_base_values)
                    new_self.data = np.array(new_data)
                    new_self.output_names = t
                    new_self.feature_names = np.array(new_data)
                    new_self.clustering = None

                # work around for 2D feature_names since they are not yet slicer supported
                feature_names_dims = []
                if "feature_names" in self._s._objects:
                    feature_names_dims = self._s._objects["feature_names"].dim
                if pos != 0 and pos in feature_names_dims and len(feature_names_dims) == 2:
                    new_values = []
                    new_data = []
                    for i in range(len(self.values)):
                        for s,v,d in zip(self.feature_names[i], self.values[i], self.data[i]):
                            if s == t:
                                new_values.append(v)
                                new_data.append(d)
                    new_self = copy.deepcopy(self)
                    new_self.values = new_values
                    new_self.data = new_data
                    new_self.feature_names = t
                    new_self.clustering = None
                    return new_self

            if issubclass(type(t), (np.int8, np.int16, np.int32, np.int64)):
                t = int(t)

            if t is not orig_t:
                tmp = list(item)
                tmp[pos] = t
                item = tuple(tmp)

        # call slicer for the real work
        new_self = copy.copy(self)
        new_self._s = self._s.__getitem__(item)
        new_self.op_history.append({
            "name": "__getitem__",
            "args": (item,),
            "prev_shape": self.shape
        })

        return new_self

    def __len__(self):
        return self.shape[0]

    def __copy__(self):
        new_exp = Explanation(
            self.values,
            self.base_values,
            self.data,
            self.display_data,
            self.instance_names,
            self.feature_names,
            self.output_names,
            self.output_indexes,
            self.lower_bounds,
            self.upper_bounds,
            self.main_effects,
            self.hierarchical_values,
            self.clustering
        )
        new_exp.op_history = copy.copy(self.op_history)
        return new_exp

    def _apply_binary_operator(self, other, binary_op, op_name):
        new_exp = self.__copy__()
        new_exp.op_history = copy.copy(self.op_history)
        new_exp.op_history.append({
            "name": op_name,
            "args": (other,),
            "prev_shape": self.shape
        })
        if isinstance(other, Explanation):
            new_exp.values = binary_op(new_exp.values, other.values)
            if new_exp.data is not None:
                new_exp.data = binary_op(new_exp.data, other.data)
            if new_exp.base_values is not None:
                new_exp.base_values = binary_op(new_exp.base_values, other.base_values)
        else:
            new_exp.values = binary_op(new_exp.values, other)
            if new_exp.data is not None:
                new_exp.data = binary_op(new_exp.data, other)
            if new_exp.base_values is not None:
                new_exp.base_values = binary_op(new_exp.base_values, other)
        return new_exp

    def __add__(self, other):
        return self._apply_binary_operator(other, operator.add, "__add__")
    
    def __radd__(self, other):
        return self._apply_binary_operator(other, operator.add, "__add__")
        
    def __sub__(self, other):
        return self._apply_binary_operator(other, operator.sub, "__sub__")
    
    def __rsub__(self, other):
        return self._apply_binary_operator(other, operator.sub, "__sub__")
    
    def __mul__(self, other):
        return self._apply_binary_operator(other, operator.mul, "__mul__")
    
    def __rmul__(self, other):
        return self._apply_binary_operator(other, operator.mul, "__mul__")
        
    def __truediv__(self, other):
        return self._apply_binary_operator(other, operator.truediv, "__truediv__")

    @property
    def abs(self):
        new_self = copy.copy(self)
        new_self.values = np.abs(new_self.values)
        new_self.op_history.append({
            "name": "abs",
            "prev_shape": self.shape
        })
        return new_self

    def _numpy_func(self, fname, **kwargs):
        new_self = copy.copy(self)
        axis = kwargs.get("axis", None)

        # collapse the slicer to right shape
        if axis == 0:
            new_self = new_self[0]
        elif axis == 1:
            new_self = new_self[1]
        elif axis == 2:
            new_self = new_self[2]
        if axis in [0,1,2]:
            new_self.op_history = new_self.op_history[:-1] # pop off the slicing operation we just used

        if self.feature_names is not None and not is_1d(self.feature_names) and axis == 0:
            new_values = self._flatten_feature_names()
            new_self.feature_names = np.array(list(new_values.keys()))
            new_self.values = np.array([getattr(np, fname)(v,0) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = getattr(np, fname)(np.array(self.values), **kwargs)
            if new_self.data is not None:
                try:
                    new_self.data = getattr(np, fname)(np.array(self.data), **kwargs)
                except:
                    new_self.data = None
            if new_self.base_values is not None and issubclass(type(axis), int) and len(self.base_values.shape) > axis:
                new_self.base_values = getattr(np, fname)(self.base_values, **kwargs)
            elif issubclass(type(axis), int):
                new_self.base_values = None

        if axis == 0 and self.clustering is not None and len(self.clustering.shape) == 3:
            if self.clustering.std(0).sum() < 1e-8:
                new_self.clustering = self.clustering[0]
            else:
                new_self.clustering = None
        
        new_self.op_history.append({
            "name": fname,
            "kwargs": kwargs,
            "prev_shape": self.shape,
            "collapsed_instances": axis == 0
        })
        
        return new_self

    def mean(self, axis):
        return self._numpy_func("mean", axis=axis)

    def max(self, axis):
        return self._numpy_func("max", axis=axis)

    def min(self, axis):
        return self._numpy_func("min", axis=axis)

    def sum(self, axis=None, grouping=None):
        if grouping is None:
            return self._numpy_func("sum", axis=axis)
        elif axis == 1 or len(self.shape) == 1:
            return group_features(self, grouping)
        else:
            raise ValueError("Only axis = 1 is supported for grouping right now...")

    # def reshape(self, *args):
    #     return self._numpy_func("reshape", newshape=args)

    @property
    def abs(self):
        return self._numpy_func("abs")

    @property
    def identity(self):
        return self

    @property
    def argsort(self):
        return self._numpy_func("argsort")

    @property
    def flip(self):
        return self._numpy_func("flip")


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
            raise ValueError("The hclust order only supports 2D arrays right now!")

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

    def _flatten_feature_names(self):
        new_values = {}
        for i in range(len(self.values)):
            for s,v in zip(self.feature_names[i], self.values[i]):
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

    def percentile(self, q, axis=None):
        new_self = copy.deepcopy(self)
        if self.feature_names is not None and not is_1d(self.feature_names) and axis == 0:
            new_values = self._flatten_feature_names()
            new_self.feature_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.percentile(v, q) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = np.percentile(new_self.values, q, axis)
            new_self.data = np.percentile(new_self.data, q, axis)
        #new_self.data = None
        new_self.op_history.append({
            "name": "percentile",
            "args": (axis,),
            "prev_shape": self.shape,
            "collapsed_instances": axis == 0
        })
        return new_self

def group_features(shap_values, feature_map):
    # TODO: support and deal with clusterings
    reverse_map = {}
    for name in feature_map:
        reverse_map[feature_map[name]] = reverse_map.get(feature_map[name], []) + [name]
    
    curr_names = shap_values.feature_names
    shap_values_new = copy.deepcopy(shap_values)
    found = {}
    i = -1
    rank1 = len(shap_values.shape) == 1
    for name in curr_names:
        new_name = feature_map.get(name, name)
        if new_name in found:
            continue
        i += 1
        found[new_name] = True
        
        new_name = feature_map.get(name, name)
        cols_to_sum = reverse_map.get(new_name, [new_name])
        old_inds = [curr_names.index(v) for v in cols_to_sum]
        
        if rank1:
            shap_values_new.values[i] = shap_values.values[old_inds].sum()
            shap_values_new.data[i] = shap_values.data[old_inds].sum()
        else:
            shap_values_new.values[:,i] = shap_values.values[:,old_inds].sum(1)
            shap_values_new.data[:,i] = shap_values.data[:,old_inds].sum(1)
        shap_values_new.feature_names[i] = new_name

    return Explanation(
        shap_values_new.values[:i] if rank1 else shap_values_new.values[:,:i],
        base_values = shap_values_new.base_values,
        data = shap_values_new.data[:i] if rank1 else shap_values_new.data[:,:i],
        display_data = None if shap_values_new.display_data is None else (shap_values_new.display_data[:,:i] if rank1 else shap_values_new.display_data[:,:i]),
        instance_names = None,
        feature_names = None if shap_values_new.feature_names is None else shap_values_new.feature_names[:i],
        output_names = None,
        output_indexes = None,
        lower_bounds = None,
        upper_bounds = None,
        main_effects = None,
        hierarchical_values = None,
        clustering = None
    )

def compute_output_dims(values, base_values, data):
    values_shape = _compute_shape(values)

    # input shape matches the data shape
    if data is not None:
        data_shape = _compute_shape(data)

    # if we are not given any data we assume it would be the same shape as the given values
    else: 
        data_shape = values_shape

    # output shape is known from the base values
    if base_values is not None:
        output_shape = _compute_shape(base_values)[1:]
    else:
        output_shape = tuple()

    interaction_order = len(values_shape) - len(data_shape) - len(output_shape)
    values_dims = list(range(len(values_shape)))
    output_dims = range(len(data_shape) + interaction_order, len(values_shape))
    return tuple(output_dims)

def is_1d(val):
    return not (isinstance(val[0], list) or isinstance(val[0], np.ndarray))

class Op():
    pass

class Percentile(Op):
    def __init__(self, percentile):
        self.percentile = percentile

    def add_repr(self, s, verbose=False):
        return "percentile("+s+", "+str(self.percentile)+")"



def _compute_shape(x):
    if not hasattr(x, "__len__") or isinstance(x, str):
        return tuple()
    elif not sp.sparse.issparse(x) and len(x) > 0 and isinstance(x[0], str):
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
            return (1,) + _compute_shape(x[0])
        else:
            first_shape = _compute_shape(x[0])
            if first_shape == tuple():
                return (len(x),)
            else: # we have an array of arrays...
                matches = np.ones(len(first_shape), dtype=np.bool)
                for i in range(1, len(x)):
                    shape = _compute_shape(x[i])
                    assert len(shape) == len(first_shape), "Arrays in Explanation objects must have consistent inner dimensions!"
                    for j in range(0, len(shape)):
                        matches[j] &= shape[j] == first_shape[j]
                return (len(x),) + tuple(first_shape[j] if match else None for j, match in enumerate(matches))

class Cohorts():
    def __init__(self, **kwargs):
        self.cohorts = kwargs
        for k in self.cohorts:
            assert isinstance(self.cohorts[k], Explanation), "All the arguments to a Cohorts set must be Explanation objects!"

    def __getitem__(self, item):
        new_cohorts = Cohorts()
        for k in self.cohorts:
            new_cohorts.cohorts[k] = self.cohorts[k].__getitem__(item)
        return new_cohorts
    
    def __getattr__(self, name):
        new_cohorts = Cohorts()
        for k in self.cohorts:
            new_cohorts.cohorts[k] = getattr(self.cohorts[k], name)
        return new_cohorts
    
    def __call__(self, *args, **kwargs):
        new_cohorts = Cohorts()
        for k in self.cohorts:
            new_cohorts.cohorts[k] = self.cohorts[k].__call__(*args, **kwargs)
        return new_cohorts

    def __repr__(self):
        return f"<shap._explanation.Cohorts object with {len(self.cohorts)} cohorts of sizes: {[v.shape for v in self.cohorts.values()]}>"


def _auto_cohorts(shap_values, max_cohorts):
    """ This uses a DecisionTreeRegressor to build a group of cohorts with similar SHAP values.
    """

    # fit a decision tree that well spearates the SHAP values 
    m = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=max_cohorts)
    m.fit(shap_values.data, shap_values.values)

    # group instances by their decision paths
    paths = m.decision_path(shap_values.data).toarray()
    unique_paths = np.unique(m.decision_path(shap_values.data).todense(), axis=0)
    path_names = []

    # mark each instance with a path name
    for i in range(shap_values.shape[0]):
        name = ""
        for j in range(len(paths[i])):
            if paths[i,j] > 0:
                feature = m.tree_.feature[j]
                threshold = m.tree_.threshold[j]
                val = shap_values.data[i,feature]
                if feature >= 0:
                    name += str(shap_values.feature_names[feature])
                    if val < threshold:
                        name += " < "
                    else:
                        name += " >= "
                    name += str(threshold) + " & "
        path_names.append(name[:-3]) # the -3 strips off the last unneeded ' & '
    path_names = np.array(path_names)
    
    # split the instances into cohorts by their path names
    cohorts = {}
    for name in np.unique(path_names):
        cohorts[name] = shap_values[path_names == name]
    
    return Cohorts(**cohorts)
