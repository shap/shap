"""Visualize the SHAP values with additive force style layouts."""

import base64
import json
import os
import random
import re
import string
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse

try:
    from IPython.display import HTML, display
    have_ipython = True
except ImportError:
    have_ipython = False

from ..plots._force_matplotlib import draw_additive_plot
from ..utils import hclust_ordering
from ..utils._exceptions import DimensionError
from ..utils._legacy import Data, DenseData, Instance, Link, Model, convert_to_link
from ._labels import labels


def force(
    base_value,
    shap_values=None,
    features=None,
    feature_names=None,
    out_names=None,
    link="identity",
    plot_cmap="RdBu",
    matplotlib=False,
    show=True,
    figsize=(20, 3),
    ordering_keys=None,
    ordering_keys_time_format=None,
    text_rotation=0,
    contribution_threshold=0.05,
):
    """Visualize the given SHAP values with an additive force layout.

    Parameters
    ----------
    base_value : float or shap.Explanation
        If a float is passed in, this is the reference value that the feature contributions start from.
        For SHAP values, it should be the value of ``explainer.expected_value``.
        However, it is recommended to pass in a SHAP :class:`.Explanation` object instead (``shap_values``
        is not necessary in this case).

    shap_values : numpy.array
        Matrix of SHAP values (# features) or (# samples x # features). If this is a
        1D array, then a single force plot will be drawn. If it is a 2D array, then a
        stacked force plot will be drawn.

    features : numpy.array
        Matrix of feature values (# features) or (# samples x # features). This provides the values of all the
        features, and should be the same shape as the ``shap_values`` argument.

    feature_names : list
        List of feature names (# features).

    out_names : str
        The name of the output of the model (plural to support multi-output plotting in the future).

    link : "identity" or "logit"
        The transformation used when drawing the tick mark labels. Using "logit" will change log-odds numbers
        into probabilities.

    plot_cmap : str or list[str]
        Color map to use. It can be a string (defaults to ``RdBu``) or a list of hex color strings.

    matplotlib : bool
        Whether to use the default Javascript output, or the (less developed) matplotlib output.
        Using matplotlib can be helpful in scenarios where rendering Javascript/HTML
        is inconvenient. Defaults to False.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.
        Only applicable when ``matplotlib`` is set to True.

    figsize :
        Figure size of the matplotlib output.

    contribution_threshold : float
        Controls the feature names/values that are displayed on force plot.
        Only features that the magnitude of their shap value is larger than min_perc * (sum of all abs shap values)
        will be displayed.

    """
    # support passing an explanation object
    if str(type(base_value)).endswith("Explanation'>"):
        shap_exp = base_value
        base_value = shap_exp.base_values
        shap_values = shap_exp.values
        if features is None:
            if shap_exp.display_data is None:
                features = shap_exp.data
            else:
                features = shap_exp.display_data
        if scipy.sparse.issparse(features):
            features = features.toarray().flatten()
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    # auto unwrap the base_value
    if isinstance(base_value, np.ndarray):
        if len(base_value) == 1:
            base_value = base_value[0]
        elif len(base_value) > 1 and np.all(base_value == base_value[0]):
            base_value = base_value[0]

    if isinstance(base_value, (np.ndarray, list)):
        if not isinstance(shap_values, (list, np.ndarray)) or len(shap_values) != len(base_value):
            emsg = (
                "In v0.20, force plot now requires the base value as the first parameter! "
                "Try shap.plots.force(explainer.expected_value, shap_values) or "
                "for multi-output models try "
                "shap.plots.force(explainer.expected_value[0], shap_values[..., 0])."
            )
            raise TypeError(emsg)

    if isinstance(shap_values, list):
        emsg = "The shap_values arg looks multi output, try `shap_values[i]` instead."
        raise TypeError(emsg)

    link = convert_to_link(link)

    if type(shap_values) != np.ndarray:
        return visualize(shap_values)

    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = list(features.columns)
        features = features.values
    elif isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (1, len(shap_values)))

    if out_names is None:
        out_names = ["f(x)"]
    elif isinstance(out_names, str):
        out_names = [out_names]

    if shap_values.shape[0] == 1:
        if feature_names is None:
            feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
        if features is None:
            features = ["" for _ in range(len(feature_names))]
        if type(features) == np.ndarray:
            features = features.flatten()

        # check that the shape of the shap_values and features match
        if len(features) != shap_values.shape[1]:
            emsg = "Length of features is not equal to the length of shap_values!"
            if len(features) == shap_values.shape[1] - 1:
                emsg += (
                    " You might be using an old format shap_values array with the base value "
                    "as the last column. In this case, just pass the array without the last column."
                )
            raise DimensionError(emsg)

        instance = Instance(np.zeros((1, len(feature_names))), features)
        e = AdditiveExplanation(
            base_value,
            np.sum(shap_values[0, :]) + base_value,
            shap_values[0, :],
            None,
            instance,
            link,
            Model(None, out_names),
            DenseData(np.zeros((1, len(feature_names))), list(feature_names))
        )

        return visualize(e,
                         plot_cmap,
                         matplotlib,
                         figsize=figsize,
                         show=show,
                         text_rotation=text_rotation,
                         min_perc=contribution_threshold)

    else:
        if matplotlib:
            raise NotImplementedError("matplotlib = True is not yet supported for force plots with multiple samples!")

        if shap_values.shape[0] > 3000:
            warnings.warn("shap.plots.force is slow for many thousands of rows, try subsampling your data.")

        exps = []
        for k in range(shap_values.shape[0]):
            if feature_names is None:
                feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
            if features is None:
                display_features = ["" for i in range(len(feature_names))]
            else:
                display_features = features[k, :]

            instance = Instance(np.ones((1, len(feature_names))), display_features)
            e = AdditiveExplanation(
                base_value,
                np.sum(shap_values[k, :]) + base_value,
                shap_values[k, :],
                None,
                instance,
                link,
                Model(None, out_names),
                DenseData(np.ones((1, len(feature_names))), list(feature_names))
            )
            exps.append(e)

        return visualize(
                    exps,
                    plot_cmap=plot_cmap,
                    ordering_keys=ordering_keys,
                    ordering_keys_time_format=ordering_keys_time_format,
                    text_rotation=text_rotation
                )


class Explanation:
    def __init__(self):
        pass


class AdditiveExplanation(Explanation):
    """Data structure for AdditiveForceVisualizer / AdditiveForceArrayVisualizer."""

    def __init__(self, base_value, out_value, effects, effects_var, instance, link, model, data):
        """Parameters
        ----------
        base_value : float
            This is the reference value that the feature contributions start from.
            For SHAP values, it should be the value of ``explainer.expected_value``.

        out_value : float
            The model prediction value, taken as the sum of the SHAP values across all
            features and the ``base_value``.

        """
        self.base_value = base_value
        self.out_value = out_value
        self.effects = effects
        self.effects_var = effects_var
        assert isinstance(instance, Instance)
        self.instance = instance
        assert isinstance(link, Link)
        self.link = link
        assert isinstance(model, Model)
        self.model = model
        assert isinstance(data, Data)
        self.data = data

err_msg = """
<div style='color: #900; text-align: center;'>
  <b>Visualization omitted, Javascript library not loaded!</b><br>
  Have you run `initjs()` in this notebook? If this notebook was from another
  user you must also trust this notebook (File -> Trust notebook). If you are viewing
  this notebook on github the Javascript has been stripped for security. If you are using
  JupyterLab this error is because a JupyterLab extension has not yet been written.
</div>"""


def getjs():
    bundle_path = os.path.join(os.path.split(__file__)[0], "resources", "bundle.js")
    with open(bundle_path, encoding="utf-8") as f:
        bundle_data = f.read()
    return f"<script charset='utf-8'>{bundle_data}</script>"


def initjs():
    """Initialize the necessary javascript libraries for interactive force plots.

    Run this only in a notebook environment with IPython installed.
    """
    assert have_ipython, "IPython must be installed to use initjs()! Run `pip install ipython` and then restart shap."

    logo_path = os.path.join(os.path.split(__file__)[0], "resources", "logoSmallGray.png")
    with open(logo_path, "rb") as f:
        logo_data = f.read()
    logo_data = base64.b64encode(logo_data).decode("utf-8")
    display(HTML(
        f"<div align='center'><img src='data:image/png;base64,{logo_data}' /></div>" +
        getjs()
    ))


def save_html(out_file, plot, full_html=True):
    """Save html plots to an output file.

    Parameters
    ----------
    out_file : str or file
        Location or file to be written to.

    plot : BaseVisualizer
        Visualizer returned by :func:`shap.plots.force()`.

    full_html : boolean (default: True)
        If ``True``, writes a complete HTML document starting
        with an ``<html>`` tag. If ``False``, only script and div
        tags are included.

    """
    if not isinstance(plot, BaseVisualizer):
        raise TypeError("`save_html` requires a Visualizer returned by `shap.plots.force()`.")

    internal_open = False
    if isinstance(out_file, str):
        out_file = open(out_file, "w", encoding="utf-8")
        internal_open = True

    if full_html:
        out_file.write("<html><head><meta http-equiv='content-type' content='text/html'; charset='utf-8'>")

    # dump the js code
    out_file.write(getjs())

    if full_html:
        out_file.write("</head><body>\n")

    out_file.write(plot.html())

    if full_html:
        out_file.write("</body></html>\n")

    if internal_open:
        out_file.close()


def id_generator(size=20, chars=string.ascii_uppercase + string.digits):
    return "i"+''.join(random.choice(chars) for _ in range(size))


def ensure_not_numpy(x):
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str_):
        return str(x)
    elif isinstance(x, np.generic):
        return float(x.item())
    else:
        return x


def verify_valid_cmap(cmap):
    """Checks that cmap is either a str or list of hex colors"""
    if not (isinstance(cmap, (str, list)) or str(type(cmap)).endswith("unicode'>")):
        emsg = f"Plot color map must be string or list! Not {type(cmap)}."
        raise TypeError(emsg)

    if isinstance(cmap, list):
        if len(cmap) < 2:
            raise ValueError("Color map must be at least two colors.")
        _rgbstring = re.compile(r'#[a-fA-F0-9]{6}$')
        for color in cmap:
            if not _rgbstring.match(color):
                raise ValueError(f"Invalid color {color} found in cmap.")

    return cmap


def visualize(
    e,
    plot_cmap="RdBu",
    matplotlib=False,
    figsize=(20, 3),
    show=True,
    ordering_keys=None,
    ordering_keys_time_format=None,
    text_rotation=0,
    min_perc=0.05,
):
    """Main interface for switching between matplotlib / javascript force plots.

    Parameters
    ----------
    e : AdditiveExplanation
        Contains the data necessary for additive force plots.

    """
    plot_cmap = verify_valid_cmap(plot_cmap)

    if isinstance(e, AdditiveExplanation):
        if matplotlib:
            return AdditiveForceVisualizer(e, plot_cmap=plot_cmap).matplotlib(
                figsize=figsize,
                show=show,
                text_rotation=text_rotation,
                min_perc=min_perc,
            )
        else:
            return AdditiveForceVisualizer(e, plot_cmap=plot_cmap)
    elif isinstance(e, Explanation):
        if matplotlib:
            raise ValueError("Matplotlib plot is only supported for additive explanations")
        return SimpleListVisualizer(e)
    elif isinstance(e, Sequence) and len(e) > 0 and isinstance(e[0], AdditiveExplanation):
        if matplotlib:
            raise ValueError("Matplotlib plot is only supported for additive explanations")
        return AdditiveForceArrayVisualizer(
            e,
            plot_cmap=plot_cmap,
            ordering_keys=ordering_keys,
            ordering_keys_time_format=ordering_keys_time_format,
        )
    else:
        raise ValueError("visualize() can only display Explanation objects (or arrays of them)!")


class BaseVisualizer:
    pass

class SimpleListVisualizer(BaseVisualizer):
    def __init__(self, e):
        if not isinstance(e, Explanation):
            emsg = "SimpleListVisualizer can only visualize Explanation objects!"
            raise TypeError(emsg)

        # build the json data
        features = {}
        for i in filter(lambda j: e.effects[j] != 0, range(len(e.data.group_names))):
            features[i] = {
                "effect": e.effects[i],
                "value": e.instance.group_display_values[i]
            }
        self.data = {
            "outNames": e.model.out_names,
            "base_value": e.base_value,
            "link": str(e.link),
            "featureNames": e.data.group_names,
            "features": features,
            "plot_cmap":e.plot_cmap.plot_cmap
        }

    def html(self):
        # assert have_ipython, "IPython must be installed to use this visualizer! Run `pip install ipython` and then restart shap."
        generated_id = id_generator()
        return f"""
<div id='{generated_id}'>{err_msg}</div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.SimpleListVisualizer, {json.dumps(self.data)}),
    document.getElementById('{generated_id}')
  );
</script>"""

    def _repr_html_(self):
        return self.html()


class AdditiveForceVisualizer(BaseVisualizer):
    """Visualizer for a single Additive Force plot."""

    def __init__(self, e, plot_cmap="RdBu"):
        """Parameters
        ----------
        e : AdditiveExplanation
            Contains the data necessary for additive force plots.

        plot_cmap : str or list[str]
            Color map to use. It can be a string (defaults to ``RdBu``) or a list of hex color strings.

        """
        if not isinstance(e, AdditiveExplanation):
            emsg = "AdditiveForceVisualizer can only visualize AdditiveExplanation objects!"
            raise TypeError(emsg)

        # build the json data
        features = {}
        for i in filter(lambda j: e.effects[j] != 0, range(len(e.data.group_names))):
            features[i] = {
                "effect": ensure_not_numpy(e.effects[i]),
                "value": ensure_not_numpy(e.instance.group_display_values[i])
            }
        self.data = {
            "outNames": e.model.out_names,
            "baseValue": ensure_not_numpy(e.base_value),
            "outValue": ensure_not_numpy(e.out_value),
            "link": str(e.link),
            "featureNames": e.data.group_names,
            "features": features,
            "plot_cmap": plot_cmap,
        }

    def html(self, label_margin=20):
        # assert have_ipython, "IPython must be installed to use this visualizer! Run `pip install ipython` and then restart shap."
        self.data["labelMargin"] = label_margin
        generated_id = id_generator()
        return f"""
<div id='{generated_id}'>{err_msg}</div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceVisualizer, {json.dumps(self.data)}),
    document.getElementById('{generated_id}')
  );
</script>"""

    def matplotlib(self, figsize, show, text_rotation, min_perc=0.05):
        fig = draw_additive_plot(self.data,
                                 figsize=figsize,
                                 show=show,
                                 text_rotation=text_rotation,
                                 min_perc=min_perc)

        return fig

    def _repr_html_(self):
        return self.html()


class AdditiveForceArrayVisualizer(BaseVisualizer):
    """Visualizer for a sequence of AdditiveExplanation, as a stacked force plot."""

    def __init__(self, arr, plot_cmap="RdBu", ordering_keys=None, ordering_keys_time_format=None):
        if not isinstance(arr[0], AdditiveExplanation):
            emsg = (
                "AdditiveForceArrayVisualizer can only visualize arrays of "
                "AdditiveExplanation objects!"
            )
            raise TypeError(emsg)

        # order the samples by their position in a hierarchical clustering
        if all(e.model.f == arr[1].model.f for e in arr):
            clustOrder = hclust_ordering(np.vstack([e.effects for e in arr]))
        else:
            emsg = "Tried to visualize an array of explanations from different models!"
            raise ValueError(emsg)

        # make sure that we put the higher predictions first...just for consistency
        if sum(arr[clustOrder[0]].effects) < sum(arr[clustOrder[-1]].effects):
            np.flipud(clustOrder) # reverse

        # build the json data
        clustOrder = np.argsort(clustOrder) # inverse permutation
        self.data = {
            "outNames": arr[0].model.out_names,
            "baseValue": ensure_not_numpy(arr[0].base_value),
            "link": arr[0].link.__str__(),
            "featureNames": arr[0].data.group_names,
            "explanations": [],
            "plot_cmap": plot_cmap,
            "ordering_keys": list(ordering_keys) if hasattr(ordering_keys, '__iter__') else None,
            "ordering_keys_time_format": ordering_keys_time_format,
        }
        for ind, e in enumerate(arr):
            self.data["explanations"].append({
                "outValue": ensure_not_numpy(e.out_value),
                "simIndex": ensure_not_numpy(clustOrder[ind])+1,
                "features": {}
            })
            for i in filter(lambda j: e.effects[j] != 0 or e.instance.x[0,j] != 0, range(len(e.data.group_names))):
                self.data["explanations"][-1]["features"][i] = {
                    "effect": ensure_not_numpy(e.effects[i]),
                    "value": ensure_not_numpy(e.instance.group_display_values[i])
                }

    def html(self):
        # assert have_ipython, "IPython must be installed to use this visualizer! Run `pip install ipython` and then restart shap."
        _id = id_generator()
        return f"""
<div id='{_id}'>{err_msg}</div>
 <script>
   if (window.SHAP) SHAP.ReactDom.render(
    SHAP.React.createElement(SHAP.AdditiveForceArrayVisualizer, {json.dumps(self.data)}),
    document.getElementById('{_id}')
  );
</script>"""

    def _repr_html_(self):
        return self.html()
