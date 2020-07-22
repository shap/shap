import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from ._labels import labels
from ..utils import format_value
from . import colors
import numpy as np


# TODO: improve the bar chart to look better like the waterfall plot (gray feature values, and a "the rest..." feature at the bottom)
def bar(shap_values, max_display=10, show=True):
    """ Create a bar plot of a set of SHAP values.

    If a single sample is passed then we plot the SHAP values as a bar chart. If an
    Explanation with many samples is passed then we plot the mean absolute value for
    each feature column as a bar chart.


    Parameters
    ----------
    shap_values : shap.Explanation
        A single row of a SHAP Explanation object (i.e. shap_values[0]) or a multi-row Explanation
        object that we want to summarize.

    max_display : int
        The maximum number of bars to display.

    show : bool
        If show is set to False then we don't call the matplotlib.pyplot.show() function. This allows
        further customization of the plot by the caller after the bar() function is finished. 

    """

    assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values paramemter must be a shap.Explanation object!"

    
    features = shap_values.data
    feature_names = shap_values.feature_names
    shap_values = shap_values.values

    # doing a global bar plot
    if len(shap_values.shape) == 2:
        shap_values = np.abs(shap_values).mean(0)
        features = None

    # unwrap pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
        
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])
    
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(shap_values))

    
    
    feature_order = np.argsort(-np.abs(shap_values))
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(shap_values):
        shap_values[feature_order[num_features-1]] = np.sum([shap_values[feature_order[i]] for i in range(num_features-1, len(shap_values))])
    
    yticklabels = []
    for i in feature_inds:
        if features is not None:
            yticklabels.append(format_value(features[i], "%0.03f") + " = " + feature_names[i])
        else:
            yticklabels.append(feature_names[i])
    if num_features < len(shap_values):
        yticklabels[-1] = "%d other features" % (len(shap_values) - num_features + 1)

    row_height = 0.5
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    #pl.axvline(0, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    negative_values_present = np.sum(shap_values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    pl.barh(
        y_pos, shap_values[feature_inds],
        0.7, align='center',
        color=[colors.blue_rgb if shap_values[feature_inds[i]] <= 0 or features is None else colors.red_rgb for i in range(len(y_pos))]
    )
    
    pl.yticks(list(y_pos) + list(y_pos), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=13)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width

    for i in range(len(y_pos)):
        ind = feature_order[i]
        if shap_values[ind] < 0:
            txt_obj = pl.text(
                shap_values[ind] - (5/72)*bbox_to_xscale, y_pos[i], format_value(shap_values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=12
            )
        else:
            txt_obj = pl.text(
                shap_values[ind] + (5/72)*bbox_to_xscale, y_pos[i], format_value(shap_values[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.blue_rgb if features is None else colors.red_rgb,
                fontsize=12
            )
    
    if features is not None:
        features = list(features)

        # try and round off any trailing zeros after the decimal point in the feature values
        for i in range(len(features)):
            try:
                if round(features[i]) == features[i]:
                    features[i] = int(features[i])
            except TypeError:
                pass # features[i] must not be a number
    
    #pl.gca().set_yticklabels(yticklabels)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=11)

    xmin,xmax = pl.gca().get_xlim()
    
    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.05)
    
    if features is None:
        pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    else:
        pl.xlabel(labels["VALUE"], fontsize=13)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = pl.gca().yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
    
    if show:
        pl.show()



def bar_legacy(shap_values, features=None, feature_names=None, max_display=None, show=True):
    
    # unwrap pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
        
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])
    
    if max_display is None:
        max_display = 7
    else:
        max_display = min(len(feature_names), max_display)
        
    
    feature_order = np.argsort(-np.abs(shap_values))
    
    # 
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    pl.barh(
        y_pos, shap_values[feature_inds],
        0.7, align='center',
        color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    pl.yticks(y_pos, fontsize=13)
    if features is not None:
        features = list(features)

        # try and round off any trailing zeros after the decimal point in the feature values
        for i in range(len(features)):
            try:
                if round(features[i]) == features[i]:
                    features[i] = int(features[i])
            except TypeError:
                pass # features[i] must not be a number
    yticklabels = []
    for i in feature_inds:
        if features is not None:
            yticklabels.append(feature_names[i] + " = " + str(features[i]))
        else:
            yticklabels.append(feature_names[i])
    pl.gca().set_yticklabels(yticklabels)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    #pl.gca().spines['left'].set_visible(False)
    
    pl.xlabel("SHAP value (impact on model output)")
    
    if show:
        pl.show()