import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from ._labels import labels
from . import colors
import numpy as np


# TODO: improve the bar chart to look better like the waterfall plot (gray feature values, and a "the rest..." feature at the bottom)
def bar(shap_values, max_display=20, show=True):
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
        color=[colors.blue_rgb if shap_values[feature_inds[i]] <= 0 or features is None else colors.red_rgb for i in range(len(y_pos))]
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
    pl.gca().tick_params('x', labelsize=11)
    
    if features is None:
        pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    else:
        pl.xlabel(labels["VALUE"], fontsize=13)
    
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