import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from . import labels
from . import colors
import numpy as np


def bar_plot(shap_values, features=None, feature_names=None, max_display=None, show=True):
    
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