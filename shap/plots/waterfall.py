import numpy as np
import matplotlib.pyplot as pl
from shap.plots import labels
from shap.common import safe_isinstance
from . import colors


def waterfall_plot(expected_value, shap_values, features=None, feature_names=None, max_display=10, show=True):
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
        
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])
        
    num_features = min(max_display, len(shap_values))
    row_height=0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(shap_values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    loc = expected_value + shap_values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    if num_features == len(shap_values):
        num_individual = num_features
    else:
        num_individual = num_features - 1
    for i in range(num_individual):
        sval = shap_values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            neg_lefts.append(loc)
        if num_individual != num_features or i + 1 < num_individual:
            pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = feature_names[order[i]] + " = " + str(features[order[i]])

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(shap_values):
        yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
        remaining_impact = expected_value - loc
        if remaining_impact < 0:
            c = colors.red_rgb
        else:
            c = colors.blue_rgb

        pl.barh([0], [remaining_impact], left=loc, color=c)
    
    # draw the bars
    pl.barh(pos_inds, pos_widths, left=pos_lefts, color=colors.red_rgb)
    pl.barh(neg_inds, neg_widths, left=neg_lefts, color=colors.blue_rgb)
    pl.yticks(range(num_features), yticklabels, fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    pl.axvline(expected_value, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    pl.axvline(expected_value + shap_values.sum(), 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    #pl.gca().tick_params(color=, labelcolor=axis_color)
    pl.xlabel("Feature impact on the model output", fontsize=13)

    if show:
        pl.show()