# %% Initialize

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as pl
import nose.tools as nt
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

import shap

random_state = 7

X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
    "random_state": random_state
}

model = lgb.train(params, d_train, 1000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=False)

explainer = shap.TreeExplainer(model)
base_value = explainer.expected_value
select = range(20)
features = X_test.iloc[select]
y_label = y_test[select]
shap_values = explainer.shap_values(features)
shap_interaction_values = explainer.shap_interaction_values(features)
features_display = X_display.loc[features.index]


# %% Visual tests

args1 = dict(base_value=base_value, shap_values=shap_values, matplotlib=True)
args2 = args1.copy()
args2["shap_values"] = shap_interaction_values

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# Basic plots with default (importance) sort and generated labels.
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

shap.decision_plot(**args1)
shap.decision_plot(**args2)

shap.decision_plot(highlight=[0, 9], **args1)

shap.decision_plot(features=features_display, **args1)
shap.decision_plot(features=features_display, **args2)

# Plot a single observation
shap.decision_plot(base_value, shap_values[0, :], matplotlib=True)
shap.decision_plot(base_value, shap_values[[0], :], matplotlib=True)
# Now, with a Pandas Series
shap.decision_plot(base_value, shap_values[0, :], features=features_display.iloc[0, :], matplotlib=True)
shap.decision_plot(base_value, shap_interaction_values[[0], :], features=features_display.iloc[0, :], matplotlib=True)
# Now with a single observation using a matrix and a Pandas Dataframe
shap.decision_plot(base_value, shap_values[[0], :], features=features_display.iloc[[0], :], matplotlib=True)
shap.decision_plot(base_value, shap_interaction_values[[0], :], features=features_display.iloc[[0], :], matplotlib=True)
# Now with feature names in the features argument.
names = features_display.columns.to_list()
shap.decision_plot(base_value, shap_values[[0], :], features=names, matplotlib=True)
shap.decision_plot(base_value, shap_interaction_values[[0], :], features=names, matplotlib=True)
# Now with feature names in the features argument as numpy.
shap.decision_plot(base_value, shap_values[[0], :], features=np.array(names), matplotlib=True)
shap.decision_plot(base_value, shap_interaction_values[[0], :], features=np.array(names), matplotlib=True)

names = features_display.columns.to_list()
args1["feature_names"] = names
args2["feature_names"] = names

# Plot font changes sizes depending on whether an interaction feature is printed.
shap.decision_plot(feature_display_range=slice(None, -11, -1), **args2)
shap.decision_plot(feature_display_range=slice(None, -9, -1), **args2)

# Highlighting by index
highlight = [1, 9]
shap.decision_plot(highlight=highlight, **args1)

# Highlighting by boolean array
predictions = base_value + shap_values.sum(1)
highlight = np.abs(predictions) > 9
shap.decision_plot(highlight=highlight, **args1)
highlight = y_label != (expit(predictions) > 0.5)
shap.decision_plot(highlight=highlight, **args1)

# Highlighting by slice
shap.decision_plot(highlight=slice(0, 10), **args1)

# Logit link
shap.decision_plot(link="logit", **args1)
shap.decision_plot(link="logit", **args2)

# Color scheme
shap.decision_plot(plot_color="coolwarm", **args1)

# Axis color
shap.decision_plot(axis_color="#FF0000", **args1)

# Y feature demarcation color
shap.decision_plot(y_demarc_color="#FF0000", **args1)

# Alpha value
shap.decision_plot(alpha=0.2, **args1)

# Disable color bar
shap.decision_plot(color_bar=False, **args1)
shap.decision_plot(color_bar=False, feature_display_range=slice(-20, None, 1), **args1)

# Disable autosize
shap.decision_plot(auto_size_plot=False, **args1)
shap.decision_plot(auto_size_plot=False, **args2)

# Enable title
shap.decision_plot(title="This doesn't look good", **args1)

# Disable show
shap.decision_plot(show=False, **args1)
pl.show()

# Flip y-axis
shap.decision_plot(feature_display_range=slice(-20, None, 1), **args1)
shap.decision_plot(feature_display_range=slice(-20, None, 1), **args2)
shap.decision_plot(**args2) # to compare with previous plot

# Use xlim
shap.decision_plot(show=False, **args1)
xlim = pl.gca().get_xlim()
pl.show()
shap.decision_plot(base_value, shap_values[11, :], features=features_display, xlim=xlim , matplotlib=True)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# No sorting
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

shap.decision_plot(feature_order="none", **args1)
shap.decision_plot(feature_order="none", feature_display_range=slice(-20, None, 1), **args1)
shap.decision_plot(feature_order="none", **args2)
shap.decision_plot(feature_order="none", feature_display_range=slice(-20, None, 1), **args2)
shap.decision_plot(feature_order=None, **args1)
shap.decision_plot(feature_order=None, **args2)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# Hierarchical cluster sorting
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

shap.decision_plot(feature_order="hclust", **args1)
shap.decision_plot(feature_order="hclust", feature_display_range=slice(-20, None, 1), **args1)
shap.decision_plot(feature_order="hclust", **args2)
shap.decision_plot(feature_order="hclust", feature_display_range=slice(-20, None, 1), **args2)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# Feature display range
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

shap.decision_plot(**args1)
shap.decision_plot(feature_display_range=range(0, 20), show=False, **args1)
xlim = pl.gca().get_xlim()
pl.show()
shap.decision_plot(feature_display_range=range(0, 1), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(0, 2), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(1, 2), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(10, 12), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(11, 9, -1), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(11, 12), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(11, 10, -1), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=range(11, 0, -1), xlim=xlim, **args1)

shap.decision_plot(feature_display_range=slice(1), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=slice(-12, -13, -1), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=slice(0, 2), xlim=xlim, **args1)
shap.decision_plot(feature_display_range=slice(-11, -13, -1), xlim=xlim, **args1)

shap.decision_plot(feature_order='hclust', feature_display_range=slice(None, -21, -1), xlim=xlim, **args2)
shap.decision_plot(feature_order='hclust', feature_display_range=slice(20, None, -1), xlim=xlim, **args2)

# decision_plot transforms negative values in a range so they are interpreted correctly in a slice.
shap.decision_plot(feature_order='hclust', feature_display_range=range(11, -1, -1), xlim=xlim, **args2)
shap.decision_plot(feature_order='hclust', feature_display_range=range(-100, 12, 1), xlim=xlim, **args2)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# Errors
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

#/// TODO finish this using nose stuff. See
# https://stackoverflow.com/questions/11767938/how-to-use-noses-assert-raises
# https://www.programcreek.com/python/example/9481/nose.tools.assert_raises

msg = "shap.decision_plot() missing 2 required positional arguments: 'base_value' and 'shap_values'x"
nt.assert_raises_regexp(TypeError, msg)
nt.assert_raises()

# TypeError:
BaseException()

shap.decision_plot(0, [])
# TypeError: The shap_values arg looks like multi output. Try shap_values[i].

shap.decision_plot(0, {})
# TypeError: The shap_values arg is the wrong type. Try explainer.shap_values().

a = np.ndarray((1001, 1))
shap.decision_plot(base_value, a, a)
# RuntimeError: Plotting 1001 observations may be slow. Consider subsampling or set override_large_data_errors=True to ignore this message.

a = np.random.rand(101, 1000000)
shap.decision_plot(base_value, a, a, matplotlib=True)
# RuntimeError: Processing shap values for 1000000 features over 101 observations may be slow. Set override_large_data_errors=True to ignore this message.

a = np.ndarray((10, 400))
shap.decision_plot(base_value, a, a, max_display=np.Inf)
# RuntimeError: Plotting 400 features may create a very large plot. Set override_large_data_errors=True to ignore this message.

shap.decision_plot(features={}, **args1)
# TypeError: The features arg is an unsupported type.

shap.decision_plot(base_value, shap_values[0, :], ['a'])
# ValueError: The feature_names arg must include all features represented in shap_values.

shap.decision_plot(feature_order='xyz', **args1)
# ValueError: The sort arg requires 'importance', 'hclust', or 'none'.

