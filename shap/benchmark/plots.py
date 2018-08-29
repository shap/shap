import numpy as np
from .experiments import run_experiments
from ..plots import colors
from . import models
from . import methods
import sklearn
try:
    import matplotlib.pyplot as pl
except ImportError:
    pass


labels = {
    "consistency_guarantees": {
        "title": "Consistency Guarantees"
    },
    "local_accuracy": {
        "title": "Local Accuracy"
    },
    "runtime": {
        "title": "Runtime"
    },
    "remove_positive": {
        "title": "Remove Positive",
        "xlabel": "Max fraction of features removed",
        "ylabel": "Negative mean model output"
    },
    "mask_remove_positive": {
        "title": "Mask Remove Positive",
        "xlabel": "Max fraction of features removed",
        "ylabel": "Negative mean model output"
    },
    "remove_negative": {
        "title": "Remove Negative",
        "xlabel": "Max fraction of features removed",
        "ylabel": "Mean model output"
    },
    "mask_remove_negative": {
        "title": "Mask Remove Negative",
        "xlabel": "Max fraction of features removed",
        "ylabel": "Mean model output"
    },
    "keep_positive": {
        "title": "Keep Positive",
        "xlabel": "Max fraction of features kept",
        "ylabel": "Mean model output"
    },
    "mask_keep_positive": {
        "title": "Mask Keep Positive",
        "xlabel": "Max fraction of features kept",
        "ylabel": "Mean model output"
    },
    "keep_negative": {
        "title": "Keep Negative",
        "xlabel": "Max fraction of features kept",
        "ylabel": "Negative mean model output"
    },
    "mask_keep_negative": {
        "title": "Mask Keep Negative",
        "xlabel": "Max fraction of features kept",
        "ylabel": "Negative mean model output"
    },
    "batch_remove_absolute_r2": {
        "title": "Batch Remove Absolute",
        "xlabel": "Fraction of features removed",
        "ylabel": "1 - R^2"
    },
    "batch_keep_absolute_r2": {
        "title": "Batch Keep Absolute",
        "xlabel": "Fraction of features kept",
        "ylabel": "R^2"
    },
    "linear_shap_corr": {
        "title": "Linear SHAP (corr)"
    },
    "linear_shap_ind": {
        "title": "Linear SHAP (ind)"
    },
    "coef": {
        "title": "Coefficents"
    },
    "random": {
        "title": "Random"
    },
    "kernel_shap_1000_meanref": {
        "title": "Kernel SHAP 1000 mean ref."
    },
    "sampling_shap_1000": {
        "title": "Sampling SHAP 1000"
    },
    "tree_shap": {
        "title": "Tree SHAP"
    },
    "saabas": {
        "title": "Saabas"
    },
    "tree_gain": {
        "title": "Gain/Gini Importance"
    },
    "mean_abs_tree_shap": {
        "title": "mean(|Tree SHAP|)"
    },
    "lasso_regression": {
        "title": "Lasso Regression"
    },
    "ridge_regression": {
        "title": "Ridge Regression"
    },
    "gbm_regression": {
        "title": "Gradient Boosting Regression"
    }
}

benchmark_color_map = {
    "Tree SHAP": "#1E88E5",
    "Linear SHAP (corr)": "#1E88E5",
    "Linear SHAP (ind)": "#ff0d57",
    "Coef": "#13B755",
    "Random": "#999999",
    "Const Random": "#666666",
    "Kernel SHAP 1000 mean ref.": "#7C52FF",
    "Kernel SHAP 100 mean ref.": "#7C52FF"
}

negated_metrics = [
    "runtime",
    "remove_positive",
    "mask_remove_positive",
    "keep_negative",
    "mask_keep_negative"
]

one_minus_metrics = [
    "batch_remove_absolute_r2"
]

def plot_curve(metric, fcounts, method_scores, cmap=benchmark_color_map):
    methods = []
    for (name,scores) in method_scores:
        if metric in negated_metrics:
            scores = -scores
        elif metric in one_minus_metrics:
            scores = 1 - scores
        auc = sklearn.metrics.auc(fcounts, scores) / fcounts[-1]
        methods.append((auc, name, scores))
    for (auc,name,scores) in sorted(methods):
        l = "{:6.3f} - ".format(auc) + labels[name]
        pl.plot(fcounts / fcounts[-1], scores, label=l, color=cmap.get(name, "#000000"), linewidth=2)
    pl.xlabel(labels[metric]["xlabel"])
    pl.ylabel(labels[metric]["ylabel"])
    pl.title(labels[metric]["title"])
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    ahandles, alabels = pl.gca().get_legend_handles_labels()
    pl.legend(reversed(ahandles), reversed(alabels))
    return pl.gcf()

def make_grid(scores, dataset, model):
    color_vals = {}
    for (_,_,method,metric),(fcounts,score) in filter(lambda x: x[0][0] == dataset and x[0][1] == model, scores):
        if metric not in color_vals:
            color_vals[metric] = {}

        if metric in negated_metrics:
            score = -score
        elif metric in one_minus_metrics:
            score = 1 - score

        if fcounts is None:
            color_vals[metric][method] = score
        else:
            auc = sklearn.metrics.auc(fcounts, score) / fcounts[-1]
            color_vals[metric][method] = auc
    
    col_keys = list(color_vals.keys())
    row_keys = list(set([v for k in col_keys for v in color_vals[k].keys()]))
    
    data = -28567 * np.ones((len(row_keys), len(col_keys)))
    
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            data[i,j] = color_vals[col_keys[j]][row_keys[i]]
            
    assert np.sum(data == -28567) == 0, "There are missing data values!"
            
    data = (data - data.min(0)) / (data.max(0) - data.min(0))
    
    # sort by performans
    inds = np.argsort(-data.mean(1))
    row_keys = [row_keys[i] for i in inds]
    data = data[inds,:]
    
    return row_keys, col_keys, data

from matplotlib.colors import LinearSegmentedColormap
red_blue_solid = LinearSegmentedColormap('red_blue_solid', {
    'red': ((0.0, 198./255, 198./255),
            (1.0, 5./255, 5./255)),

    'green': ((0.0, 34./255, 34./255),
              (1.0, 198./255, 198./255)),

    'blue': ((0.0, 5./255, 5./255),
             (1.0, 24./255, 24./255)),

    'alpha': ((0.0, 1, 1),
              (1.0, 1, 1))
})
from IPython.core.display import HTML
def plot_grids(dataset, model_names):

    scores = []
    for model in model_names:
        scores.extend(run_experiments(dataset=dataset, model=model))
    
    out = "" # background: rgb(30, 136, 229)
    out += "<div style='font-weight: regular; font-size: 24px; text-align: center; background: #f8f8f8; color: #000; padding: 20px;'>SHAP Benchmark</div>"
    out += "<div style='height: 1px; background: #ddd;'></div>"
    #out += "<div style='height: 7px; background-image: linear-gradient(to right, rgb(30, 136, 229), rgb(255, 13, 87));'></div>"
    out += "<table style='border-width: 1px; font-size: 14px; margin-left: 40px'>"
    for ind,model in enumerate(model_names):
        row_keys, col_keys, data = make_grid(scores, dataset, model)
        
        model_title = getattr(models, dataset+"__"+model).__doc__.split("\n")[0].strip()

        if ind == 0:
            out += "<tr><td style='background: #fff'></td></td>"
            for j in range(data.shape[1]):
                metric_title = labels[col_keys[j]]["title"]
                out += "<td style='width: 40px; background: #fff'><div style='margin-bottom: -5px; white-space: nowrap; transform: rotate(-45deg); transform-origin: left top 0; width: 1.5em; margin-top: 8em'>" + metric_title + "</div></td>"
            out += "</tr>"
        out += "<tr><td style='background: #fff'></td><td colspan='%d' style='background: #fff; font-weight: bold; text-align: center'>%s</td></tr>" % (data.shape[1], model_title)
        for i in range(data.shape[0]):
            out += "<tr>"
#             if i == 0:
#                 out += "<td rowspan='%d' style='background: #fff; text-align: center; white-space: nowrap; vertical-align: middle; '><div style='font-weight: bold; transform: rotate(-90deg); transform-origin: left top 0; width: 1.5em; margin-top: 8em'>%s</div></td>" % (data.shape[0], model_name)
            method_title = getattr(methods, row_keys[i]).__doc__.split("\n")[0].strip()
            out += "<td style='background: #ffffff' title='shap.LinearExplainer(model)'>" + method_title + "</td>"
            for j in range(data.shape[1]):
                out += "<td style='width: 40px; height: 40px; background-color: rgb" + str(tuple(v*255 for v in colors.red_blue_solid(data[i,j])[:-1])) + "'></td>"
            out += "</tr>"
            
        out += "<tr><td colspan='%d' style='background: #fff'></td>" % (data.shape[1] + 1)
    out += "</table>"
    
    return HTML(out)