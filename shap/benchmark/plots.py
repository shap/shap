import base64
import io
import os

import numpy as np
import sklearn
from matplotlib.colors import LinearSegmentedColormap

from .. import __version__
from ..plots import colors
from . import methods, metrics, models
from .experiments import run_experiments

try:
    import matplotlib
    import matplotlib.pyplot as pl
    from IPython.display import HTML
except ImportError:
    pass


metadata = {
    # "runtime": {
    #     "title": "Runtime",
    #     "sort_order": 1
    # },
    # "local_accuracy": {
    #     "title": "Local Accuracy",
    #     "sort_order": 2
    # },
    # "consistency_guarantees": {
    #     "title": "Consistency Guarantees",
    #     "sort_order": 3
    # },
    # "keep_positive_mask": {
    #     "title": "Keep Positive (mask)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Mean model output",
    #     "sort_order": 4
    # },
    # "keep_negative_mask": {
    #     "title": "Keep Negative (mask)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 5
    # },
    # "keep_absolute_mask__r2": {
    #     "title": "Keep Absolute (mask)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "R^2",
    #     "sort_order": 6
    # },
    # "keep_absolute_mask__roc_auc": {
    #     "title": "Keep Absolute (mask)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "ROC AUC",
    #     "sort_order": 6
    # },
    # "remove_positive_mask": {
    #     "title": "Remove Positive (mask)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 7
    # },
    # "remove_negative_mask": {
    #     "title": "Remove Negative (mask)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Mean model output",
    #     "sort_order": 8
    # },
    # "remove_absolute_mask__r2": {
    #     "title": "Remove Absolute (mask)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "1 - R^2",
    #     "sort_order": 9
    # },
    # "remove_absolute_mask__roc_auc": {
    #     "title": "Remove Absolute (mask)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "1 - ROC AUC",
    #     "sort_order": 9
    # },
    # "keep_positive_resample": {
    #     "title": "Keep Positive (resample)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Mean model output",
    #     "sort_order": 10
    # },
    # "keep_negative_resample": {
    #     "title": "Keep Negative (resample)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 11
    # },
    # "keep_absolute_resample__r2": {
    #     "title": "Keep Absolute (resample)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "R^2",
    #     "sort_order": 12
    # },
    # "keep_absolute_resample__roc_auc": {
    #     "title": "Keep Absolute (resample)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "ROC AUC",
    #     "sort_order": 12
    # },
    # "remove_positive_resample": {
    #     "title": "Remove Positive (resample)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 13
    # },
    # "remove_negative_resample": {
    #     "title": "Remove Negative (resample)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Mean model output",
    #     "sort_order": 14
    # },
    # "remove_absolute_resample__r2": {
    #     "title": "Remove Absolute (resample)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "1 - R^2",
    #     "sort_order": 15
    # },
    # "remove_absolute_resample__roc_auc": {
    #     "title": "Remove Absolute (resample)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "1 - ROC AUC",
    #     "sort_order": 15
    # },
    # "remove_positive_retrain": {
    #     "title": "Remove Positive (retrain)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 11
    # },
    # "remove_negative_retrain": {
    #     "title": "Remove Negative (retrain)",
    #     "xlabel": "Max fraction of features removed",
    #     "ylabel": "Mean model output",
    #     "sort_order": 12
    # },
    # "keep_positive_retrain": {
    #     "title": "Keep Positive (retrain)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Mean model output",
    #     "sort_order": 6
    # },
    # "keep_negative_retrain": {
    #     "title": "Keep Negative (retrain)",
    #     "xlabel": "Max fraction of features kept",
    #     "ylabel": "Negative mean model output",
    #     "sort_order": 7
    # },
    # "batch_remove_absolute__r2": {
    #     "title": "Batch Remove Absolute",
    #     "xlabel": "Fraction of features removed",
    #     "ylabel": "1 - R^2",
    #     "sort_order": 13
    # },
    # "batch_keep_absolute__r2": {
    #     "title": "Batch Keep Absolute",
    #     "xlabel": "Fraction of features kept",
    #     "ylabel": "R^2",
    #     "sort_order": 8
    # },
    # "batch_remove_absolute__roc_auc": {
    #     "title": "Batch Remove Absolute",
    #     "xlabel": "Fraction of features removed",
    #     "ylabel": "1 - ROC AUC",
    #     "sort_order": 13
    # },
    # "batch_keep_absolute__roc_auc": {
    #     "title": "Batch Keep Absolute",
    #     "xlabel": "Fraction of features kept",
    #     "ylabel": "ROC AUC",
    #     "sort_order": 8
    # },

    # "linear_shap_corr": {
    #     "title": "Linear SHAP (corr)"
    # },
    # "linear_shap_ind": {
    #     "title": "Linear SHAP (ind)"
    # },
    # "coef": {
    #     "title": "Coefficients"
    # },
    # "random": {
    #     "title": "Random"
    # },
    # "kernel_shap_1000_meanref": {
    #     "title": "Kernel SHAP 1000 mean ref."
    # },
    # "sampling_shap_1000": {
    #     "title": "Sampling SHAP 1000"
    # },
    # "tree_shap_tree_path_dependent": {
    #     "title": "Tree SHAP"
    # },
    # "saabas": {
    #     "title": "Saabas"
    # },
    # "tree_gain": {
    #     "title": "Gain/Gini Importance"
    # },
    # "mean_abs_tree_shap": {
    #     "title": "mean(|Tree SHAP|)"
    # },
    # "lasso_regression": {
    #     "title": "Lasso Regression"
    # },
    # "ridge_regression": {
    #     "title": "Ridge Regression"
    # },
    # "gbm_regression": {
    #     "title": "Gradient Boosting Regression"
    # }
}

benchmark_color_map = {
    "tree_shap": "#1E88E5",
    "deep_shap": "#1E88E5",
    "linear_shap_corr": "#1E88E5",
    "linear_shap_ind": "#ff0d57",
    "coef": "#13B755",
    "random": "#999999",
    "const_random": "#666666",
    "kernel_shap_1000_meanref": "#7C52FF"
}

# negated_metrics = [
#     "runtime",
#     "remove_positive_retrain",
#     "remove_positive_mask",
#     "remove_positive_resample",
#     "keep_negative_retrain",
#     "keep_negative_mask",
#     "keep_negative_resample"
# ]

# one_minus_metrics = [
#     "remove_absolute_mask__r2",
#     "remove_absolute_mask__roc_auc",
#     "remove_absolute_resample__r2",
#     "remove_absolute_resample__roc_auc"
# ]

def get_method_color(method):
    for line in getattr(methods, method).__doc__.split("\n"):
        line = line.strip()
        if line.startswith("color = "):
            v = line.split("=")[1].strip()
            if v.startswith("red_blue_circle("):
                return colors.red_blue_circle(float(v[16:-1]))
            else:
                return v
    return "#000000"

def get_method_linestyle(method):
    for line in getattr(methods, method).__doc__.split("\n"):
        line = line.strip()
        if line.startswith("linestyle = "):
            return line.split("=")[1].strip()
    return "solid"

def get_metric_attr(metric, attr):
    for line in getattr(metrics, metric).__doc__.split("\n"):
        line = line.strip()

        # string
        prefix = attr+" = \""
        suffix = "\""
        if line.startswith(prefix) and line.endswith(suffix):
            return line[len(prefix):-len(suffix)]

        # number
        prefix = attr+" = "
        if line.startswith(prefix):
            return float(line[len(prefix):])
    return ""

def plot_curve(dataset, model, metric, cmap=benchmark_color_map):
    experiments = run_experiments(dataset=dataset, model=model, metric=metric)
    pl.figure()
    method_arr = []
    for (name,(fcounts,scores)) in experiments:
        _,_,method,_ = name
        transform = get_metric_attr(metric, "transform")
        if transform == "negate":
            scores = -scores
        elif transform == "one_minus":
            scores = 1 - scores
        auc = sklearn.metrics.auc(fcounts, scores) / fcounts[-1]
        method_arr.append((auc, method, scores))
    for (auc,method,scores) in sorted(method_arr):
        method_title = getattr(methods, method).__doc__.split("\n")[0].strip()
        label = f"{auc:6.3f} - " + method_title
        pl.plot(
            fcounts / fcounts[-1], scores, label=label,
            color=get_method_color(method), linewidth=2,
            linestyle=get_method_linestyle(method)
            )
    metric_title = getattr(metrics, metric).__doc__.split("\n")[0].strip()
    pl.xlabel(get_metric_attr(metric, "xlabel"))
    pl.ylabel(get_metric_attr(metric, "ylabel"))
    model_title = getattr(models, dataset+"__"+model).__doc__.split("\n")[0].strip()
    pl.title(metric_title + " - " + model_title)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    ahandles, alabels = pl.gca().get_legend_handles_labels()
    pl.legend(reversed(ahandles), reversed(alabels))
    return pl.gcf()

def plot_human(dataset, model, metric, cmap=benchmark_color_map):
    experiments = run_experiments(dataset=dataset, model=model, metric=metric)
    pl.figure()
    method_arr = []
    for (name,(fcounts,scores)) in experiments:
        _,_,method,_ = name
        diff_sum = np.sum(np.abs(scores[1] - scores[0]))
        method_arr.append((diff_sum, method, scores[0], scores[1]))

    inds = np.arange(3)    # the x locations for the groups
    inc_width = (1.0 / len(method_arr)) * 0.8
    width = inc_width * 0.9
    pl.bar(inds, method_arr[0][2], width, label="Human Consensus", color="black", edgecolor="white")
    i = 1
    line_style_to_hatch = {
        "dashed": "///",
        "dotted": "..."
    }
    for (diff_sum, method, _, methods_attrs) in sorted(method_arr):
        method_title = getattr(methods, method).__doc__.split("\n")[0].strip()
        label = f"{diff_sum:.2f} - " + method_title
        pl.bar(
            inds + inc_width * i, methods_attrs.flatten(), width, label=label, edgecolor="white",
            color=get_method_color(method), hatch=line_style_to_hatch.get(get_method_linestyle(method), None)
        )
        i += 1
    metric_title = getattr(metrics, metric).__doc__.split("\n")[0].strip()
    pl.xlabel("Features in the model")
    pl.ylabel("Feature attribution value")
    model_title = getattr(models, dataset+"__"+model).__doc__.split("\n")[0].strip()
    pl.title(metric_title + " - " + model_title)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    ahandles, alabels = pl.gca().get_legend_handles_labels()
    #pl.legend(ahandles, alabels)
    pl.xticks(np.array([0, 1, 2, 3]) - (inc_width + width)/2, ["", "", "", ""])

    pl.gca().xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.4, 1.4, 2.4]))
    pl.gca().xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(["Fever", "Cough", "Headache"]))
    pl.gca().tick_params(which='minor', length=0)

    pl.axhline(0, color="#aaaaaa", linewidth=0.5)

    box = pl.gca().get_position()
    pl.gca().set_position([
        box.x0, box.y0 + box.height * 0.3,
        box.width, box.height * 0.7
    ])

    # Put a legend below current axis
    pl.gca().legend(ahandles, alabels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    return pl.gcf()

def _human_score_map(human_consensus, methods_attrs):
    """Converts human agreement differences to numerical scores for coloring."""
    v = 1 - min(np.sum(np.abs(methods_attrs - human_consensus)) / (np.abs(human_consensus).sum() + 1), 1.0)
    return v

def make_grid(scores, dataset, model, normalize=True, transform=True):
    color_vals = {}
    metric_sort_order = {}
    for (_,_,method,metric),(fcounts,score) in filter(lambda x: x[0][0] == dataset and x[0][1] == model, scores):
        metric_sort_order[metric] = get_metric_attr(metric, "sort_order")
        if metric not in color_vals:
            color_vals[metric] = {}

        if transform:
            transform_type = get_metric_attr(metric, "transform")
            if transform_type == "negate":
                score = -score
            elif transform_type == "one_minus":
                score = 1 - score
            elif transform_type == "negate_log":
                score = -np.log10(score)

        if fcounts is None:
            color_vals[metric][method] = score
        elif fcounts == "human":
            color_vals[metric][method] = _human_score_map(*score)
        else:
            auc = sklearn.metrics.auc(fcounts, score) / fcounts[-1]
            color_vals[metric][method] = auc
    # print(metric_sort_order)
    # col_keys = sorted(list(color_vals.keys()), key=lambda v: metric_sort_order[v])
    # print(col_keys)
    col_keys = list(color_vals.keys())
    row_keys = list({v for k in col_keys for v in color_vals[k].keys()})

    data = -28567 * np.ones((len(row_keys), len(col_keys)))

    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            data[i,j] = color_vals[col_keys[j]][row_keys[i]]

    assert np.sum(data == -28567) == 0, "There are missing data values!"

    if normalize:
        data = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-8)

    # sort by performans
    inds = np.argsort(-data.mean(1))
    row_keys = [row_keys[i] for i in inds]
    data = data[inds,:]

    return row_keys, col_keys, data



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
def plot_grids(dataset, model_names, out_dir=None):

    if out_dir is not None:
        os.mkdir(out_dir)

    scores = []
    for model in model_names:
        scores.extend(run_experiments(dataset=dataset, model=model))

    prefix = "<style type='text/css'> .shap_benchmark__select:focus { outline-width: 0 }</style>"
    out = "" # background: rgb(30, 136, 229)

    # out += "<div style='font-weight: regular; font-size: 24px; text-align: center; background: #f8f8f8; color: #000; padding: 20px;'>SHAP Benchmark</div>\n"
    # out += "<div style='height: 1px; background: #ddd;'></div>\n"
    #out += "<div style='height: 7px; background-image: linear-gradient(to right, rgb(30, 136, 229), rgb(255, 13, 87));'></div>"

    out += "<div style='position: fixed; left: 0px; top: 0px; right: 0px; height: 230px; background: #fff;'>\n" # box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    out += "<div style='position: absolute; bottom: 0px; left: 0px; right: 0px;' align='center'><table style='border-width: 1px; margin-right: 100px'>\n"
    for ind,model in enumerate(model_names):
        row_keys, col_keys, data = make_grid(scores, dataset, model)
#         print(data)
#         print(colors.red_blue_solid(0.))
#         print(colors.red_blue_solid(1.))
#         return
        for metric in col_keys:
            save_plot = False
            if metric.startswith("human_"):
                plot_human(dataset, model, metric)
                save_plot = True
            elif metric not in ["local_accuracy", "runtime", "consistency_guarantees"]:
                plot_curve(dataset, model, metric)
                save_plot = True

            if save_plot:
                buf = io.BytesIO()
                pl.gcf().set_size_inches(1200.0/175,1000.0/175)
                pl.savefig(buf, format='png', dpi=175)
                if out_dir is not None:
                    pl.savefig(f"{out_dir}/plot_{dataset}_{model}_{metric}.pdf", format='pdf')
                pl.close()
                buf.seek(0)
                data_uri = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
                plot_id = "plot__"+dataset+"__"+model+"__"+metric
                prefix += f"<div onclick='document.getElementById(\"{plot_id}\").style.display = \"none\"' style='display: none; position: fixed; z-index: 10000; left: 0px; right: 0px; top: 0px; bottom: 0px; background: rgba(255,255,255,0.9);' id='{plot_id}'>"
                prefix += "<img width='600' height='500' style='margin-left: auto; margin-right: auto; margin-top: 230px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);' src='data:image/png;base64,%s'>" % data_uri
                prefix += "</div>"

        model_title = getattr(models, dataset+"__"+model).__doc__.split("\n")[0].strip()

        if ind == 0:
            out += "<tr><td style='background: #fff; width: 250px'></td></td>"
            for j in range(data.shape[1]):
                metric_title = getattr(metrics, col_keys[j]).__doc__.split("\n")[0].strip()
                out += "<td style='width: 40px; min-width: 40px; background: #fff; text-align: right;'><div style='margin-left: 10px; margin-bottom: -5px; white-space: nowrap; transform: rotate(-45deg); transform-origin: left top 0; width: 1.5em; margin-top: 8em'>" + metric_title + "</div></td>"
            out += "</tr>\n"
            out += "</table></div></div>\n"
            out += "<table style='border-width: 1px; margin-right: 100px; margin-top: 230px;'>\n"
        out += "<tr><td style='background: #fff'></td><td colspan='%d' style='background: #fff; font-weight: bold; text-align: center; margin-top: 10px;'>%s</td></tr>\n" % (data.shape[1], model_title)
        for i in range(data.shape[0]):
            out += "<tr>"
#             if i == 0:
#                 out += "<td rowspan='%d' style='background: #fff; text-align: center; white-space: nowrap; vertical-align: middle; '><div style='font-weight: bold; transform: rotate(-90deg); transform-origin: left top 0; width: 1.5em; margin-top: 8em'>%s</div></td>" % (data.shape[0], model_name)
            method_title = getattr(methods, row_keys[i]).__doc__.split("\n")[0].strip()
            out += "<td style='background: #ffffff; text-align: right; width: 250px' title='shap.LinearExplainer(model)'>" + method_title + "</td>\n"
            for j in range(data.shape[1]):
                plot_id = "plot__"+dataset+"__"+model+"__"+col_keys[j]
                out += "<td onclick='document.getElementById(\"%s\").style.display = \"block\"' style='padding: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px solid #999; width: 42px; min-width: 42px; height: 34px; background-color: #fff'>" % plot_id
                #out += "<div style='opacity: "+str(2*(max(1-data[i,j], data[i,j])-0.5))+"; background-color: rgb" + str(tuple(v*255 for v in colors.red_blue_solid(0. if data[i,j] < 0.5 else 1.)[:-1])) + "; height: "+str((30*max(1-data[i,j], data[i,j])))+"px; margin-left: auto; margin-right: auto; width:"+str((30*max(1-data[i,j], data[i,j])))+"px'></div>"
                out += "<div style='opacity: "+str(1)+"; background-color: rgb" + str(tuple(int(v*255) for v in colors.red_blue_no_bounds(5*(data[i,j]-0.8))[:-1])) + "; height: "+str(30*data[i,j])+"px; margin-left: auto; margin-right: auto; width:"+str(30*data[i,j])+"px'></div>"
                #out += "<div style='float: left; background-color: #eee; height: 10px; width: "+str((40*(1-data[i,j])))+"px'></div>"
                out += "</td>\n"
            out += "</tr>\n" #

        out += "<tr><td colspan='%d' style='background: #fff'></td></tr>" % (data.shape[1] + 1)
    out += "</table>"

    out += "<div style='position: fixed; left: 0px; top: 0px; right: 0px; text-align: left; padding: 20px; text-align: right'>\n"
    out += "<div style='float: left; font-weight: regular; font-size: 24px; color: #000;'>SHAP Benchmark <span style='font-size: 14px; color: #777777;'>v"+__version__+"</span></div>\n"
# select {
#   margin: 50px;
#   width: 150px;
#   padding: 5px 35px 5px 5px;
#   font-size: 16px;
#   border: 1px solid #ccc;
#   height: 34px;
#   -webkit-appearance: none;
#   -moz-appearance: none;
#   appearance: none;
#   background: url(http://www.stackoverflow.com/favicon.ico) 96% / 15% no-repeat #eee;
# }
    #out += "<div style='display: inline-block; margin-right: 20px; font-weight: normal; text-decoration: none; font-size: 18px; color: #000;'>Dataset:</div>\n"

    out += "<select id='shap_benchmark__select' onchange=\"document.location = '../' + this.value + '/index.html'\"dir='rtl' class='shap_benchmark__select' style='font-weight: normal; font-size: 20px; color: #000; padding: 10px; background: #fff; border: 1px solid #fff; -webkit-appearance: none; appearance: none;'>\n"
    out += "<option value='human' "+("selected" if dataset == "human" else "")+">Agreement with Human Intuition</option>\n"
    out += "<option value='corrgroups60' "+("selected" if dataset == "corrgroups60" else "")+">Correlated Groups 60 Dataset</option>\n"
    out += "<option value='independentlinear60' "+("selected" if dataset == "independentlinear60" else "")+">Independent Linear 60 Dataset</option>\n"
    #out += "<option>CRIC</option>\n"
    out += "</select>\n"
    #out += "<script> document.onload = function() { document.getElementById('shap_benchmark__select').value = '"+dataset+"'; }</script>"
    #out += "<div style='display: inline-block; margin-left: 20px; font-weight: normal; text-decoration: none; font-size: 18px; color: #000;'>CRIC</div>\n"
    out += "</div>\n"

    # output the legend
    out += "<table style='border-width: 0px; width: 100px; position: fixed; right: 50px; top: 200px; background: rgba(255, 255, 255, 0.9)'>\n"
    out += "<tr><td style='background: #fff; font-weight: normal; text-align: center'>Higher score</td></tr>\n"
    legend_size = 21
    for i in range(legend_size-9):
        out += "<tr>"
        out += "<td style='padding: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px solid #999; height: 34px'>"
        val = (legend_size-i-1) / (legend_size-1)
        out += "<div style='opacity: 1; background-color: rgb" + str(tuple(int(v*255) for v in colors.red_blue_no_bounds(5*(val-0.8)))[:-1]) + "; height: "+str(30*val)+"px; margin-left: auto; margin-right: auto; width:"+str(30*val)+"px'></div>"
        out += "</td>"
        out += "</tr>\n" #
    out += "<tr><td style='background: #fff; font-weight: normal; text-align: center'>Lower score</td></tr>\n"
    out += "</table>\n"

    if out_dir is not None:
        with open(out_dir + "/index.html", "w") as f:
            f.write("<html><body style='margin: 0px; font-size: 16px; font-family: \"Myriad Pro\", Arial, sans-serif;'><center>")
            f.write(prefix)
            f.write(out)
            f.write("</center></body></html>")
    else:
        return HTML(prefix + out)
