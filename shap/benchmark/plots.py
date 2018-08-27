import numpy as np
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
    "tree_gini": {
        "title": "Gain/Gini Importance"
    },
    "mean_abs_tree_shap": {
        "title": "mean(|Tree SHAP|)"
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
