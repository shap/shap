import numpy as np
import sklearn
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    pass
from . import labels
from . import colors
from ..common import convert_name

def embedding_plot(ind, shap_values, feature_names=None, method="pca", alpha=1.0):
    """ Use the SHAP values as an embedding which we project to 2D for visualization.

    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to use to color the embedding.
        If this is a string it is either the name of the feature, or it can have the
        form "rank(int)" to specify the feature with that rank (ordered by mean absolute
        SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features). It can also be a pre-projected
        array that has already been reduced to 2D, in which case the method argument
        will be ignored and the provided projection will be used directly.

    feature_names : None or list
        The names of the features in the shap_values array.

    method : "pca"
        How to reduce the dimensions of the shap_values to 2D. Currently only "pca" is
        supported.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.
    """
    
    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
    
    ind = convert_name(ind, shap_values, feature_names)
    
    # see if we need to compute the embedding
    if shap_values.shape[1] != 2:
        if method == "pca":
            pca = sklearn.decomposition.PCA(2)
            embedding_values = pca.fit_transform(shap_values)
        else:
            print("Unsupported embedding method:", method)
    else:
        embedding_values = shap_values

    pl.scatter(
        embedding_values[:,0], embedding_values[:,1], c=shap_values[:,ind],
        cmap=colors.red_blue_solid, alpha=alpha, linewidth=0
    )
    pl.axis("off")
    #pl.title(feature_names[ind])
    cb = pl.colorbar()
    cb.set_label("SHAP value for\n"+feature_names[ind], size=13)
    cb.outline.set_visible(False)
    
    
    pl.gcf().set_size_inches(7.5, 5)
    bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 10)
    cb.set_alpha(1)
    pl.show()
    