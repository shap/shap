import numpy as np
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    pass
from . import colors
from .. import Explanation
from ..utils import OpChain
from ._utils import convert_ordering, convert_color

def heatmap(shap_values, instance_order=Explanation.hclust(), feature_values=Explanation.abs.mean(0), 
            feature_order=None, max_display=10, cmap=colors.red_white_blue, show=True):
    """ Create a heatmap plot of a set of SHAP values.

    This plot is designed to show the population substructure of a dataset using supervised
    clustering and a heatmap. Supervised clustering involves clustering data points not by their original
    feature values but by their explanations. By default we cluster using shap.utils.hclust_ordering
    but any clustering can be used to order the samples.

    Parameters
    ----------
    shap_values : shap.Explanation
        A multi-row Explanation object that we want to visualize in a cluster ordering.
    
    instance_order : OpChain or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct sample ordering given as an numpy.ndarray.

    feature_values : OpChain or numpy.ndarray
        A function that returns a global summary value for each input feature, or an array of such values.

    feature_order : None, OpChain, or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct input feature ordering given as an numpy.ndarray. If None then we use
        feature_values.argsort
        
    max_display : int
        The maximum number of features to display.

    show : bool
        If show is set to False then we don't call the matplotlib.pyplot.show() function. This allows
        further customization of the plot by the caller after the bar() function is finished. 

    """

    # sort the SHAP values matrix by rows and columns
    values = shap_values.values
    if issubclass(type(feature_values), OpChain):
        feature_values = feature_values.apply(Explanation(values))
    if issubclass(type(feature_values), Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), OpChain):
        feature_order = feature_order.apply(Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise TypeError("Unsupported feature_order: %s!" % str(feature_order))
    xlabel = "Instances"
    instance_order = convert_ordering(instance_order, shap_values)
    # if issubclass(type(instance_order), OpChain):
    #     #xlabel += " " + instance_order.summary_string("SHAP values")
    #     instance_order = instance_order.apply(Explanation(values))
    # elif not hasattr(instance_order, "__len__"):
    #     raise Exception("Unsupported instance_order: %s!" % str(instance_order))
    # else:
    #     instance_order_ops = None

    feature_names = np.array(shap_values.feature_names)[feature_order]
    values = shap_values.values[instance_order][:,feature_order]
    feature_values = feature_values[feature_order]

    # collapse
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :max_display-1] = values[:, :max_display-1]
        new_values[:, max_display-1] = values[:, max_display-1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:max_display-1] = feature_values[:max_display-1]
        new_feature_values[max_display-1] = feature_values[max_display-1:].sum()
        feature_names = list(feature_names[:max_display])
        feature_names[-1] = "Sum of %d other features" % (values.shape[1] - max_display + 1)
        values = new_values
        feature_values = new_feature_values
    
    # define the plot size
    row_height = 0.5
    pl.gcf().set_size_inches(8, values.shape[1] * row_height + 2.5)

    # plot the matrix of SHAP values as a heat map
    vmin = np.nanpercentile(values.flatten(), 1)
    vmax = np.nanpercentile(values.flatten(), 99)
    pl.imshow(
        values.T, aspect=0.7 * values.shape[0]/values.shape[1], interpolation="nearest", vmin=min(vmin,-vmax), vmax=max(-vmin,vmax),
        cmap=cmap
    )
    yticks_pos = np.arange(values.shape[1])
    yticks_labels = feature_names

    pl.yticks([-1.5] + list(yticks_pos), ["f(x)"] + list(yticks_labels), fontsize=13)
    
    pl.ylim(values.shape[1]-0.5, -3)
    
    
    
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(True)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['bottom'].set_visible(False)
    pl.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.mean(0)
    pl.plot(-fx/np.abs(fx).max() - 1.5, color="#000000", linewidth=1)
    #pl.colorbar()
    pl.gca().spines['left'].set_bounds(values.shape[1]-0.5, -0.5)
    pl.gca().spines['right'].set_bounds(values.shape[1]-0.5, -0.5)
    b = pl.barh(
        yticks_pos, (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20, 
        0.7, align='center', color="#000000", left=values.shape[0] * 1.0 - 0.5
        #color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    for v in b:
        v.set_clip_on(False)
    pl.xlim(-0.5, values.shape[0]-0.5)
    pl.xlabel(xlabel)
    
    
    
    if True:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array([min(vmin,-vmax), max(-vmin,vmax)])
        cb = pl.colorbar(m, ticks=[min(vmin,-vmax), max(-vmin,vmax)], aspect=1000, fraction=0.0090, pad=0.10,
                        panchor=(0,0.05))
        #cb.set_ticklabels([min(vmin,-vmax), max(-vmin,vmax)])
        cb.set_label("SHAP value", size=12, labelpad=-10)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 15)
        cb.ax.set_anchor((1,0.2))
        #cb.draw_all()
        
    for i in [0]:
        pl.gca().get_yticklines()[i].set_visible(False)
    
    if show:
        pl.show()