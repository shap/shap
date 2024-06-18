import matplotlib.pyplot as pl
import numpy as np

from .. import Explanation
from ..utils import OpChain
from . import colors
from ._labels import labels
from ._utils import convert_ordering

no_plotly = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except:
    no_plotly = True

def heatmap(shap_values, instance_order=Explanation.hclust(), feature_values=Explanation.abs.mean(0),
            feature_order=None, max_display=10, cmap=None, show=True,
            plot_width=8, ax=None, rendering_engine='matplotlib'):
    """Create a heatmap plot of a set of SHAP values.

    This plot is designed to show the population substructure of a dataset using supervised
    clustering and a heatmap.
    Supervised clustering involves clustering data points not by their original
    feature values but by their explanations.
    By default, we cluster using :func:`shap.utils.hclust_ordering`,
    but any clustering can be used to order the samples.

    Parameters
    ----------
    shap_values : shap.Explanation
        A multi-row :class:`.Explanation` object that we want to visualize in a
        cluster ordering.

    instance_order : OpChain or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct sample ordering given as an ``numpy.ndarray``.

    feature_values : OpChain or numpy.ndarray
        A function that returns a global summary value for each input feature, or an array of such values.

    feature_order : None, OpChain, or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct input feature ordering given as an ``numpy.ndarray``.
        If ``None``, then we use ``feature_values.argsort``.

    max_display : int
        The maximum number of features to display (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    plot_width : int, default 8
        The width of the heatmap plot.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.
        
    rendering_engine : str
        Plot framework used to render the plot. Any of 'matplotlib' (default) or 'plotly'        

    Returns
    -------
 	ax: matplotlib Axes
        Returns the Axes object with the plot drawn onto it. Only returned if 
		``rendering_engine=="matplotlib"``.
    fig: Plotly Figure
        Returns the Figure object with the plot drawn onto it. Only returned if 
		``rendering_engine=="plotly"``.
    Examples
    --------
    See `heatmap plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html>`_.

    """
    if no_plotly and rendering_engine.lower() == 'plotly':
        raise ValueError('Plotly must be installed prior to using it as a rendering engine.')

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
        raise Exception(f"Unsupported feature_order: {str(feature_order)}!")
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

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display-1]
        new_values[:, -1] = values[:, max_display-1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display-1]
        new_feature_values[-1] = feature_values[max_display-1:].sum()
        feature_names = [
            *feature_names[:max_display-1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    if rendering_engine == 'plotly':
        if cmap is None: 
            cmap = [[0, 'rgb(255,0,128)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(0,128,255)']]
            
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0, horizontal_spacing=0,
                            shared_xaxes=True, shared_yaxes=True, 
                            column_widths=[0.9, 0.1], row_heights=[0.1, 0.9], )
        
        # plot the matrix of SHAP values as a heat map
        vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
        zmin = min(vmin,-vmax)
        zmax=max(-vmin,vmax)
        fig.add_trace(go.Heatmap(z=values.T, zmin=zmin, zmax=zmax, colorscale=cmap, reversescale=True,
                                 colorbar={'thickness':10, 'tickmode':'array', 'tickvals':[zmin, zmax]}), 
                      row=2, col=1)

        # plot the f(x) line chart above the heat map
        fx = values.T.mean(0)
        fig.add_trace(go.Scatter(y=-fx / np.abs(fx).max() - 1.5, mode="lines", line={'color':'black', 'width':1}), row=1, col=1)        
        fig.add_hline(y=-1.5, line_color='#aaaaaa', line_dash='dash', line_width=0.5, row=1, col=1)
        
        # plot the bar plot on the right spine of the heat map
        heatmap_yticks_pos = np.arange(values.shape[1])
        bar_sizes = (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20
        fig.add_trace(go.Bar(x=bar_sizes, y=heatmap_yticks_pos, width=0.7, marker_color="black",
                             orientation='h'), row=2, col=2)
        fig.update_layout(plot_bgcolor="white", showlegend=False)

        # adjust the axes ticks and spines for the heat map + f(x) line chart
        fig['layout']['yaxis'].update(title='f(x)', showticklabels=False)
        fig['layout']['xaxis3'].update(title=xlabel)
        fig['layout']['xaxis4'].update(showticklabels=False)
        fig['layout']['yaxis3'].update(tickmode='array', tickvals=list(range(len(feature_names))), ticktext=feature_names)
        fig.update_yaxes(autorange="reversed")
        
        # Colorbar title
        fig.update_layout(annotations=[{'text':'SHAP Value', 'textangle':-90, 'showarrow':False, 
                                        'xref':'paper', 'yref':'paper', 'x':1.025, 'y':0.5, 'xshift':40}])
                
        if show:
            fig.show()
            
        return fig
    else:
        if cmap is None: 
            cmap= colors.red_white_blue

        row_height = 0.5
        if ax is None:
            pl.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
            ax = pl.gca()

        # plot the matrix of SHAP values as a heat map
        vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
        ax.imshow(
            values.T,
            aspect=0.7 * values.shape[0] / values.shape[1],
            interpolation="nearest",
            vmin=min(vmin,-vmax),
            vmax=max(-vmin,vmax),
            cmap=cmap,
        )
    
        # adjust the axes ticks and spines for the heat map + f(x) line chart
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines[["left", "right"]].set_visible(True)
        ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
        ax.spines[["top", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", direction="out")
    
        ax.set_ylim(values.shape[1] - row_height, -3)
        heatmap_yticks_pos = np.arange(values.shape[1])
        heatmap_yticks_labels = feature_names
        ax.yaxis.set_ticks(
            [-1.5, *heatmap_yticks_pos],
            [r"$f(x)$", *heatmap_yticks_labels],
            fontsize=13,
        )
        # remove the y-tick line for the f(x) label
        ax.yaxis.get_ticklines()[0].set_visible(False)
    
        ax.set_xlim(-0.5, values.shape[0] - 0.5)
        ax.set_xlabel(xlabel)
    
        # plot the f(x) line chart above the heat map
        ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
        fx = values.T.mean(0)
        ax.plot(
            -fx / np.abs(fx).max() - 1.5,
            color="#000000",
            linewidth=1,
        )
    
        # plot the bar plot on the right spine of the heat map
        bar_container = ax.barh(
            heatmap_yticks_pos,
            (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
            height=0.7,
            align="center",
            color="#000000",
            left=values.shape[0] * 1.0 - 0.5,
            # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
        )
        for b in bar_container:
            b.set_clip_on(False)
    
        # draw the color bar
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap)
        m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
        cb = pl.colorbar(
            m,
            ticks=[min(vmin, -vmax), max(-vmin, vmax)],
            ax=ax,
            aspect=80,
            fraction=0.01,
            pad=0.10,  # padding between the cb and the main axes
        )
        cb.set_label(labels["VALUE"], size=12, labelpad=-10)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        # cb.ax.set_aspect((bbox.height - 0.9) * 15)
        # cb.draw_all()
    
        if show:
            pl.show()

        return ax
