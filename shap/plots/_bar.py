import matplotlib.pyplot as pl
import numpy as np
import scipy

from .. import Cohorts, Explanation
from ..utils import format_value, ordinal_str
from ..utils._exceptions import DimensionError
from . import colors
from ._labels import labels
from ._utils import (
    convert_ordering,
    dendrogram_coords,
    get_sort_order,
    merge_nodes,
    sort_inds,
)


# TODO: improve the bar chart to look better like the waterfall plot with numbers inside the bars when they fit
# TODO: Have the Explanation object track enough data so that we can tell (and so show) how many instances are in each cohort
def bar(shap_values, max_display=10, order=Explanation.abs, clustering=None, clustering_cutoff=0.5,
        merge_cohorts=False, show_data="auto", show=True):
    """Create a bar plot of a set of SHAP values.

    If a single sample is passed, then we plot the SHAP values as a bar chart. If an
    :class:`.Explanation` with many samples is passed, then we plot the mean absolute
    value for each feature column as a bar chart.


    Parameters
    ----------
    shap_values : shap.Explanation or shap.Cohorts or dictionary of shap.Explanation objects
        A single row of a SHAP :class:`.Explanation` object (i.e. ``shap_values[0]``) or
        a multi-row Explanation object that we want to summarize.

    max_display : int
        How many top features to include in the bar plot (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------

    See `bar plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html>`_.

    """

    # assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values parameter must be a shap.Explanation object!"

    # convert Explanation objects to dictionaries
    if isinstance(shap_values, Explanation):
        cohorts = {"": shap_values}
    elif isinstance(shap_values, Cohorts):
        cohorts = shap_values.cohorts
    elif isinstance(shap_values, dict):
        cohorts = shap_values
    else:
        emsg = (
            "The shap_values argument must be an Explanation object, Cohorts "
            "object, or dictionary of Explanation objects!"
        )
        raise TypeError(emsg)

    # unpack our list of Explanation objects we need to plot
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i, exp in enumerate(cohort_exps):
        if not isinstance(exp, Explanation):
            emsg = (
                "The shap_values argument must be an Explanation object, Cohorts "
                "object, or dictionary of Explanation objects!"
            )
            raise TypeError(emsg)

        if len(exp.shape) == 2:
            # collapse the Explanation arrays to be of shape (#features,)
            cohort_exps[i] = exp.abs.mean(0)
        if cohort_exps[i].shape != cohort_exps[0].shape:
            emsg = (
                "When passing several Explanation objects, they must all have "
                "the same number of feature columns!"
            )
            raise DimensionError(emsg)
        # TODO: check other attributes for equality? like feature names perhaps? probably clustering as well.

    # unpack the Explanation object
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    if clustering is None:
        partition_tree = getattr(cohort_exps[0], "clustering", None)
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering
    if partition_tree is not None:
        assert partition_tree.shape[1] == 4, "The clustering provided by the Explanation object does not seem to be a partition tree (which is all shap.plots.bar supports)!"
    op_history = cohort_exps[0].op_history
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    if len(values[0]) == 0:
        raise Exception("The passed Explanation is empty! (so there is nothing to plot)")

    # we show the data on auto only when there are no transforms
    if show_data == "auto":
        show_data = len(op_history) == 0

    # TODO: Rather than just show the "1st token", "2nd token", etc. it would be better to show the "Instance 0's 1st but", etc
    if issubclass(type(feature_names), str):
        feature_names = [ordinal_str(i)+" "+feature_names for i in range(len(values[0]))]

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "SHAP value"
    for op in op_history:
        if op["name"] == "abs":
            xlabel = "|"+xlabel+"|"
        elif op["name"] == "__getitem__":
            pass # no need for slicing to effect our label, it will be used later to find the sizes of cohorts
        else:
            xlabel = str(op["name"])+"("+xlabel+")"

    # find how many instances are in each cohort (if they were created from an Explanation object)
    cohort_sizes = []
    for exp in cohort_exps:
        for op in exp.op_history:
            if op.get("collapsed_instances", False): # see if this if the first op to collapse the instances
                cohort_sizes.append(op["prev_shape"][0])
                break


    # unwrap any pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values[0]))
    max_display = min(max_display, num_features)

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(values[0]))]
    orig_values = values.copy()
    while True:
        feature_order = np.argsort(np.mean([np.argsort(convert_ordering(order, Explanation(values[i]))) for i in range(values.shape[0])], 0))
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values).mean(0))

            # now relax the requirement to match the partition tree ordering for connections above clustering_cutoff
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, clustering_cutoff, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= clustering_cutoff:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values).mean(0), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        else:
            full_print = " + ".join([feature_names[i] for i in inds])
            if len(full_print) <= 40:
                feature_names_new.append(full_print)
            else:
                max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
                feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    # build our y-tick labels
    yticklabels = []
    for i in feature_inds:
        if features is not None and show_data:
            yticklabels.append(format_value(features[i], "%0.03f") + " = " + feature_names[i])
        else:
            yticklabels.append(feature_names[i])
    if num_features < len(values[0]):
        yticklabels[-1] = "Sum of %d other features" % num_cut

    # compute our figure size based on how many features we are showing
    row_height = 0.5
    pl.gcf().set_size_inches(8, num_features * row_height * np.sqrt(len(values)) + 1.5)

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[:,feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    patterns = (None, '\\\\', '++', 'xx', '////', '*', 'o', 'O', '.', '-')
    total_width = 0.7
    bar_width = total_width / len(values)
    for i in range(len(values)):
        ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
        pl.barh(
            y_pos + ypos_offset, values[i,feature_inds],
            bar_width, align='center',
            color=[colors.blue_rgb if values[i,feature_inds[j]] <= 0 else colors.red_rgb for j in range(len(y_pos))],
            hatch=patterns[i], edgecolor=(1,1,1,0.8), label=f"{cohort_labels[i]} [{cohort_sizes[i] if i < len(cohort_sizes) else None}]"
        )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    pl.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=13)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    #xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen/width

    for i in range(len(values)):
        ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
        for j in range(len(y_pos)):
            ind = feature_order[j]
            if values[i,ind] < 0:
                pl.text(
                    values[i,ind] - (5/72)*bbox_to_xscale, y_pos[j] + ypos_offset, format_value(values[i,ind], '%+0.02f'),
                    horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                    fontsize=12
                )
            else:
                pl.text(
                    values[i,ind] + (5/72)*bbox_to_xscale, y_pos[j] + ypos_offset, format_value(values[i,ind], '%+0.02f'),
                    horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                    fontsize=12
                )

    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i+1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    if features is not None:
        features = list(features)

        # try and round off any trailing zeros after the decimal point in the feature values
        for i in range(len(features)):
            try:
                if round(features[i]) == features[i]:
                    features[i] = int(features[i])
            except Exception:
                pass # features[i] must not be a number

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=11)

    xmin,xmax = pl.gca().get_xlim()
    ymin,ymax = pl.gca().get_ylim()

    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.05)

    # if features is None:
    #     pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    # else:
    pl.xlabel(xlabel, fontsize=13)

    if len(values) > 1:
        pl.legend(fontsize=12)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = pl.gca().yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    # draw a dendrogram if we are given a partition tree
    if partition_tree is not None:

        # compute the dendrogram line positions based on our current feature order
        feature_pos = np.argsort(feature_order)
        ylines,xlines = dendrogram_coords(feature_pos, partition_tree)

        # plot the distance cut line above which we don't show tree edges
        xmin,xmax = pl.xlim()
        xlines_min,xlines_max = np.min(xlines),np.max(xlines)
        ct_line_pos = (clustering_cutoff / (xlines_max - xlines_min)) * 0.1 * (xmax - xmin) + xmax
        pl.text(
            ct_line_pos + 0.005 * (xmax - xmin), (ymax - ymin)/2, "Clustering cutoff = " + format_value(clustering_cutoff, '%0.02f'),
            horizontalalignment='left', verticalalignment='center', color="#999999",
            fontsize=12, rotation=-90
        )
        l = pl.axvline(ct_line_pos, color="#dddddd", dashes=(1, 1))
        l.set_clip_on(False)

        for (xline, yline) in zip(xlines, ylines):

            # normalize the x values to fall between 0 and 1
            xv = (np.array(xline) / (xlines_max - xlines_min))

            # only draw if we are not going past distance threshold
            if np.array(xline).max() <= clustering_cutoff:

                # only draw if we are not going past the bottom of the plot
                if yline.max() < max_display:
                    l = pl.plot(
                        xv * 0.1 * (xmax - xmin) + xmax,
                        max_display - np.array(yline),
                        color="#999999"
                    )
                    for v in l:
                        v.set_clip_on(False)

    if show:
        pl.show()



# def compute_sort_counts(partition_tree, leaf_values, pos=None):
#     if pos is None:
#         pos = partition_tree.shape[0]-1

#     M = partition_tree.shape[0] + 1

#     if pos < 0:
#         return 1,leaf_values[pos + M]

#     left = int(partition_tree[pos, 0]) - M
#     right = int(partition_tree[pos, 1]) - M

#     left_val,left_sum = compute_sort_counts(partition_tree, leaf_values, left)
#     right_val,right_sum = compute_sort_counts(partition_tree, leaf_values, right)

#     if left_sum > right_sum:
#         left_val = right_val + 1
#     else:
#         right_val = left_val + 1

#     if left >= 0:
#         partition_tree[left,3] = left_val
#     if right >= 0:
#         partition_tree[right,3] = right_val


#     return max(left_val, right_val) + 1, max(left_sum, right_sum)

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
