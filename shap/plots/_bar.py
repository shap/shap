import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from ._labels import labels
from ..utils import format_value, ordinal_str
from . import colors
import numpy as np
import scipy
import copy
from .. import Explanation


# TODO: improve the bar chart to look better like the waterfall plot with numbers inside the bars when they fit
def bar(shap_values, max_display=10, order=Explanation.abs.argsort, clustering=None, cluster_threshold=0.5, show=True):
    """ Create a bar plot of a set of SHAP values.

    If a single sample is passed then we plot the SHAP values as a bar chart. If an
    Explanation with many samples is passed then we plot the mean absolute value for
    each feature column as a bar chart.


    Parameters
    ----------
    shap_values : shap.Explanation
        A single row of a SHAP Explanation object (i.e. shap_values[0]) or a multi-row Explanation
        object that we want to summarize.

    max_display : int
        The maximum number of bars to display.

    show : bool
        If show is set to False then we don't call the matplotlib.pyplot.show() function. This allows
        further customization of the plot by the caller after the bar() function is finished. 

    """

    assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values paramemter must be a shap.Explanation object!"

    if len(shap_values.shape) == 2:
        shap_values = shap_values.abs.mean(0)

    # unpack the Explanation object
    features = shap_values.data
    feature_names = shap_values.input_names
    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering
    if partition_tree is not None:
        assert partition_tree.shape[1] == 4, "The clustering provided by the Explanation object does not seem to be a partition tree (which is all shap.plots.bar supports)!"
    transform_history = shap_values.transform_history
    values = np.array(shap_values.values)

    if len(values) == 0:
        raise Exception("The passed Explanation is empty! (so there is nothing to plot)")

    # TODO: Rather than just show the "1st token", "2nd token", etc. it would be better to show the "Instance 0's 1st but", etc
    if issubclass(type(feature_names), str):
        feature_names = [ordinal_str(i)+" "+feature_names for i in range(len(values))]

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "SHAP value"
    for op in transform_history:
        if op[0] == "abs":
            xlabel = "|"+xlabel+"|"
        elif op[0] == "__getitem__":
            pass # no need for slicing to effect our label
        else:
            xlabel = str(op[0])+"("+xlabel+")"

    # unwrap any pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
    
    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])
    
    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(values))]
    orig_values = values.copy()
    while True:
        feature_order = order.apply(shap_values)
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the parition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(values, dist, clust_order, cluster_threshold, order)
        
            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= cluster_threshold:
                values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
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
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(orig_values[inds])
            feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values))])
        values[feature_order[num_features-1]] = np.sum([values[feature_order[i]] for i in range(num_features-1, len(values))])
    
    # build our y-tick labels
    yticklabels = []
    for i in feature_inds:
        if features is not None:
            yticklabels.append(format_value(features[i], "%0.03f") + " = " + feature_names[i])
        else:
            yticklabels.append(feature_names[i])
    if num_features < len(values):
        yticklabels[-1] = "Sum of %d other features" % num_cut

    # compute our figure size based on how many features we are showing
    row_height = 0.5
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    pl.barh(
        y_pos, values[feature_inds],
        0.7, align='center',
        color=[colors.blue_rgb if values[feature_inds[i]] <= 0 else colors.red_rgb for i in range(len(y_pos))]
    )

    # draw the yticks
    pl.yticks(list(y_pos) + list(y_pos), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=13)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    #xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width

    for i in range(len(y_pos)):
        ind = feature_order[i]
        if values[ind] < 0:
            pl.text(
                values[ind] - (5/72)*bbox_to_xscale, y_pos[i], format_value(values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=12
            )
        else:
            pl.text(
                values[ind] + (5/72)*bbox_to_xscale, y_pos[i], format_value(values[ind], '%+0.02f'),
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
            except TypeError:
                pass # features[i] must not be a number
    
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=11)

    xmin,xmax = pl.gca().get_xlim()
    
    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.05)
    
    # if features is None:
    #     pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    # else:
    pl.xlabel(xlabel, fontsize=13)

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
        l = pl.axvline((cluster_threshold / (xlines_max - xlines_min)) * 0.1 * (xmax - xmin) + xmax, color="#dddddd", dashes=(1, 1))
        l.set_clip_on(False)
        
        for (xline, yline) in zip(xlines, ylines):
            
            # normalize the x values to fall between 0 and 1
            xv = (np.array(xline) / (xlines_max - xlines_min))

            # only draw if we are not going past distance threshold
            if np.array(xline).max() <= cluster_threshold:

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

def get_sort_order(shap_values, dist, clust_order, cluster_threshold, order):
    """ Returns a sorted order of the values where we respect the clustering order when dist[i,j] < cluster_threshold
    """
    
    #feature_imp = np.abs(values)

    # if partition_tree is not None:
    #     new_tree = fill_internal_max_values(partition_tree, shap_values)
    #     clust_order = sort_inds(new_tree, np.abs(shap_values))
    clust_inds = np.argsort(clust_order)

    feature_order = order.apply(Explanation(shap_values))
    # print("feature_order", feature_order)
    for i in range(len(feature_order)-1):
        ind1 = feature_order[i]
        next_ind = feature_order[i+1]
        next_ind_pos = i + 1
        for j in range(i+1,len(feature_order)):
            ind2 = feature_order[j]

            
            
            #if feature_imp[ind] > 
            # if ind1 == 2:
            #     print(ind1, ind2, dist[ind1,ind2])
            if dist[ind1,ind2] <= cluster_threshold:
                
                # if ind1 == 2:
                #     print(clust_inds)
                #     print(ind1, ind2, next_ind, dist[ind1,ind2], clust_inds[ind2], clust_inds[next_ind])
                if dist[ind1,next_ind] > cluster_threshold or clust_inds[ind2] < clust_inds[next_ind]:
                    next_ind = ind2
                    next_ind_pos = j
            # print("next_ind", next_ind)
            # print("next_ind_pos", next_ind_pos)
        
        # insert the next_ind next
        for j in range(next_ind_pos, i+1, -1):
            #print("j", j)
            feature_order[j] = feature_order[j-1]
        feature_order[i+1] = next_ind
        #print(feature_order)

        
    
    return feature_order

def merge_nodes(values, partition_tree, orig_inds):
    """ This merges the two clustered leaf nodes with the smallest total value.
    """
    M = partition_tree.shape[0] + 1

    ptind = 0
    min_val = np.inf
    for i in range(partition_tree.shape[0]):
        ind1 = int(partition_tree[i,0])
        ind2 = int(partition_tree[i,1])
        if ind1 < M and ind2 < M:
            val = np.abs(values[ind1]) + np.abs(values[ind2])
            if val < min_val:
                min_val = val
                ptind = i
                #print("ptind", ptind, min_val)

    ind1 = int(partition_tree[ptind,0])
    ind2 = int(partition_tree[ptind,1])
    if ind1 > ind2:
        tmp = ind1
        ind1 = ind2
        ind2 = tmp
    
    values_new = values.copy()
    values_new[ind1] += values_new[ind2]
    values_new = np.delete(values_new, ind2)
    
    partition_tree_new = partition_tree.copy()
    for i in range(partition_tree_new.shape[0]):
        i0 = int(partition_tree_new[i,0])
        i1 = int(partition_tree_new[i,1])
        if i0 == ind2:
            partition_tree_new[i,0] = ind1
        elif i0 > ind2:
            partition_tree_new[i,0] -= 1
            if i0 == ptind + M:
                partition_tree_new[i,0] = ind1
            elif i0 > ptind + M:
                partition_tree_new[i,0] -= 1

            
        if i1 == ind2:
            partition_tree_new[i,1] = ind1
        elif i1 > ind2:
            partition_tree_new[i,1] -= 1
            if i1 == ptind + M:
                partition_tree_new[i,1] = ind1
            elif i1 > ptind + M:
                partition_tree_new[i,1] -= 1
    partition_tree_new = np.delete(partition_tree_new, ptind, axis=0)
    
    orig_inds_new = copy.deepcopy(orig_inds)
    orig_inds_new[ind1] += orig_inds_new[ind2]
    del orig_inds_new[ind2]

    # update the counts to be correct
    fill_counts(partition_tree_new)
    
    return values_new, partition_tree_new, orig_inds_new

def dendrogram_coords(leaf_positions, partition_tree):
    """ Returns the x and y coords of the lines of a dendrogram where the leaf order is given.

    Note that scipy can compute these coords as well, but it does not allow you to easily specify
    a specific leaf order, hence this reimplementation.
    """
    
    xout = []
    yout = []
    _dendrogram_coords_rec(partition_tree.shape[0]-1, leaf_positions, partition_tree, xout, yout)
    
    return np.array(xout), np.array(yout)
def _dendrogram_coords_rec(pos, leaf_positions, partition_tree, xout, yout):
    M = partition_tree.shape[0] + 1
    
    if pos < 0:
        return leaf_positions[pos + M], 0
    
    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M
    
    x_left, y_left = _dendrogram_coords_rec(left, leaf_positions, partition_tree, xout, yout)
    x_right, y_right = _dendrogram_coords_rec(right, leaf_positions, partition_tree, xout, yout)
    
    y_curr = partition_tree[pos, 2]
    
    xout.append([x_left, x_left, x_right, x_right])
    yout.append([y_left, y_curr, y_curr, y_right])
    
    return (x_left + x_right) / 2, y_curr

def fill_internal_max_values(partition_tree, leaf_values):
    """ This fills the forth column of the partition tree matrix with the max leaf value in that cluster.
    """
    M = partition_tree.shape[0] + 1
    new_tree = partition_tree.copy()
    for i in range(new_tree.shape[0]):
        val = 0
        if new_tree[i,0] < M:
            ind = int(new_tree[i,0])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i,0])-M
            val = max(val, np.abs(new_tree[ind,3])) # / partition_tree[ind,2])
        if new_tree[i,1] < M:
            ind = int(new_tree[i,1])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i,1])-M
            val = max(val, np.abs(new_tree[ind,3])) # / partition_tree[ind,2])
        new_tree[i,3] = val
    return new_tree

def fill_counts(partition_tree):
    """ This updates the 
    """
    M = partition_tree.shape[0] + 1
    for i in range(partition_tree.shape[0]):
        val = 0
        if partition_tree[i,0] < M:
            ind = int(partition_tree[i,0])
            val += 1
        else:
            ind = int(partition_tree[i,0])-M
            val += partition_tree[ind,3]
        if partition_tree[i,1] < M:
            ind = int(partition_tree[i,1])
            val += 1
        else:
            ind = int(partition_tree[i,1])-M
            val += partition_tree[ind,3]
        partition_tree[i,3] = val

def sort_inds(partition_tree, leaf_values, pos=None, inds=None):
    if inds is None:
        inds = []
    
    if pos is None:
        partition_tree = fill_internal_max_values(partition_tree, leaf_values)
        pos = partition_tree.shape[0]-1
    
    M = partition_tree.shape[0] + 1
        
    if pos < 0:
        inds.append(pos + M)
        return
    
    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M
    
    
    left_val = partition_tree[left,3] if left >= 0 else leaf_values[left + M]
    right_val = partition_tree[right,3] if right >= 0 else leaf_values[right + M]
    
    if left_val < right_val:
        tmp = right
        right = left
        left = tmp
    
    sort_inds(partition_tree, leaf_values, left, inds)
    sort_inds(partition_tree, leaf_values, right, inds)
    
    return inds

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