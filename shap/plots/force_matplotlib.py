import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib


def draw_bars(out_value, features, width_separators, width_bar):
    """Draw the bars and separators."""
    rectangle_list = []
    separator_list = []
    
    val = out_value
    if (float(features[0][0]) >= 0) | (len(features) == 0):
        feature_type = 'positive'
    else:
        feature_type = 'negative'
    
    for index, features in zip(range(len(features)), features):
        if feature_type == 'positive':
            left_bound = val - float(features[0])
            right_bound = val
            val = left_bound
            
            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            colors = ['#FF0D57', '#FFC3D5']
        else:
            left_bound = val
            right_bound = val + np.abs(float(features[0]))
            val = right_bound
            
            separator_indent = - np.abs(width_separators)
            separator_pos = right_bound
            colors = ['#1E88E5', '#D1E6FA']
        
        # Create rectangle
        if index == 0:
            if feature_type == 'positive':
                points_rectangle = [[left_bound, 0],
                                    [right_bound, 0],
                                    [right_bound, width_bar],
                                    [left_bound, width_bar],
                                    [left_bound + separator_indent, (width_bar / 2)]
                                    ]
            else:
                points_rectangle = [[right_bound, 0],
                                    [left_bound, 0],
                                    [left_bound, width_bar],
                                    [right_bound, width_bar],
                                    [right_bound + separator_indent, (width_bar / 2)]
                                    ]
        
        else:
            points_rectangle = [[left_bound, 0],
                                [right_bound, 0],
                                [right_bound + separator_indent * 0.90, (width_bar / 2)],
                                [right_bound, width_bar],
                                [left_bound, width_bar],
                                [left_bound + separator_indent * 0.90, (width_bar / 2)]]

        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                           facecolor=colors[0], linewidth=0)
        rectangle_list += [line]

        # Create seperator
        points_separator = [[separator_pos, 0],
                            [separator_pos + separator_indent, (width_bar / 2)],
                            [separator_pos, width_bar]]
        
        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=colors[1], lw=3)
        separator_list += [line]

    return rectangle_list, separator_list


def draw_labels(fig, ax, out_value, features, total_effect=0, min_perc=0.05):
    start_text = out_value
    val = out_value
    
    # Get feature type
    if (float(features[0][0]) >= 0) | (len(features) == 0):
        feature_type = 'positive'
    else:
        feature_type = 'negative'
    
    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        alignement = 'right'
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        alignement = 'left'
        sign = -1
    
    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[val, val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = val
    
    box_end = out_value
    
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        if np.abs(float(feature[0])) < min_perc * np.abs(total_effect):
            continue
        
        val = val - sign * np.abs(float(feature[0]))
        
        text = feature[2] + ' = ' + feature[1]
        text_out_val = plt.text(start_text - sign * np.abs(start_text) * 0.01,
                                -0.15, text,
                                fontsize=12, color=colors[0],
                                horizontalalignment=alignement)
        text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))
        
        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents()\
                               .transformed(ax.transData.inverted())
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]
        
        # If the feature goes over the side of the plot, we remove that label
        # and stop drawing labels
        if box_end_ > ax.get_xlim()[1]:
            text_out_val.remove()
            break
        
        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * np.abs(box_end_) * 0.01
            x, y = np.array([[val, box_end, box_end],
                             [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end
    
    # Create shadding
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [val, 0], [box_end, -0.08],
            [box_end, -0.2], [out_value, -0.2],
            [out_value, 0]]
    
    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch) 
    
    if feature_type == 'positive':
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.
    
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)
    
    Z, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(Z2, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.3,
                    origin='lower', extent=extent_shading,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)
    
    return fig, ax

def format_data(data):
    """Format data."""
    # Format negative features
    neg_features = np.array([[data['features'][x]['effect'],
                              data['features'][x]['value'],
                              data['featureNames'][x]]
                             for x in data['features'].keys() if data['features'][x]['effect'] < 0])
    
    neg_features = np.array(sorted(neg_features, key=lambda x: float(x[0]), reverse=False))
    
    # Format postive features
    pos_features = np.array([[data['features'][x]['effect'],
                              data['features'][x]['value'],
                              data['featureNames'][x]]
                             for x in data['features'].keys() if data['features'][x]['effect'] >= 0])
    pos_features = np.array(sorted(pos_features, key=lambda x: float(x[0]), reverse=True))
    
    # Convert using the link function
    if data['link'] == 'identity':
        pass
    elif data['link'] == 'logit':
        # Convert ouput value
        
        # Scale of total effect:
        
        for i in neg_features:
            pass
    else:
        assert False, 'ERROR: Unrecognized link function: ' + str(data['link'])
    return neg_features, pos_features


def draw_additive_plot(data, figsize, show):
    """Draw additive plot."""
    # Turn off interactive plot
    if show == False:
        plt.ioff()
    
    # Format data
    neg_features, pos_features = format_data(data)
    
    # Compute overall metrics
    total_neg = np.sum(neg_features[:, 0].astype(float))
    total_pos = np.sum(pos_features[:, 0].astype(float))
    
    base_value = data['baseValue']
    out_value = data['outValue']
    
    # Define plots
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute axis limit
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])

    ax.set_xlim(min(out_value - np.abs(total_pos), base_value) - padding,
                max(out_value + np.abs(total_neg), base_value) + padding)

    plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
                    labeltop=True, labelbottom=False)
    plt.locator_params(axis='x', nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != 'top':
            spine.set_visible(False)

    # Define width of bar
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200
    
    # Create bar for negative shap values
    rectangle_list, separator_list = draw_bars(out_value, neg_features,
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)
    
    for i in separator_list:
        ax.add_patch(i)
    
    # Create bar for positive shap values
    rectangle_list, separator_list = draw_bars(out_value, pos_features,
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)
    
    for i in separator_list:
        ax.add_patch(i)

    # higher lower legend
    plt.text(out_value - np.abs(out_value) * 0.02, 0.405, 'higher',
             fontsize=13, color='#FF0D57',
             horizontalalignment='right')

    plt.text(out_value + np.abs(out_value) * 0.02, 0.405, 'lower',
             fontsize=13, color='#1E88E5',
             horizontalalignment='left')

    plt.text(out_value, 0.4, r'$\leftarrow$',
             fontsize=13, color='#1E88E5',
             horizontalalignment='center')

    plt.text(out_value, 0.425, r'$\rightarrow$',
             fontsize=13, color='#FF0D57',
             horizontalalignment='center')
    
    # Add labels
    total_effect = np.abs(total_neg) + total_pos
    fig, ax = draw_labels(fig, ax, out_value, neg_features,
                          total_effect, min_perc=0.05)
    
    fig, ax = draw_labels(fig, ax, out_value, pos_features,
                          total_effect, min_perc=0.05)
    
    # Convert value to the ones displayed
    if data['link'] == 'identity':
        out_value_display = out_value
        base_value_display = base_value
        
    elif data['link'] == 'logit':
        out_value_display = 1 / (1 + np.exp(-out_value))
        base_value_display = 1 / (1 + np.exp(-base_value))
        
        tick_labels = ax.get_xticklabels()
        current_tick_labels = []
        for i in tick_labels:
            tick_label = i.get_text()
            new_tick_label = round(1 / (1 + np.exp(-float(tick_label))), 4)
            
            current_tick_labels += [new_tick_label]
        ax.set_xticklabels(current_tick_labels, fontdict=None, minor=False)
        
    else:
        assert False, "ERROR: Unrecognized link function: " + str(data['link'])
        
    # Add output value
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    text_out_val = plt.text(out_value, 0.25, '{0:.2f}'.format(out_value_display),
                            fontproperties=font,
                            fontsize=14,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    text_out_val = plt.text(out_value, 0.33, data['outNames'][0],
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    # Add label for base value
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)
    
    text_out_val = plt.text(base_value, 0.33, 'base value',
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    if show:
        plt.show()
    else:
        return plt.gcf()
    
