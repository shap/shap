import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def draw_bars(out_value, features, feature_type, width_separators, width_bar):
    """Draw the bars and separators."""
    rectangle_list = []
    separator_list = []

    pre_val = out_value
    for index, features in zip(range(len(features)), features):
        if feature_type == 'positive':
            left_bound = float(features[0])
            right_bound = pre_val
            pre_val = left_bound

            separator_indent = np.abs(width_separators)
            separator_pos = left_bound
            colors = ['#FF0D57', '#FFC3D5']
        else:
            left_bound = pre_val
            right_bound = float(features[0])
            pre_val = right_bound

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

        # Create separator
        points_separator = [[separator_pos, 0],
                            [separator_pos + separator_indent, (width_bar / 2)],
                            [separator_pos, width_bar]]

        line = plt.Polygon(points_separator, closed=None, fill=None,
                           edgecolor=colors[1], lw=3)
        separator_list += [line]

    return rectangle_list, separator_list


def draw_labels(fig, ax, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05, text_rotation=0):
    start_text = out_value
    pre_val = out_value

    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        alignment = 'right'
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        alignment = 'left'
        sign = -1

    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        # Compute value for current feature
        val = float(feature[0])

        # Draw labels.
        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + ' = ' + feature[1]

        if text_rotation != 0:
            va_alignment = 'top'
        else:
            va_alignment = 'baseline'

        text_out_val = plt.text(start_text - sign * offset_text,
                                -0.15, text,
                                fontsize=12, color=colors[0],
                                horizontalalignment=alignment,
                                va=va_alignment,
                                rotation=text_rotation)
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
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end],
                             [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        # Update previous value
        pre_val = float(feature[0])


    # Create line for labels
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [pre_val, 0], [box_end, -0.08],
            [box_end, -0.2], [out_value, -0.2],
            [out_value, 0]]

    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    # Extend axis if needed
    lower_lim, upper_lim = ax.get_xlim()
    if (box_end < lower_lim):
        ax.set_xlim(box_end, upper_lim)

    if (box_end > upper_lim):
        ax.set_xlim(lower_lim, box_end)

    # Create shading
    if feature_type == 'positive':
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.

    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)

    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
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

    # Format positive features
    pos_features = np.array([[data['features'][x]['effect'],
                              data['features'][x]['value'],
                              data['featureNames'][x]]
                             for x in data['features'].keys() if data['features'][x]['effect'] >= 0])
    pos_features = np.array(sorted(pos_features, key=lambda x: float(x[0]), reverse=True))

    # Define link function
    if data['link'] == 'identity':
        def convert_func(x):
            return x
    elif data['link'] == 'logit':
        def convert_func(x):
            return 1 / (1 + np.exp(-x))
    else:
        emsg = f"ERROR: Unrecognized link function: {data['link']}"
        raise ValueError(emsg)

    # Convert negative feature values to plot values
    neg_val = data['outValue']
    for i in neg_features:
        val = float(i[0])
        neg_val = neg_val + np.abs(val)
        i[0] = convert_func(neg_val)
    if len(neg_features) > 0:
        total_neg = np.max(neg_features[:, 0].astype(float)) - \
                    np.min(neg_features[:, 0].astype(float))
    else:
        total_neg = 0

    # Convert positive feature values to plot values
    pos_val = data['outValue']
    for i in pos_features:
        val = float(i[0])
        pos_val = pos_val - np.abs(val)
        i[0] = convert_func(pos_val)

    if len(pos_features) > 0:
        total_pos = np.max(pos_features[:, 0].astype(float)) - \
                    np.min(pos_features[:, 0].astype(float))
    else:
        total_pos = 0

    # Convert output value and base value
    data['outValue'] = convert_func(data['outValue'])
    data['baseValue'] = convert_func(data['baseValue'])

    return neg_features, total_neg, pos_features, total_pos


def draw_output_element(out_name, out_value, ax):
    # Add output value
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    text_out_val = plt.text(out_value, 0.25, f'{out_value:.2f}',
                            fontproperties=font,
                            fontsize=14,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    text_out_val = plt.text(out_value, 0.33, out_name,
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def draw_base_element(base_value, ax):
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = plt.text(base_value, 0.33, 'base value',
                            fontsize=12, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def draw_higher_lower_element(out_value, offset_text):
    plt.text(out_value - offset_text, 0.405, 'higher',
             fontsize=13, color='#FF0D57',
             horizontalalignment='right')

    plt.text(out_value + offset_text, 0.405, 'lower',
             fontsize=13, color='#1E88E5',
             horizontalalignment='left')

    plt.text(out_value, 0.4, r'$\leftarrow$',
             fontsize=13, color='#1E88E5',
             horizontalalignment='center')

    plt.text(out_value, 0.425, r'$\rightarrow$',
             fontsize=13, color='#FF0D57',
             horizontalalignment='center')


def update_axis_limits(ax, total_pos, pos_features, total_neg,
                       neg_features, base_value, out_value):
    ax.set_ylim(-0.5, 0.15)
    padding = np.max([np.abs(total_pos) * 0.2,
                      np.abs(total_neg) * 0.2])

    if len(pos_features) > 0:
        min_x = min(np.min(pos_features[:, 0].astype(float)), base_value) - padding
    else:
        min_x = out_value - padding
    if len(neg_features) > 0:
        max_x = max(np.max(neg_features[:, 0].astype(float)), base_value) + padding
    else:
        max_x = out_value + padding
    ax.set_xlim(min_x, max_x)

    plt.tick_params(top=True, bottom=False, left=False, right=False, labelleft=False,
                    labeltop=True, labelbottom=False)
    plt.locator_params(axis='x', nbins=12)

    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
        if key != 'top':
            spine.set_visible(False)


def draw_additive_plot(data, figsize, show, text_rotation=0, min_perc=0.05):
    """Draw additive plot."""
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # Format data
    neg_features, total_neg, pos_features, total_pos = format_data(data)

    # Compute overall metrics
    base_value = data['baseValue']
    out_value = data['outValue']
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04

    # Define plots
    fig, ax = plt.subplots(figsize=figsize)

    # Compute axis limit
    update_axis_limits(ax, total_pos, pos_features, total_neg,
                       neg_features, base_value, out_value)

    # Define width of bar
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    # Create bar for negative shap values
    rectangle_list, separator_list = draw_bars(out_value, neg_features, 'negative',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Create bar for positive shap values
    rectangle_list, separator_list = draw_bars(out_value, pos_features, 'positive',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Add labels
    total_effect = np.abs(total_neg) + total_pos
    fig, ax = draw_labels(fig, ax, out_value, neg_features, 'negative',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    fig, ax = draw_labels(fig, ax, out_value, pos_features, 'positive',
                          offset_text, total_effect, min_perc=min_perc, text_rotation=text_rotation)

    # higher lower legend
    draw_higher_lower_element(out_value, offset_text)

    # Add label for base value
    draw_base_element(base_value, ax)

    # Add output label
    out_names = data['outNames'][0]
    draw_output_element(out_names, out_value, ax)

    # Scale axis
    if data['link'] == 'logit':
        plt.xscale('logit')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain')

    if show:
        plt.show()
    else:
        return plt.gcf()
