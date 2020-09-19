import numpy as np
import warnings
from . import colors
from ..utils import ordinal_str
import random
import string


# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
def text(shap_values, num_starting_labels=0, group_threshold=1, separator='', xmin=None, xmax=None, cmax=None):
    """ Plots an explanation of a string of text using coloring and interactive labels.
    
    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.
    """
    from IPython.core.display import display, HTML

    def values_min_max(values, base_values):
        """ Used to pick our axis limits.
        """
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2:
        tokens, values, group_sizes = process_shap_values(shap_values[0], group_threshold, separator)
        xmin, xmax, cmax = values_min_max(values, shap_values[0].base_values)
        for i in range(1,len(shap_values)):
            tokens, values, group_sizes = process_shap_values(shap_values[i], group_threshold, separator)
            xmin_i,xmax_i,cmax_i = values_min_max(values, shap_values[i].base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        for i in range(len(shap_values)):
            display(HTML("<br/><b>"+ordinal_str(i)+" instance:</b><br/>"))
            text(shap_values[i], num_starting_labels=num_starting_labels, group_threshold=group_threshold, separator=separator, xmin=xmin, xmax=xmax, cmax=cmax)
        return
    
    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(shap_values.values, shap_values.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new
    

    tokens, values, group_sizes = process_shap_values(shap_values, group_threshold, separator)
    
    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    maxv = values.max()
    minv = values.min()
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())
    
    uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [t.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') for t in tokens]
    out += svg_force_plot(values, shap_values.base_values, shap_values.base_values + values.sum(), encoded_tokens, uuid, xmin, xmax)
    
    for i in range(len(tokens)):
        scaled_value = 0.5 + 0.5 * values[i] / cmax
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0]*255, color[1]*255, color[2]*255, color[3])
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"
        
        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])
        
        # the HTML for this token
        out += "<div style='display: " + wrapper_display + "; text-align: center;'>" \
             + "<div style='display: " + label_display + "; color: #999; padding-top: 0px; font-size: 12px;'>" \
             + value_label \
             + "</div>" \
             + f"<div id='_tp_{uuid}_ind_{i}'" \
             +   "style='display: inline; background: rgba" + str(color) + "; border-radius: 3px; padding: 0px'" \
             +   "onclick=\"if (this.previousSibling.style.display == 'none') {" \
             +       "this.previousSibling.style.display = 'block';" \
             +       "this.parentNode.style.display = 'inline-block';" \
             +     "} else {" \
             +       "this.previousSibling.style.display = 'none';" \
             +       "this.parentNode.style.display = 'inline';" \
             +     "}" \
             +   "\"" \
             +   f"onmouseover=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;" \
             +   "\"" \
             +   f"onmouseout=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;" \
             +   "\"" \
             + ">" \
             + tokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') \
             + "</div>" \
             + "</div>"

    display(HTML(out))

def process_shap_values(shap_values, group_threshold, separator):

    # unpack the Explanation object
    tokens = shap_values.data
    clustering = getattr(shap_values, "clustering", None)
    if getattr(shap_values, "hierarchical_values", None) is not None:
        values = shap_values.hierarchical_values
    else:
        values = shap_values.values

    # See if we got hierarchical input data. If we did then we need to reprocess the 
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError("The length of the attribution values must match the number of " + \
                             "tokens if shap_values.clustering is None! When passing hierarchical " + \
                             "attributions the clustering is also required.")
        
        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i,0])
            ri = int(clustering[i,1])
            groups.append(groups[li] + groups[ri])
            lower_values[M+i] = lower_values[li] + lower_values[ri] + values[M+i]
            max_values[i+M] = max(abs(values[M+i]) / len(groups[M+i]), max_values[li], max_values[ri])
    
        # compute the upper_values
        upper_values = np.zeros(len(values))
        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i-M,0])
            ri = int(clustering[i-M,1])
            upper_values[i] = value
            value += values[i]
#             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
#             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)
        
        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []
        def merge_tokens(new_tokens, new_values, group_sizes, i):
            
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)
            else:

                # compute the dividend at internal nodes
                li = int(clustering[i-M,0])
                ri = int(clustering[i-M,1])
                dv = abs(values[i]) / len(groups[i])
                
                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(separator.join([tokens[g] for g in groups[li]]) + separator + separator.join([tokens[g] for g in groups[ri]]))
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))
                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)
        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)
        
        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)
        M = len(tokens) 
    else:
        group_sizes = np.ones(M)

    return tokens, values, group_sizes

def svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax):
    

    def xpos(xval):
        return 100 * (xval - xmin)  / (xmax - xmin)

    s = ''
    s += '<svg width="100%" height="80px">'
    
    ### x-axis marks ###

    # draw x axis line
    s += '<line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" />'

    # draw base value
    def draw_tick_mark(xval, label=None, bold=False):
        s = ""
        s += '<line x1="%f%%" y1="33" x2="%f%%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" />' % ((xpos(xval),) * 2)
        if not bold:
            s += '<text x="%f%%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">%f</text>' % (xpos(xval),xval)
        else:
            s += '<text x="%f%%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">%f</text>' % (xpos(xval),xval)
            s += '<text x="%f%%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle">%f</text>' % (xpos(xval),xval)
        if label is not None:
            s += '<text x="%f%%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">%s</text>' % (xpos(xval), label)
        return s

    s += draw_tick_mark(base_values, label="base value")
    tick_interval = (xmax - xmin) / 7
    side_buffer = (xmax - xmin) / 14
    for i in range(1,10):
        pos = base_values - i * tick_interval
        if pos < xmin + side_buffer:
            break
        s += draw_tick_mark(pos)
    for i in range(1,10):
        pos = base_values + i * tick_interval
        if pos > xmax - side_buffer:
            break
        s += draw_tick_mark(pos)
    s += draw_tick_mark(fx, bold=True, label="f(x)")
    
    
    ### Positive value marks ###
    
    red = tuple(colors.red_rgb * 255)
    light_red = (255, 195, 213)
    
    # draw base red bar
    x = fx - values[values > 0].sum()
    w = 100 * values[values > 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(x)}%" width="{w}%" y="40" height="18" style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # a line under the bar to animate
        s += f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{red};stroke-width:2; opacity: 0"/>'
        
        # the text label cropped and centered
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'
        
        # the text label cropped and centered
        s += f'<svg x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += f'  </svg>'
        s += f'</svg>'
        
        last_pos = pos
    
    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({2*j-8},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
            
        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate({2*j-0},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
        
        last_pos = pos
    
    # center padding
    s += f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{red}"/>'
        
    # cover up a notch at the end of the red bar
    pos = fx - values[values > 0].sum()
    s += f'<g transform="translate(-11.5,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += f'    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f'  </svg>'
    s += f'</g>'


    # draw the light red divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-1.5,0)">'
            s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{light_red};stroke-width:2" />'
            s += f'  </svg>'
            s += f'</g>'
        
        # mouse over rectangle
        s += f'<rect x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%"'
        s += f'      onmouseover="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'underline\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'"'
        s += f'      onmouseout="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'none\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'" style="fill:rgb(0,0,0,0)" />'
        
        last_pos = pos
        
    
    ### Negative value marks ###
    
    blue = tuple(colors.blue_rgb * 255)
    light_blue = (208, 230, 250)
    
    # draw base blue bar
    w = 100 * -values[values < 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # a line under the bar to animate
        s += f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{blue};stroke-width:2; opacity: 0"/>'
        
        # the value text
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'
        
        # the text label cropped and centered
        s += f'<svg x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += f'  </svg>'
        s += f'</svg>'
        
        last_pos = pos
    
    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({-2*j+2},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
            
        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate(-{2*j+8},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
        
        last_pos = pos
    
    # center padding
    s += f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{blue}"/>'
    
    # cover up a notch at the end of the blue bar
    pos = fx - values[values < 0].sum()
    s += f'<g transform="translate(-6.0,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += f'    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f'  </svg>'
    s += f'</g>'

    # draw the light blue divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-6.0,0)">'
            s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{light_blue};stroke-width:2" />'
            s += f'  </svg>'
            s += f'</g>'
            
        # mouse over rectangle
        s += f'<rect x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%"'
        s += f'      onmouseover="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'underline\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'"'
        s += f'      onmouseout="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'none\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'" style="fill:rgb(0,0,0,0)" />'
        
        last_pos = pos

    s += '</svg>'
    
    return s


def text_old(shap_values, tokens, partition_tree=None, num_starting_labels=0, group_threshold=1, separator=''):
    """ Plots an explanation of a string of text using coloring and interactive labels.
    
    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.
    """
    
    # See if we got hierarchical input data. If we did then we need to reprocess the 
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(shap_values) != M:
        
        # make sure we were given a partition tree
        if partition_tree is None:
            raise ValueError("The length of the attribution values must match the number of " + \
                             "tokens if partition_tree is None! When passing hierarchical " + \
                             "attributions the partition_tree is also required.")
        
        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(shap_values))
        lower_values[:M] = shap_values[:M]
        max_values = np.zeros(len(shap_values))
        max_values[:M] = np.abs(shap_values[:M])
        for i in range(partition_tree.shape[0]):
            li = partition_tree[i,0]
            ri = partition_tree[i,1]
            groups.append(groups[li] + groups[ri])
            lower_values[M+i] = lower_values[li] + lower_values[ri] + shap_values[M+i]
            max_values[i+M] = max(abs(shap_values[M+i]) / len(groups[M+i]), max_values[li], max_values[ri])
    
        # compute the upper_values
        upper_values = np.zeros(len(shap_values))
        def lower_credit(upper_values, partition_tree, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = partition_tree[i-M,0]
            ri = partition_tree[i-M,1]
            upper_values[i] = value
            value += shap_values[i]
#             lower_credit(upper_values, partition_tree, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
#             lower_credit(upper_values, partition_tree, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, partition_tree, li, value * 0.5)
            lower_credit(upper_values, partition_tree, ri, value * 0.5)

        lower_credit(upper_values, partition_tree, len(shap_values) - 1)
        
        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_shap_values = []
        group_sizes = []
        def merge_tokens(new_tokens, new_values, group_sizes, i):
            
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)
            else:

                # compute the dividend at internal nodes
                li = partition_tree[i-M,0]
                ri = partition_tree[i-M,1]
                dv = abs(shap_values[i]) / len(groups[i])
                
                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(separator.join([tokens[g] for g in groups[li]]) + separator + separator.join([tokens[g] for g in groups[ri]]))
                    new_values.append(group_values[i] / len(groups[i]))
                    group_sizes.append(len(groups[i]))
                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)
        merge_tokens(new_tokens, new_shap_values, group_sizes, len(group_values) - 1)
        
        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        shap_values = np.array(new_shap_values)
        group_sizes = np.array(group_sizes)
        M = len(tokens) 
    else:
        group_sizes = np.ones(M)
    
    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(shap_values))[:num_starting_labels]
    maxv = shap_values.max()
    minv = shap_values.min()
    out = ""
    for i in range(M):
        scaled_value = 0.5 + 0.5 * shap_values[i] / max(abs(maxv), abs(minv))
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0]*255, color[1]*255, color[2]*255, color[3])
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"
        
        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(shap_values[i].round(3))
        else:
            value_label = str((shap_values[i] * group_sizes[i]).round(3)) + " / " + str(group_sizes[i])
        
        # the HTML for this token
        out += "<div style='display: " + wrapper_display + "; text-align: center;'>" \
             + "<div style='display: " + label_display + "; color: #999; padding-top: 0px; font-size: 12px;'>" \
             + value_label \
             + "</div>" \
             + "<div " \
             +   "style='display: inline; background: rgba" + str(color) + "; border-radius: 3px; padding: 0px'" \
             +   "onclick=\"if (this.previousSibling.style.display == 'none') {" \
             +       "this.previousSibling.style.display = 'block';" \
             +       "this.parentNode.style.display = 'inline-block';" \
             +     "} else {" \
             +       "this.previousSibling.style.display = 'none';" \
             +       "this.parentNode.style.display = 'inline';" \
             +     "}" \
             +   "\"" \
             + ">" \
             + tokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') \
             + "</div>" \
             + "</div>"

    from IPython.core.display import display, HTML
    return display(HTML(out))