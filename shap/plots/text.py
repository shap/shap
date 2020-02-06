import numpy as np
import warnings
from . import colors
            
def text_plot(shap_values, tokens, partition_tree=None, num_starting_labels=0, group_theshold=0.5):
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
            lower_credit(upper_values, partition_tree, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, partition_tree, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
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
                if dv > group_theshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(" ".join([tokens[g] for g in groups[li]]) + " " + " ".join([tokens[g] for g in groups[ri]]))
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
             + "<div style='display: " + label_display + "; color: #999; padding-top: 4px; font-size: 12px;'>" \
             + value_label \
             + "</div>" \
             + "<div " \
             +   "style='display: inline; background: rgba" + str(color) + "; border-radius: 5px; padding: 1px'" \
             +   "onclick=\"if (this.previousSibling.style.display == 'none') {" \
             +       "this.previousSibling.style.display = 'block';" \
             +       "this.parentNode.style.display = 'inline-block';" \
             +     "} else {" \
             +       "this.previousSibling.style.display = 'none';" \
             +       "this.parentNode.style.display = 'inline';" \
             +     "}" \
             +   "\"" \
             + ">" \
             + tokens[i].replace("<", "&lt;").replace(">", "&gt;") \
             + "</div>" \
             + "</div> "

    from IPython.core.display import display, HTML
    return display(HTML(out))