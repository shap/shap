import json
import random
import string
import warnings

import numpy as np

from . import colors

try:
    from IPython.display import HTML
    from IPython.display import display as ipython_display

    have_ipython = True
except ImportError:
    have_ipython = False


# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
def text(
    shap_values,
    num_starting_labels=0,
    grouping_threshold=0.01,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
    display=True,
):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap values for a string (#input_tokens x output_tokens).

    num_starting_labels : int
        Number of tokens (sorted in descending order by corresponding SHAP values)
        that are uncovered in the initial view.
        When set to 0, all tokens are covered.

    grouping_threshold : float
        If the component substring effects are less than a ``grouping_threshold``
        fraction of an unlowered interaction effect, then we visualize the entire group
        as a single chunk. This is primarily used for explanations that were computed
        with fixed_context set to 1 or 0 when using the :class:`.explainers.Partition`
        explainer, since this causes interaction effects to be left on internal nodes
        rather than lowered.

    separator : string
        The string separator that joins tokens grouped by interaction effects and
        unbroken string spans. Defaults to the empty string ``""``.

    xmin : float
        Minimum shap value bound.

    xmax : float
        Maximum shap value bound.

    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens.

    display: bool
        Whether to display or return html to further manipulate or embed. Default: ``True``

    Examples
    --------
    See `text plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html>`_.

    """

    def values_min_max(values, base_values):
        """Used to pick our axis limits."""
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (shap_values.output_names is None or isinstance(shap_values.output_names, str)):
        xmin = 0
        xmax = 0
        cmax = 0

        for i, v in enumerate(shap_values):
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, group_sizes = process_shap_values(v.data, values, grouping_threshold, separator, clustering)

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, v.base_values)
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
    <br>
    <hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
    <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
    </div>
                """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        if display:
            _ipython_display_html(out)
            return
        else:
            return out

    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
            tokens, values, group_sizes = process_shap_values(
                shap_values[:, i].data, values, grouping_threshold, separator, clustering
            )

            # if i == 0:
            #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
            #     continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, shap_values[:, i].base_values)
            if xmin_computed is None or xmin_i < xmin_computed:
                xmin_computed = xmin_i
            if xmax_computed is None or xmax_i > xmax_computed:
                xmax_computed = xmax_i
            if cmax_computed is None or cmax_i > cmax_computed:
                cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = f"""<div align='center'>
<script>
    document._hover_{uuid} = '_tp_{uuid}_output_0';
    document._zoom_{uuid} = undefined;
    function _output_onclick_{uuid}(i) {{
        var next_id = undefined;

        if (document._zoom_{uuid} !== undefined) {{
            document.getElementById(document._zoom_{uuid}+ '_zoom').style.display = 'none';

            if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
                document.getElementById(document._zoom_{uuid}).style.display = 'block';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = '3px solid #000000';
            }} else {{
                document.getElementById(document._zoom_{uuid}).style.display = 'none';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = 'none';
            }}
        }}
        if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
            next_id = '_tp_{uuid}_output_' + i;
            document.getElementById(next_id).style.display = 'none';
            document.getElementById(next_id + '_zoom').style.display = 'block';
            document.getElementById(next_id+'_name').style.borderBottom = '3px solid #000000';
        }}
        document._zoom_{uuid} = next_id;
    }}
    function _output_onmouseover_{uuid}(i, el) {{
        if (document._zoom_{uuid} !== undefined) {{ return; }}
        if (document._hover_{uuid} !== undefined) {{
            document.getElementById(document._hover_{uuid} + '_name').style.borderBottom = 'none';
            document.getElementById(document._hover_{uuid}).style.display = 'none';
        }}
        document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
        el.style.borderBottom = '3px solid #000000';
        document._hover_{uuid} = '_tp_{uuid}_output_' + i;
    }}
</script>
<div style=\"color: rgb(120,120,120); font-size: 12px;\">outputs</div>"""
        output_values = shap_values.values.sum(0) + shap_values.base_values
        output_max = np.max(np.abs(output_values))
        for i, name in enumerate(shap_values.output_names):
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = colors.red_transparent_blue(scaled_value)
            color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
            # '#dddddd' if i == 0 else '#ffffff' border-bottom: {'3px solid #000000' if i == 0 else 'none'};
            out += f"""
<div style="display: inline; border-bottom: {"3px solid #000000" if i == 0 else "none"}; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name"
    onclick="_output_onclick_{uuid}({i})"
    onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>"""
        out += "<br><br>"
        for i, name in enumerate(shap_values.output_names):
            out += f"<div id='_tp_{uuid}_output_{i}' style='display: {'block' if i == 0 else 'none'}';>"
            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
            out += "</div>"
            out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                display=False,
            )
            out += "</div>"
        out += "</div>"
        if display:
            _ipython_display_html(out)
            return
        else:
            return out
        # text_to_text(shap_values)
        # return

    if len(shap_values.shape) == 3:
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(shap_values[j, :, i])
                tokens, values, group_sizes = process_shap_values(
                    shap_values[j, :, i].data, values, grouping_threshold, separator, clustering
                )

                xmin_i, xmax_i, cmax_i = values_min_max(values, shap_values[j, :, i].base_values)
                if xmin_computed is None or xmin_i < xmin_computed:
                    xmin_computed = xmin_i
                if xmax_computed is None or xmax_i > xmax_computed:
                    xmax_computed = xmax_i
                if cmax_computed is None or cmax_i > cmax_computed:
                    cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
<br>
<hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
<div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
</div>
            """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        if display:
            _ipython_display_html(out)
            return
        else:
            return out

    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(shap_values.values, shap_values.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(
        shap_values.data, values, grouping_threshold, separator, clustering
    )

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())

    # uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens]
    output_name = shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    out += svg_force_plot(
        values,
        shap_values.base_values,
        shap_values.base_values + values.sum(),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
        output_name,
    )
    out += (
        "<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-top: -15px;\">inputs</div>"
    )
    for i, token in enumerate(tokens):
        scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

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
        out += f"""<div style='display: {wrapper_display}; text-align: center;'
    ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: 12px;'>{value_label}</div
        ><div id='_tp_{uuid}_ind_{i}'
            style='display: inline; background: rgba{color}; border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {{
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            }} else {{
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }}"
            onmouseover="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
        >{token.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "")}</div></div>"""
    out += "</div>"

    if display:
        _ipython_display_html(out)
        return
    else:
        return out


def process_shap_values(tokens, values, grouping_threshold, separator, clustering=None, return_meta_data=False):
    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                "tokens if shap_values.clustering is None! When passing hierarchical "
                "attributions the clustering is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = lower_values[li] + lower_values[ri] + values[M + i]
            max_values[i + M] = max(abs(values[M + i]) / len(groups[M + i]), max_values[li], max_values[ri])

        # compute the upper_values
        upper_values = np.zeros(len(values))

        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i - M, 0])
            ri = int(clustering[i - M, 1])
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

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:
                # compute the dividend at internal nodes
                li = int(clustering[i - M, 0])
                ri = int(clustering[i - M, 1])
                dv = abs(values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if max(max_values[li], max_values[ri]) < dv * grouping_threshold:
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i

                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i

                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens)
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    if return_meta_data:
        return tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids
    else:
        return tokens, values, group_sizes


def svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax, output_name):
    def xpos(xval):
        return 100 * (xval - xmin) / (xmax - xmin + 1e-8)

    s = ""
    s += '<svg width="100%" height="80px">'

    ### x-axis marks ###

    # draw x axis line
    s += '<line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" />'

    # draw base value
    def draw_tick_mark(xval, label=None, bold=False, backing=False):
        s = ""
        s += f'<line x1="{xpos(xval)}%" y1="33" x2="{xpos(xval)}%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" />'
        if not bold:
            if backing:
                s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
            s += f'<text x="{xpos(xval)}%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
        else:
            if backing:
                s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
            s += f'<text x="{xpos(xval)}%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle">{xval:g}</text>'
        if label is not None:
            s += f'<text x="{xpos(xval)}%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">{label}</text>'
        return s

    xcenter = round((xmax + xmin) / 2, int(round(1 - np.log10(xmax - xmin + 1e-8))))
    s += draw_tick_mark(xcenter)
    #    np.log10(xmax - xmin)

    tick_interval = round((xmax - xmin) / 7, int(round(1 - np.log10(xmax - xmin + 1e-8))))

    # tick_interval = (xmax - xmin) / 7
    side_buffer = (xmax - xmin) / 14
    for i in range(1, 10):
        pos = xcenter - i * tick_interval
        if pos < xmin + side_buffer:
            break
        s += draw_tick_mark(pos)
    for i in range(1, 10):
        pos = xcenter + i * tick_interval
        if pos > xmax - side_buffer:
            break
        s += draw_tick_mark(pos)
    s += draw_tick_mark(base_values, label="base value", backing=True)
    s += draw_tick_mark(
        fx, bold=True, label=f'f<tspan baseline-shift="sub" font-size="8px">{output_name}</tspan>(inputs)', backing=True
    )

    ### Positive value marks ###

    red = tuple(colors.red_rgb * 255)
    light_red = (255, 195, 213)

    # draw base red bar
    x = fx - values[values > 0].sum()
    w = 100 * values[values > 0].sum() / (xmax - xmin + 1e-8)
    s += f'<rect x="{xpos(x)}%" width="{w}%" y="40" height="18" style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{red};stroke-width:2; opacity: 0"/>'

        # the text label cropped and centered
        s += f'<text x="{(xpos(last_pos) + xpos(pos)) / 2}%" y="71" font-size="12px" id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%">'
        s += '  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += "  </svg>"
        s += "</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({2 * j - 8},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate({2 * j - 0},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{red}"/>'

    # cover up a notch at the end of the red bar
    pos = fx - values[values > 0].sum()
    s += '<g transform="translate(-11.5,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += '    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += "  </svg>"
    s += "</g>"

    # draw the light red divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += '<g transform="translate(-1.5,0)">'
            s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{light_red};stroke-width:2" />'
            s += "  </svg>"
            s += "</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%"'
        s += '      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += '"'
        s += '      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += '" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    ### Negative value marks ###

    blue = tuple(colors.blue_rgb * 255)
    light_blue = (208, 230, 250)

    # draw base blue bar
    w = 100 * -values[values < 0].sum() / (xmax - xmin + 1e-8)
    s += f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{blue};stroke-width:2; opacity: 0"/>'

        # the value text
        s += f'<text x="{(xpos(last_pos) + xpos(pos)) / 2}%" y="71" font-size="12px" fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%">'
        s += '  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += "  </svg>"
        s += "</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({-2 * j + 2},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate(-{2 * j + 8},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += "  </svg>"
                s += "</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{blue}"/>'

    # cover up a notch at the end of the blue bar
    pos = fx - values[values < 0].sum()
    s += '<g transform="translate(-6.0,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += '    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += "  </svg>"
    s += "</g>"

    # draw the light blue divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += '<g transform="translate(-6.0,0)">'
            s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{light_blue};stroke-width:2" />'
            s += "  </svg>"
            s += "</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%"'
        s += '      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += '"'
        s += '      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += '" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    s += "</svg>"

    return s


def text_old(shap_values, tokens, partition_tree=None, num_starting_labels=0, grouping_threshold=1, separator=""):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.
    """
    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    warnings.warn(
        "This function is not used within the shap library and will therefore be removed in an upcoming release. "
        "If you rely on this function, please open an issue: https://github.com/shap/shap/issues.",
        FutureWarning,
    )
    M = len(tokens)
    if len(shap_values) != M:
        # make sure we were given a partition tree
        if partition_tree is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                "tokens if partition_tree is None! When passing hierarchical "
                "attributions the partition_tree is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(shap_values))
        lower_values[:M] = shap_values[:M]
        max_values = np.zeros(len(shap_values))
        max_values[:M] = np.abs(shap_values[:M])
        for i in range(partition_tree.shape[0]):
            li = partition_tree[i, 0]
            ri = partition_tree[i, 1]
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = lower_values[li] + lower_values[ri] + shap_values[M + i]
            max_values[i + M] = max(abs(shap_values[M + i]) / len(groups[M + i]), max_values[li], max_values[ri])

        # compute the upper_values
        upper_values = np.zeros(len(shap_values))

        def lower_credit(upper_values, partition_tree, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = partition_tree[i - M, 0]
            ri = partition_tree[i - M, 1]
            upper_values[i] = value
            value += shap_values[i]

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
                li = partition_tree[i - M, 0]
                ri = partition_tree[i - M, 1]
                dv = abs(shap_values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if dv > grouping_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
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
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

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
        out += (
            "<div style='display: "
            + wrapper_display
            + "; text-align: center;'>"
            + "<div style='display: "
            + label_display
            + "; color: #999; padding-top: 0px; font-size: 12px;'>"
            + value_label
            + "</div>"
            + "<div "
            + "style='display: inline; background: rgba"
            + str(color)
            + "; border-radius: 3px; padding: 0px'"
            + "onclick=\"if (this.previousSibling.style.display == 'none') {"
            + "this.previousSibling.style.display = 'block';"
            + "this.parentNode.style.display = 'inline-block';"
            + "} else {"
            + "this.previousSibling.style.display = 'none';"
            + "this.parentNode.style.display = 'inline';"
            + "}"
            + '"'
            + ">"
            + tokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "")
            + "</div>"
            + "</div>"
        )

    return _ipython_display_html(out)


def text_to_text(shap_values):
    # unique ID added to HTML elements and function to avoid collision of different instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    saliency_plot_markup = saliency_plot(shap_values)
    heatmap_markup = heatmap(shap_values)

    html = f"""
    <html>
    <div id="{uuid}_viz_container">
      <div id="{uuid}_viz_header" style="padding:15px;border-style:solid;margin:5px;font-family:sans-serif;font-weight:bold;">
        Visualization Type:
        <select name="viz_type" id="{uuid}_viz_type" onchange="selectVizType_{uuid}(this)">
          <option value="heatmap" selected="selected">Input/Output - Heatmap</option>
          <option value="saliency-plot">Saliency Plot</option>
        </select>
      </div>
      <div id="{uuid}_content" style="padding:15px;border-style:solid;margin:5px;">
          <div id = "{uuid}_saliency_plot_container" class="{uuid}_viz_container" style="display:none">
              {saliency_plot_markup}
          </div>

          <div id = "{uuid}_heatmap_container" class="{uuid}_viz_container">
              {heatmap_markup}
          </div>
      </div>
    </div>
    </html>
    """

    javascript = f"""
    <script>
        function selectVizType_{uuid}(selectObject) {{

          /* Hide all viz */

            var elements = document.getElementsByClassName("{uuid}_viz_container")
          for (var i = 0; i < elements.length; i++){{
              elements[i].style.display = 'none';
          }}

          var value = selectObject.value;
          if ( value === "saliency-plot" ){{
              document.getElementById('{uuid}_saliency_plot_container').style.display  = "block";
          }}
          else if ( value === "heatmap" ) {{
              document.getElementById('{uuid}_heatmap_container').style.display  = "block";
          }}
        }}
    </script>
    """

    _ipython_display_html(javascript + html)


def saliency_plot(shap_values):
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    unpacked_values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids = process_shap_values(
        shap_values.data, unpacked_values[:, 0], 1, "", clustering, True
    )

    def compress_shap_matrix(shap_matrix, group_sizes):
        compressed_matrix = np.zeros((group_sizes.shape[0], shap_matrix.shape[1]))
        counter = 0
        for index in range(len(group_sizes)):
            compressed_matrix[index, :] = np.sum(shap_matrix[counter : counter + group_sizes[index], :], axis=0)
            counter += group_sizes[index]

        return compressed_matrix

    compressed_shap_matrix = compress_shap_matrix(shap_values.values, group_sizes)

    # generate background colors of saliency plot

    def get_colors(shap_values):
        input_colors = []
        cmax = max(abs(compressed_shap_matrix.min()), abs(compressed_shap_matrix.max()))
        for row_index in range(compressed_shap_matrix.shape[0]):
            input_colors_row = []
            for col_index in range(compressed_shap_matrix.shape[1]):
                scaled_value = 0.5 + 0.5 * compressed_shap_matrix[row_index, col_index] / cmax
                color = colors.red_transparent_blue(scaled_value)
                color = "rgba" + str((color[0] * 255, color[1] * 255, color[2] * 255, color[3]))
                input_colors_row.append(color)
            input_colors.append(input_colors_row)

        return input_colors

    model_output = shap_values.output_names

    input_colors = get_colors(shap_values)

    out = '<table border = "1" cellpadding = "5" cellspacing = "5" style="overflow-x:scroll;display:block;">'

    # add top row containing input tokens
    out += "<tr>"
    out += "<th></th>"

    for j in range(compressed_shap_matrix.shape[0]):
        out += (
            "<th>"
            + tokens[j].replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "").replace("▁", "").replace("Ġ", "")
            + "</th>"
        )
    out += "</tr>"

    for row_index in range(compressed_shap_matrix.shape[1]):
        out += "<tr>"
        out += (
            "<th>"
            + model_output[row_index]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            .replace("▁", "")
            .replace("Ġ", "")
            + "</th>"
        )
        for col_index in range(compressed_shap_matrix.shape[0]):
            out += (
                '<th style="background:'
                + input_colors[col_index][row_index]
                + '">'
                + str(round(compressed_shap_matrix[col_index][row_index], 3))
                + "</th>"
            )
        out += "</tr>"

    out += "</table>"

    saliency_plot_html = f"""
        <div id="{uuid}_saliency_plot" class="{uuid}_viz_content">
            <div style="margin:5px;font-family:sans-serif;font-weight:bold;">
                <span style="font-size: 20px;"> Saliency Plot </span>
                <br>
                x-axis: Output Text
                <br>
                y-axis: Input Text
            </div>
            {out}
        </div>
    """
    return saliency_plot_html


def heatmap(shap_values):
    # constants

    TREE_NODE_KEY_TOKENS = "tokens"
    TREE_NODE_KEY_CHILDREN = "children"

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    def get_color(shap_value, cmax):
        scaled_value = 0.5 + 0.5 * shap_value / cmax
        color = colors.red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
        return color

    def process_text_to_text_shap_values(shap_values):
        processed_values = []

        unpacked_values, clustering = unpack_shap_explanation_contents(shap_values)
        max_val = 0

        for index, output_token in enumerate(shap_values.output_names):
            tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids = process_shap_values(
                shap_values.data, unpacked_values[:, index], 1, "", clustering, True
            )
            processed_value = {
                "tokens": tokens,
                "values": values,
                "group_sizes": group_sizes,
                "token_id_to_node_id_mapping": token_id_to_node_id_mapping,
                "collapsed_node_ids": collapsed_node_ids,
            }

            processed_values.append(processed_value)
            max_val = max(max_val, np.max(values))

        return processed_values, max_val

    # unpack input tokens and output tokens
    model_input = shap_values.data
    model_output = shap_values.output_names

    processed_values, max_val = process_text_to_text_shap_values(shap_values)

    # generate dictionary containing precomputed background colors and shap values which are addressable by html token ids
    colors_dict = {}
    shap_values_dict = {}
    token_id_to_node_id_mapping = {}
    cmax = max(abs(shap_values.values.min()), abs(shap_values.values.max()), max_val)

    # input token -> output token color and label value mapping

    for row_index in range(len(model_input)):
        color_values = {}
        shap_values_list = {}

        for col_index in range(len(model_output)):
            color_values[uuid + "_output_flat_token_" + str(col_index)] = "rgba" + str(
                get_color(shap_values.values[row_index][col_index], cmax)
            )
            shap_values_list[uuid + "_output_flat_value_label_" + str(col_index)] = round(
                shap_values.values[row_index][col_index], 3
            )

        colors_dict[f"{uuid}_input_node_{row_index}_content"] = color_values
        shap_values_dict[f"{uuid}_input_node_{row_index}_content"] = shap_values_list

    # output token -> input token color and label value mapping

    for col_index in range(len(model_output)):
        color_values = {}
        shap_values_list = {}

        for row_index in range(processed_values[col_index]["collapsed_node_ids"].shape[0]):
            color_values[
                uuid + "_input_node_" + str(processed_values[col_index]["collapsed_node_ids"][row_index]) + "_content"
            ] = "rgba" + str(get_color(processed_values[col_index]["values"][row_index], cmax))
            shap_label_value_str = str(round(processed_values[col_index]["values"][row_index], 3))
            if processed_values[col_index]["group_sizes"][row_index] > 1:
                shap_label_value_str += "/" + str(processed_values[col_index]["group_sizes"][row_index])

            shap_values_list[
                uuid + "_input_node_" + str(processed_values[col_index]["collapsed_node_ids"][row_index]) + "_label"
            ] = shap_label_value_str

        colors_dict[uuid + "_output_flat_token_" + str(col_index)] = color_values
        shap_values_dict[uuid + "_output_flat_token_" + str(col_index)] = shap_values_list

        token_id_to_node_id_mapping_dict = {}

        for index, node_id in enumerate(processed_values[col_index]["token_id_to_node_id_mapping"].tolist()):
            token_id_to_node_id_mapping_dict[f"{uuid}_input_node_{index}_content"] = (
                f"{uuid}_input_node_{int(node_id)}_content"
            )

        token_id_to_node_id_mapping[uuid + "_output_flat_token_" + str(col_index)] = token_id_to_node_id_mapping_dict

    # convert python dictionary into json to be inserted into the runtime javascript environment
    colors_json = json.dumps(colors_dict)
    shap_values_json = json.dumps(shap_values_dict)
    token_id_to_node_id_mapping_json = json.dumps(token_id_to_node_id_mapping)

    javascript_values = (
        "<script> "
        f"colors_{uuid} = {colors_json}\n"
        f" shap_values_{uuid} = {shap_values_json}\n"
        f" token_id_to_node_id_mapping_{uuid} = {token_id_to_node_id_mapping_json}\n"
        "</script> \n "
    )

    def generate_tree(shap_values):
        num_tokens = len(shap_values.data)
        token_list = {}

        for index in range(num_tokens):
            node_content = {}
            node_content[TREE_NODE_KEY_TOKENS] = shap_values.data[index]
            node_content[TREE_NODE_KEY_CHILDREN] = {}
            token_list[str(index)] = node_content

        counter = num_tokens
        for pair in shap_values.clustering:
            first_node = str(int(pair[0]))
            second_node = str(int(pair[1]))

            new_node_content = {}
            new_node_content[TREE_NODE_KEY_CHILDREN] = {
                first_node: token_list[first_node],
                second_node: token_list[second_node],
            }

            token_list[str(counter)] = new_node_content
            counter += 1

            del token_list[first_node]
            del token_list[second_node]

        return token_list

    tree = generate_tree(shap_values)

    # generates the input token html elements
    # each element contains the label value (initially hidden) and the token text

    input_text_html = ""

    def populate_input_tree(input_index, token_list_subtree, input_text_html):
        content = token_list_subtree[input_index]
        input_text_html += (
            f'<div id="{uuid}_input_node_{input_index}_container" style="display:inline;text-align:center">'
        )

        input_text_html += (
            f'<div id="{uuid}_input_node_{input_index}_label" style="display:none; padding-top: 0px; font-size:12px;">'
        )

        input_text_html += "</div>"

        if token_list_subtree[input_index][TREE_NODE_KEY_CHILDREN]:
            input_text_html += f'<div id="{uuid}_input_node_{input_index}_content" style="display:inline;">'
            for child_index, child_content in token_list_subtree[input_index][TREE_NODE_KEY_CHILDREN].items():
                input_text_html = populate_input_tree(
                    child_index, token_list_subtree[input_index][TREE_NODE_KEY_CHILDREN], input_text_html
                )
            input_text_html += "</div>"
        else:
            input_text_html += (
                f'<div id="{uuid}_input_node_{input_index}_content"'
                "style='display: inline; background:transparent; border-radius: 3px; padding: 0px;cursor: default;cursor: pointer;'"
                f'onmouseover="onMouseHoverFlat_{uuid}(this.id)" '
                f'onmouseout="onMouseOutFlat_{uuid}(this.id)" '
                f'onclick="onMouseClickFlat_{uuid}(this.id)" '
                ">"
            )
            input_text_html += (
                content[TREE_NODE_KEY_TOKENS]
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ##", "")
                .replace("▁", "")
                .replace("Ġ", "")
            )
            input_text_html += "</div>"

        input_text_html += "</div>"

        return input_text_html

    input_text_html = populate_input_tree(list(tree.keys())[0], tree, input_text_html)

    # generates the output token html elements
    output_text_html = ""

    for i in range(len(model_output)):
        output_text_html += (
            "<div style='display:inline; text-align:center;'>"
            f"<div id='{uuid}_output_flat_value_label_{i}'"
            "style='display:none;color: #999; padding-top: 0px; font-size:12px;'>"
            "</div>"
            f"<div id='{uuid}_output_flat_token_{i}'"
            "style='display: inline; background:transparent; border-radius: 3px; padding: 0px;cursor: default;cursor: pointer;'"
            f'onmouseover="onMouseHoverFlat_{uuid}(this.id)" '
            f'onmouseout="onMouseOutFlat_{uuid}(this.id)" '
            f'onclick="onMouseClickFlat_{uuid}(this.id)" '
            ">"
            + model_output[i]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            .replace("▁", "")
            .replace("Ġ", "")
            + " </div>"
            + "</div>"
        )

    heatmap_html = f"""
        <div id="{uuid}_heatmap" class="{uuid}_viz_content">
          <div id="{uuid}_heatmap_header" style="padding:15px;margin:5px;font-family:sans-serif;font-weight:bold;">
            <div style="display:inline">
              <span style="font-size: 20px;"> Input/Output - Heatmap </span>
            </div>
            <div style="display:inline;float:right">
              Layout :
              <select name="alignment" id="{uuid}_alignment" onchange="selectAlignment_{uuid}(this)">
                <option value="left-right" selected="selected">Left/Right</option>
                <option value="top-bottom">Top/Bottom</option>
              </select>
            </div>
          </div>
          <div id="{uuid}_heatmap_content" style="display:flex;">
            <div id="{uuid}_input_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_input_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Input Text
              </div>
              <div id="{uuid}_input_content" style="margin:5px;font-family:sans-serif;">
                  {input_text_html}
              </div>
            </div>
            <div id="{uuid}_output_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_output_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Output Text
              </div>
              <div id="{uuid}_output_content" style="margin:5px;font-family:sans-serif;">
                  {output_text_html}
              </div>
            </div>
          </div>
        </div>
    """

    heatmap_javascript = f"""
        <script>
            function selectAlignment_{uuid}(selectObject) {{
                var value = selectObject.value;
                if ( value === "left-right" ){{
                  document.getElementById('{uuid}_heatmap_content').style.display  = "flex";
                }}
                else if ( value === "top-bottom" ) {{
                  document.getElementById('{uuid}_heatmap_content').style.display  = "inline";
                }}
            }}

            var {uuid}_heatmap_flat_state = null;

            function onMouseHoverFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    setBackgroundColors_{uuid}(id);
                    document.getElementById(id).style.backgroundColor  = "grey";
                }}

                if (getIdSide_{uuid}(id) === 'input' && getIdSide_{uuid}({uuid}_heatmap_flat_state) === 'output'){{

                    label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                    if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none'){{
                        document.getElementById(label_content_id).style.textShadow = "0px 0px 1px #000000";
                    }}

                }}

            }}

            function onMouseOutFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    cleanValuesAndColors_{uuid}(id);
                    document.getElementById(id).style.backgroundColor  = "transparent";
                }}

                if (getIdSide_{uuid}(id) === 'input' && getIdSide_{uuid}({uuid}_heatmap_flat_state) === 'output'){{

                    label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                    if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none'){{
                        document.getElementById(label_content_id).style.textShadow = "inherit";
                    }}

                }}

            }}

            function onMouseClickFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === id) {{

                    // If the clicked token was already selected

                    document.getElementById(id).style.backgroundColor  = "transparent";
                    cleanValuesAndColors_{uuid}(id);
                    {uuid}_heatmap_flat_state = null;
                }}
                else {{
                    if ({uuid}_heatmap_flat_state === null) {{

                        // No token previously selected, new token clicked on

                        cleanValuesAndColors_{uuid}(id)
                        {uuid}_heatmap_flat_state = id;
                        document.getElementById(id).style.backgroundColor  = "grey";
                        setLabelValues_{uuid}(id);
                        setBackgroundColors_{uuid}(id);
                    }}
                    else {{
                        if (getIdSide_{uuid}({uuid}_heatmap_flat_state) === getIdSide_{uuid}(id)) {{

                            // User clicked a token on the same side as the currently selected token

                            cleanValuesAndColors_{uuid}({uuid}_heatmap_flat_state)
                            document.getElementById({uuid}_heatmap_flat_state).style.backgroundColor  = "transparent";
                            {uuid}_heatmap_flat_state = id;
                            document.getElementById(id).style.backgroundColor  = "grey";
                            setLabelValues_{uuid}(id);
                            setBackgroundColors_{uuid}(id);
                        }}
                        else{{

                            if (getIdSide_{uuid}(id) === 'input') {{
                                label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                                if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none') {{
                                    document.getElementById(label_content_id).previousElementSibling.style.display = 'block';
                                    document.getElementById(label_content_id).parentNode.style.display = 'inline-block';
                                    document.getElementById(label_content_id).style.textShadow = "0px 0px 1px #000000";
                                  }}
                                else {{
                                    document.getElementById(label_content_id).previousElementSibling.style.display = 'none';
                                    document.getElementById(label_content_id).parentNode.style.display = 'inline';
                                    document.getElementById(label_content_id).style.textShadow  = "inherit";
                                  }}

                            }}
                            else {{
                                if (document.getElementById(id).previousElementSibling.style.display == 'none') {{
                                    document.getElementById(id).previousElementSibling.style.display = 'block';
                                    document.getElementById(id).parentNode.style.display = 'inline-block';
                                  }}
                                else {{
                                    document.getElementById(id).previousElementSibling.style.display = 'none';
                                    document.getElementById(id).parentNode.style.display = 'inline';
                                  }}
                            }}

                        }}
                    }}

                }}
            }}

            function setLabelValues_{uuid}(id) {{
                for(const token in shap_values_{uuid}[id]){{
                    document.getElementById(token).innerHTML = shap_values_{uuid}[id][token];
                    document.getElementById(token).nextElementSibling.title = 'SHAP Value : ' + shap_values_{uuid}[id][token];
                }}
            }}

            function setBackgroundColors_{uuid}(id) {{
                for(const token in colors_{uuid}[id]){{
                    document.getElementById(token).style.backgroundColor  = colors_{uuid}[id][token];
                }}
            }}

            function cleanValuesAndColors_{uuid}(id) {{
                for(const token in shap_values_{uuid}[id]){{
                    document.getElementById(token).innerHTML = "";
                    document.getElementById(token).nextElementSibling.title = "";
                }}
                 for(const token in colors_{uuid}[id]){{
                    document.getElementById(token).style.backgroundColor  = "transparent";
                    document.getElementById(token).previousElementSibling.style.display = 'none';
                    document.getElementById(token).parentNode.style.display = 'inline';
                    document.getElementById(token).style.textShadow  = "inherit";
                }}
            }}

            function getIdSide_{uuid}(id) {{
                if (id === null) {{
                    return 'null'
                }}
                return id.split("_")[1];
            }}
        </script>
    """

    return heatmap_html + heatmap_javascript + javascript_values


def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return np.array(values), clustering


def _ipython_display_html(data):
    """Check IPython is installed, then display HTML"""
    if not have_ipython:
        msg = "IPython is required for this function but is not installed. Fix this with `pip install ipython`."
        raise ImportError(msg)
    return ipython_display(HTML(data))
