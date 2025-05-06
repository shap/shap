from __future__ import annotations

import json
import random
import string
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import HTML, display

    have_ipython = True
except ImportError:
    have_ipython = False

from .._explanation import Explanation
from ..utils import ordinal_str
from ..utils._legacy import kmeans
from . import colors

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def image(
    shap_values: Explanation | np.ndarray | list[np.ndarray],
    pixel_values: np.ndarray | None = None,
    labels: list[str] | np.ndarray | None = None,
    true_labels: list | None = None,
    width: int | None = 20,
    aspect: float | None = 0.2,
    hspace: float | Literal["auto"] | None = 0.2,
    labelpad: float | None = None,
    cmap: str | Colormap | None = colors.red_transparent_blue,
    vmax: float | None = None,
    show: bool | None = True,
):
    """Plots SHAP values for image inputs.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shape
        (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being
        explained.

    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image.
        It should be the same
        shape as each array in the ``shap_values`` list of arrays.

    labels : list or np.ndarray
        List or ``np.ndarray`` (# samples x top_k classes) of names for each of the
        model outputs that are being explained.

    true_labels: list
        List of a true image labels to plot.

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    cmap: str or matplotlib.colors.Colormap
        Colormap to use when plotting the SHAP values.

    vmax: Optional float
        Sets the colormap scale for SHAP values from ``-vmax`` to ``+vmax``.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------
    See `image plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/image.html>`_.

    """
    # support passing an explanation object
    if isinstance(shap_values, Explanation):
        shap_exp: Explanation = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = cast("list[np.ndarray]", [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])])
        elif len(shap_exp.output_dims) == 0:
            shap_values = cast("list[np.ndarray]", [shap_exp.values])
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = cast("np.ndarray", shap_exp.data)
        if labels is None:
            labels = cast("list[str]", shap_exp.output_names)
    else:
        assert isinstance(pixel_values, np.ndarray), (
            "The input pixel_values must be a numpy array or an Explanation object must be provided!"
        )

    # multi_output = True
    if not isinstance(shap_values, list):
        # multi_output = False
        shap_values = cast("list[np.ndarray]", [shap_values])

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    # if labels is not None:
    #     labels = np.array(labels)
    #     if labels.shape[0] != shap_values[0].shape[0] and labels.shape[0] == len(shap_values):
    #         labels = np.tile(np.array([labels]), shap_values[0].shape[0])
    #     assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    #     if multi_output:
    #         assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
    #     else:
    #         assert len(labels[0].shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {"pad": labelpad}

    # plot our explanations
    x: np.ndarray = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size, squeeze=False)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # if x_curr.max() > 1:
        #     x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1))
            )
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap("gray"))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis("off")
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()

        max_val = np.nanpercentile(abs_vals, 99.9) if vmax is None else vmax

        for i in range(len(shap_values)):
            if labels is not None:
                if row == 0:
                    axes[row, i + 1].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(
                x_curr_gray, cmap=plt.get_cmap("gray"), alpha=0.15, extent=(-1, sv.shape[1], sv.shape[0], -1)
            )
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis("off")
    if hspace == "auto":
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(
        im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0] / aspect
    )
    cb.outline.set_visible(False)  # type: ignore
    if show:
        plt.show()


def image_to_text(shap_values):
    """Plots SHAP values for image inputs with test outputs.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# width x height x channels x num output tokens). One array
        for each sample

    """
    if not have_ipython:  # pragma: no cover
        msg = "IPython is required for this function but is not installed. Fix this with `pip install ipython`."
        raise ImportError(msg)

    if len(shap_values.values.shape) == 5:
        for i in range(shap_values.values.shape[0]):
            display(HTML(f"<br/><b>{ordinal_str(i)} instance:</b><br/>"))
            image_to_text(shap_values[i])

        return

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # creating input html tokens

    model_output = shap_values.output_names

    output_text_html = ""

    for i in range(model_output.shape[0]):
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

    # computing gray scale images
    image_data = shap_values.data
    image_height = image_data.shape[0]
    image_width = image_data.shape[1]

    # computing gray scale image
    image_data_gray_scale = np.ones((image_height, image_width, 4)) * 255 * 0.5
    image_data_gray_scale[:, :, 0] = np.mean(image_data, axis=2).astype(int)
    image_data_gray_scale[:, :, 1] = image_data_gray_scale[:, :, 0]
    image_data_gray_scale[:, :, 2] = image_data_gray_scale[:, :, 0]

    # computing shap color values for every pixel and for every output token

    shap_values_color_maps = shap_values.values[:, :, 0, :]
    max_val = np.nanpercentile(np.abs(shap_values.values), 99.9)

    shap_values_color_dict = {}

    for index in range(model_output.shape[0]):
        shap_values_color_dict[f"{uuid}_output_flat_token_{index}"] = (
            (colors.red_transparent_blue(0.5 + 0.5 * shap_values_color_maps[:, :, index] / max_val) * 255)
            .astype(int)
            .tolist()
        )

    # converting to json to be read in javascript

    image_data_json = json.dumps(shap_values.data.astype(int).tolist())
    shap_values_color_dict_json = json.dumps(shap_values_color_dict)
    image_data_gray_scale_json = json.dumps(image_data_gray_scale.astype(int).tolist())

    image_viz_html = f"""

        <div id="{uuid}_image_viz" class="{uuid}_image_viz_content">
          <div id="{uuid}_image_viz_header" style="padding:15px;margin:5px;font-family:sans-serif;font-weight:bold;">
            <div style="display:inline">
              <span style="font-size: 20px;"> Input/Output - Heatmap </span>
            </div>
          </div>
          <div id="{uuid}_image_viz_content" style="display:flex;">
            <div id="{uuid}_image_viz_input_container" style="padding:15px;border-style:solid;margin:5px;flex:2;">
              <div id="{uuid}_image_viz_input_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Input Image
              </div>
              <div id="{uuid}_image_viz_input_content" style="margin:5px;font-family:sans-serif;">
                  <canvas id="{uuid}_image_canvas" style="cursor:grab;width:100%;max-height:500px;"></canvas>
                  <br>
                  <br>
                  <div id="{uuid}_tools">
                      <div id="{uuid}_zoom">
                        <span style="font-size:12px;margin-right:15px;"> Zoom </span>
                        <button id="{uuid}_minus_button" class="zoom-button" onclick="{uuid}_zoom(-1)" style="background-color: #555555;color: white; border:none;font-size:15px;">-</button>
                        <button id="{uuid}_plus_button" class="zoom-button" onclick="{uuid}_zoom(1)" style="background-color: #555555;color: white; border:none;font-size:15px;">+</button>
                        <button id="{uuid}_reset_button" class="zoom-button" onclick="{uuid}_reset()" style="background-color: #555555;color: white; border:none;font-size:15px;"> Reset </button>
                      </div>
                      <br>
                      <div id="{uuid}_opacity" style="display:none">
                      <span style="font-size:12px;margin-right:15px;"> Shap-Overlay Opacity </span>
                      <input type="range" min="1" max="100" value="35" style="width:100px" oninput="{uuid}_set_opacity(this.value)">
                      </div>
                  </div>
              </div>
            </div>
            <div id="{uuid}_image_viz_output_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_image_viz_output_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Output Text
              </div>
              <div id="{uuid}_image_viz_output_content" style="margin:5px;font-family:sans-serif;">
                  {output_text_html}
              </div>
            </div>
          </div>
        </div>

    """

    image_viz_script = f"""
        <script>

            var {uuid}_heatmap_flat_state = null;
            var {uuid}_opacity = 0.35

            function onMouseHoverFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor  = "grey";
                    {uuid}_update_image_and_overlay(id);
                }}
            }}

            function onMouseOutFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor  = "transparent";
                    {uuid}_update_image_and_overlay(null);
                }}
            }}

            function onMouseClickFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor  = "grey";
                    document.getElementById('{uuid}_opacity').style.display  = "block";
                    {uuid}_update_image_and_overlay(id);
                    {uuid}_heatmap_flat_state = id;
                }}
                else {{
                    if ({uuid}_heatmap_flat_state === id) {{
                        document.getElementById(id).style.backgroundColor  = "transparent";
                        document.getElementById('{uuid}_opacity').style.display  = "none";
                        {uuid}_update_image_and_overlay(null);
                        {uuid}_heatmap_flat_state = null;
                    }}
                    else {{
                        document.getElementById({uuid}_heatmap_flat_state).style.backgroundColor  = "transparent";
                        document.getElementById(id).style.backgroundColor  = "grey";
                        {uuid}_update_image_and_overlay(id)
                        {uuid}_heatmap_flat_state = id
                    }}
                }}
            }}

            const {uuid}_image_data_matrix = {image_data_json};
            const {uuid}_image_data_gray_scale = {image_data_gray_scale_json};
            const {uuid}_image_height = {image_height};
            const {uuid}_image_width = {image_width};
            const {uuid}_shap_values_color_dict = {shap_values_color_dict_json};

            {uuid}_canvas = document.getElementById('{uuid}_image_canvas');
            {uuid}_context = {uuid}_canvas.getContext('2d');

            var {uuid}_imageData = {uuid}_convert_image_matrix_to_data({uuid}_image_data_matrix, {image_height}, {image_width}, {uuid}_context);
            var {uuid}_currImagData = {uuid}_imageData;


            {uuid}_trackTransforms({uuid}_context);
            initial_scale_factor = Math.min({uuid}_canvas.height/{uuid}_image_height,{uuid}_canvas.width/{uuid}_image_width);
            {uuid}_context.scale(initial_scale_factor, initial_scale_factor);

            function {uuid}_update_image_and_overlay(selected_id) {{
                if (selected_id == null) {{
                    {uuid}_currImagData = {uuid}_imageData;
                    {uuid}_redraw();
                }}
                else {{
                    {uuid}_currImagData = {uuid}_blend_image_shap_map({uuid}_image_data_gray_scale, {uuid}_shap_values_color_dict[selected_id], {image_height}, {image_width}, {uuid}_opacity, {uuid}_context);
                    {uuid}_redraw();
                }}
            }}

            function {uuid}_set_opacity(value) {{
                {uuid}_opacity = value/100;

                if ({uuid}_heatmap_flat_state !== null ) {{
                    {uuid}_currImagData = {uuid}_blend_image_shap_map({uuid}_image_data_gray_scale, {uuid}_shap_values_color_dict[{uuid}_heatmap_flat_state], {image_height}, {image_width}, {uuid}_opacity, {uuid}_context);
                    {uuid}_redraw();
                }}
            }}

            function {uuid}_redraw() {{

                // Clear the entire canvas
                var p1 = {uuid}_context.transformedPoint(0, 0);
                var p2 = {uuid}_context.transformedPoint({uuid}_canvas.width, {uuid}_canvas.height);
                {uuid}_context.clearRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

                {uuid}_context.save();
                {uuid}_context.setTransform(1, 0, 0, 1, 0, 0);
                {uuid}_context.clearRect(0, 0, {uuid}_canvas.width, {uuid}_canvas.height);
                {uuid}_context.restore();

                createImageBitmap({uuid}_currImagData, {{ premultiplyAlpha: 'premultiply' }}).then(function(imgBitmap) {{
                    {uuid}_context.drawImage(imgBitmap, 0, 0);
                }});
            }}
            {uuid}_redraw();
            {uuid}_context.save();

            var lastX = {uuid}_canvas.width / 2,
                lastY = {uuid}_canvas.height / 2;

            var dragStart, dragged;

            {uuid}_canvas.addEventListener('mousedown', function(evt) {{
                document.body.style.mozUserSelect = document.body.style.webkitUserSelect = document.body.style.userSelect = 'none';
                lastX = evt.offsetX || (evt.pageX - {uuid}_canvas.offsetLeft);
                lastY = evt.offsetY || (evt.pageY - {uuid}_canvas.offsetTop);
                dragStart = {uuid}_context.transformedPoint(lastX, lastY);
                dragged = false;
                document.getElementById('{uuid}_image_canvas').style.cursor = 'grabbing';
            }}, false);

            {uuid}_canvas.addEventListener('mousemove', function(evt) {{
                lastX = evt.offsetX || (evt.pageX - {uuid}_canvas.offsetLeft);
                lastY = evt.offsetY || (evt.pageY - {uuid}_canvas.offsetTop);
                dragged = true;
                if (dragStart) {{
                    var pt = {uuid}_context.transformedPoint(lastX, lastY);
                    {uuid}_context.translate(pt.x - dragStart.x, pt.y - dragStart.y);
                    {uuid}_redraw();
                }}
            }}, false);

            {uuid}_canvas.addEventListener('mouseup', function(evt) {{
                dragStart = null;
                document.getElementById('{uuid}_image_canvas').style.cursor = 'grab';
            }}, false);

            var scaleFactor = 1.1;

            var {uuid}_zoom = function(clicks) {{
                var pt = {uuid}_context.transformedPoint(lastX, lastY);
                {uuid}_context.translate(pt.x, pt.y);
                var factor = Math.pow(scaleFactor, clicks);
                {uuid}_context.scale(factor, factor);
                {uuid}_context.translate(-pt.x, -pt.y);
                {uuid}_redraw();
            }}

            var {uuid}_reset = function(clicks) {{
                {uuid}_context.restore();
                {uuid}_redraw();
                {uuid}_context.save();
            }}

            var handleScroll = function(evt) {{
                var delta = evt.wheelDelta ? evt.wheelDelta / 40 : evt.detail ? -evt.detail : 0;
                if (delta) {uuid}_zoom(delta);
                return evt.preventDefault() && false;
            }}

            {uuid}_canvas.addEventListener('DOMMouseScroll', handleScroll, false);
            {uuid}_canvas.addEventListener('mousewheel', handleScroll, false);



            function {uuid}_trackTransforms(ctx) {{
                var svg = document.createElementNS("http://www.w3.org/2000/svg", 'svg');
                var xform = svg.createSVGMatrix();
                ctx.getTransform = function() {{
                    return xform;
                }}

                var savedTransforms = [];
                var save = ctx.save;
                ctx.save = function() {{
                    savedTransforms.push(xform.translate(0, 0));
                    return save.call(ctx);
                }}

                var restore = ctx.restore;
                ctx.restore = function() {{
                    xform = savedTransforms.pop();
                    return restore.call(ctx);
                }}

                var scale = ctx.scale;
                ctx.scale = function(sx, sy) {{
                    xform = xform.scaleNonUniform(sx, sy);
                    return scale.call(ctx, sx, sy);
                }}

                var rotate = ctx.rotate;
                ctx.rotate = function(radians) {{
                    xform = xform.rotate(radians * 180 / Math.PI);
                    return rotate.call(ctx, radians);
                }}

                var translate = ctx.translate;
                ctx.translate = function(dx, dy) {{
                    xform = xform.translate(dx, dy);
                    return translate.call(ctx, dx, dy);
                }}

                var transform = ctx.transform;
                ctx.transform = function(a, b, c, d, e, f) {{
                    var m2 = svg.createSVGMatrix();
                    m2.a = a;
                    m2.b = b;
                    m2.c = c;
                    m2.d = d;
                    m2.e = e;
                    m2.f = f;
                    xform = xform.multiply(m2);
                    return transform.call(ctx, a, b, c, d, e, f);
                }}

                var setTransform = ctx.setTransform;
                ctx.setTransform = function(a, b, c, d, e, f) {{
                    xform.a = a;
                    xform.b = b;
                    xform.c = c;
                    xform.d = d;
                    xform.e = e;
                    xform.f = f;
                    return setTransform.call(ctx, a, b, c, d, e, f);
                }}

                var pt = svg.createSVGPoint();
                ctx.transformedPoint = function(x, y) {{
                    pt.x = x;
                    pt.y = y;
                    return pt.matrixTransform(xform.inverse());
                }}
            }}


            function {uuid}_convert_image_matrix_to_data(image_data_matrix, image_height, image_width, context) {{

                var imageData = context.createImageData(image_height, image_width);

                for(var row_index = 0; row_index < image_height; row_index++) {{
                    for(var col_index = 0; col_index < image_width; col_index++) {{

                        index = (row_index * image_width + col_index) * 4;

                        imageData.data[index + 0] = image_data_matrix[row_index][col_index][0];
                        imageData.data[index + 1] = image_data_matrix[row_index][col_index][1];
                        imageData.data[index + 2] = image_data_matrix[row_index][col_index][2];
                        imageData.data[index + 3] = 255;
                    }}
                }}

                return imageData;
            }}

            function {uuid}_blend_image_shap_map(image_data_matrix, shap_color_map, image_height, image_width, alpha, context) {{
                var blendedImageData = context.createImageData(image_height, image_width);

                for(var row_index = 0; row_index < image_height; row_index++) {{

                    for(var col_index = 0; col_index < image_width; col_index++) {{

                        index = (row_index * image_width + col_index) * 4;

                        blendedImageData.data[index + 0] = image_data_matrix[row_index][col_index][0] * alpha + (shap_color_map[row_index][col_index][0]) * ( 1 - alpha);
                        blendedImageData.data[index + 1] = image_data_matrix[row_index][col_index][1] * alpha + (shap_color_map[row_index][col_index][1]) * ( 1 - alpha);
                        blendedImageData.data[index + 2] = image_data_matrix[row_index][col_index][2] * alpha + (shap_color_map[row_index][col_index][2]) * ( 1 - alpha);
                        blendedImageData.data[index + 3] = image_data_matrix[row_index][col_index][3] * alpha + (shap_color_map[row_index][col_index][3]) * ( 1 - alpha);
                    }}
                }}

                return blendedImageData;
            }}

        </script>
    """

    display(HTML(image_viz_html + image_viz_script))
