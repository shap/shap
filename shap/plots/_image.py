from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from .._explanation import Explanation
from ..utils._legacy import kmeans
from . import colors

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def image(
    shap_values: Explanation | np.ndarray | list[np.ndarray],
    pixel_values: np.ndarray | None = None,
    labels: list[str] | list[list[str]] | np.ndarray | None = None,
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

    # labels: (rows (images), columns (top_k classes) ) or (1, columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels)
        labels = labels.reshape(-1, len(shap_values))

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
                # Add labels if there are labels for each sample, or if not, only for the first row
                if labels.shape[0] > 1 or row == 0:
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


def image_to_text(
    shap_values: Explanation,
    max_display: int = 10,
    cmap: str | Colormap | None = colors.red_transparent_blue,
    vmax: float | None = None,
    show: bool = True,
):
    """Plots SHAP values for image inputs with text outputs.

    Parameters
    ----------
    shap_values : Explanation
        An Explanation object with values of shape
        (height, width, channels, num_tokens) for a single instance or
        (num_instances, height, width, channels, num_tokens) for a batch.

    max_display : int
        Maximum number of output tokens to show.

    cmap: str or matplotlib.colors.Colormap
        Colormap to use when plotting the SHAP values.

    vmax: float or None
        Sets the colormap scale for SHAP values from ``-vmax`` to ``+vmax``.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    """
    if not isinstance(shap_values, Explanation):
        raise TypeError("image_to_text requires an Explanation object as input.")

    # handle batched input
    if len(shap_values.values.shape) == 5:
        for i in range(shap_values.values.shape[0]):
            image_to_text(shap_values[i], max_display=max_display, cmap=cmap, vmax=vmax, show=show)
        return

    image_data = shap_values.data
    values = shap_values.values
    output_names = shap_values.output_names
    num_tokens = min(max_display, values.shape[-1])

    # get a grayscale version of the image
    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
        image_gray = 0.2989 * image_data[:, :, 0] + 0.5870 * image_data[:, :, 1] + 0.1140 * image_data[:, :, 2]
    elif len(image_data.shape) == 3:
        image_gray = image_data.mean(axis=2)
    else:
        image_gray = image_data

    # sum shap values across color channels
    if len(values.shape) == 4:
        sv_per_token = values.sum(axis=2)
    else:
        sv_per_token = values

    max_val = np.nanpercentile(np.abs(sv_per_token[:, :, :num_tokens]), 99.9) if vmax is None else vmax

    ncols = num_tokens + 1
    fig_size = np.array([3 * ncols, 3.0])
    if fig_size[0] > 20:
        fig_size *= 20 / fig_size[0]

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=fig_size, squeeze=False)

    # show original image
    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
        axes[0, 0].imshow(image_data.astype(np.uint8) if image_data.max() > 1 else image_data)
    else:
        axes[0, 0].imshow(image_gray, cmap=plt.get_cmap("gray"))
    axes[0, 0].set_title("Input Image", fontsize=10)
    axes[0, 0].axis("off")

    # plot per-token shap heatmaps
    im = None
    for i in range(num_tokens):
        ax = axes[0, i + 1]
        sv = sv_per_token[:, :, i]
        ax.imshow(
            image_gray,
            cmap=plt.get_cmap("gray"),
            alpha=0.15,
            extent=(-1, sv.shape[1], sv.shape[0], -1),
        )
        im = ax.imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
        token_label = str(output_names[i]).replace("▁", "").replace("Ġ", "").replace(" ##", "")
        ax.set_title(token_label, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    if im is not None:
        cb = fig.colorbar(
            im,
            ax=np.ravel(axes).tolist(),
            label="SHAP value",
            orientation="horizontal",
            aspect=fig_size[0] / 0.2,
        )
        cb.outline.set_visible(False)  # type: ignore
    if show:
        plt.show()
