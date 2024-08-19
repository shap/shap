### ------------ vendored from skimage.color.colorconv -------------
# This is a small chunk of code from the skimage package. It is reproduced
# here because all we need is a couple color conversion routines (lab2rgb, lch2lab)
# and adding all of skimage as dependency is really heavy.


# Copyright: 2009-2022 the scikit-image team
# All rights reserved.

# License: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# .
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from warnings import warn

import numpy as np
from scipy import linalg


def _prepare_colorarray(arr, *, channel_axis=-1):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.shape[channel_axis] != 3:
        msg = f"the input array must have size 3 along `channel_axis`, got {arr.shape}"
        raise ValueError(msg)

    return arr.astype(np.float64)


xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = linalg.inv(xyz_from_rgb)


def _convert(matrix, arr):
    """Do the color space conversion.

    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : (..., C=3, ...) array_like
        The input array. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The converted array. Same dimensions as input.
    """
    arr = _prepare_colorarray(arr)

    return arr @ matrix.T.astype(arr.dtype)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : (..., C=3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def _lab2xyz(lab):
    """Convert CIE-LAB to XYZ color space.

    Internal function for :func:`~.lab2xyz` and others. In addition to the
    converted image, return the number of invalid pixels in the Z channel for
    correct warning propagation.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in XYZ format. Same dimensions as input.
    n_invalid : int
        Number of invalid pixels in the Z channel after conversion.
    """
    arr = _prepare_colorarray(lab, channel_axis=-1).copy()

    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    invalid = np.atleast_1d(z < 0).nonzero()
    n_invalid = invalid[0].size
    if n_invalid != 0:
        # Warning should be emitted by caller
        if z.ndim > 0:
            z[invalid] = 0
        else:
            z = 0

    out = np.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = np.array([0.95047, 1.0, 1.08883])
    out *= xyz_ref_white
    return out, n_invalid


def lab2rgb(lab):
    """Convert image in CIE-LAB to sRGB color space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in sRGB color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., C=3, ...).
    """
    xyz, n_invalid = _lab2xyz(lab)
    if n_invalid != 0:
        warn(
            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )
    return xyz2rgb(xyz)


def lch2lab(lch):
    """Convert image in CIE-LCh to CIE-LAB color space.

    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color
    space.

    Parameters
    ----------
    lch : (..., C=3, ...) array_like
        The input image in CIE-LCh color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the C values range from 0 to 100;
        the h values range from 0 to ``2*pi``.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LAB format, of same shape as input.
    """
    lch = _prepare_lab_array(lch)

    c, h = lch[..., 1], lch[..., 2]
    lch[..., 1], lch[..., 2] = c * np.cos(h), c * np.sin(h)
    return lch


def _prepare_lab_array(arr):
    """Ensure input for lab2lch and lch2lab is well-formed.

    Input array must be in floating point and have at least 3 elements in the
    last dimension. Returns a new array by default.
    """
    arr = np.asarray(arr, dtype=np.float64)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError("Input image has less than 3 channels.")
    return arr
