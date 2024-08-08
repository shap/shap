### ------------ vendored from skimage.color.colorconv -------------
# This is a small chunk of code from the skimage package. It is reproduced
# here because all we need is a couple color conversion routines (lab2rgb, lch2lab)
# and adding all of skimage as dependency is really heavy.

# Disable linting on vendored code:
# ruff: noqa

# Copyright (C) 2019, the scikit-image team
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

# Other submodules also partially vendored
from .utils import _supported_float_type, channel_as_last_axis
from .dtype import img_as_float32, img_as_float64


def _prepare_colorarray(arr, force_copy=False, *, channel_axis=-1):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.shape[channel_axis] != 3:
        msg = f"the input array must have size 3 along `channel_axis`, " f"got {arr.shape}"
        raise ValueError(msg)

    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = img_as_float32
    else:
        _func = img_as_float64
    return _func(arr, force_copy=force_copy)


xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = linalg.inv(xyz_from_rgb)


_illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
        "R": (1.098466069456375, 1, 0.3558228003436005),
    },
    "B": {
        "2": (0.9909274480248003, 1, 0.8531327322886154),
        "10": (0.9917777147717607, 1, 0.8434930535866175),
        "R": (0.9909274480248003, 1, 0.8531327322886154),
    },
    "C": {
        "2": (0.980705971659919, 1, 1.1822494939271255),
        "10": (0.9728569189782166, 1, 1.1614480488951577),
        "R": (0.980705971659919, 1, 1.1822494939271255),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
        "R": (0.9639501491621826, 1, 0.8241280285499208),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
        "R": (0.9565317453467969, 1, 0.9202554587037198),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),  # This was: `lab_ref_white`
        "10": (0.94809667673716, 1, 1.0730513595166162),
        "R": (0.9532057125493769, 1, 1.0853843816469158),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
        "R": (0.9497220898840717, 1, 1.226393520724154),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0), "R": (1.0, 1.0, 1.0)},
}


def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    """Get the CIE XYZ tristimulus values.

    Given an illuminant and observer, this function returns the CIE XYZ tristimulus
    values [2]_ scaled such that :math:`Y = 1`.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function ``grDevices::convertColor`` [3]_.
    dtype: dtype, optional
        Output data type.

    Returns
    -------
    values : array
        Array with 3 elements :math:`X, Y, Z` containing the CIE XYZ tristimulus values
        of the given illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
    .. [2] https://en.wikipedia.org/wiki/CIE_1931_color_space#Meaning_of_X,_Y_and_Z
    .. [3] https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/convertColor

    Notes
    -----
    The CIE XYZ tristimulus values are calculated from :math:`x, y` [1]_, using the
    formula

    .. math:: X = x / y

    .. math:: Y = 1

    .. math:: Z = (1 - x - y) / y

    The only exception is the illuminant "D65" with aperture angle 2° for
    backward-compatibility reasons.

    Examples
    --------
    Get the CIE XYZ tristimulus values for a "D65" illuminant for a 10 degree field of
    view

    >>> xyz_tristimulus_values(illuminant="D65", observer="10")
    array([0.94809668, 1.        , 1.07305136])
    """
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_illuminants[illuminant][observer], dtype=dtype)  # type: ignore
    except KeyError:
        raise ValueError(f"Unknown illuminant/observer combination " f"(`{illuminant}`, `{observer}`)")


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


@channel_as_last_axis()
def xyz2rgb(xyz, *, channel_axis=-1):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : (..., C=3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2rgb
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_rgb = xyz2rgb(img_xyz)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def _lab2xyz(lab, illuminant, observer):
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
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer)
    out *= xyz_ref_white
    return out, n_invalid


@channel_as_last_axis()
def lab2rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    """Convert image in CIE-LAB to sRGB color space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the a* and b* values range from -128 to 127.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in sRGB color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.
    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
    supported illuminants.

    See Also
    --------
    rgb2lab

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    xyz, n_invalid = _lab2xyz(lab, illuminant, observer)
    if n_invalid != 0:
        warn(
            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )
    return xyz2rgb(xyz)


@channel_as_last_axis()
def lch2lab(lch, *, channel_axis=-1):
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
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LAB format, of same shape as input.

    Raises
    ------
    ValueError
        If `lch` does not have at least 3 channels (i.e., L*, C, and h).

    Notes
    -----
    The h channel (i.e., hue) is expressed as an angle in range ``(0, 2*pi)``.

    See Also
    --------
    lab2lch

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/HCL_color_space
    .. [3] https://en.wikipedia.org/wiki/CIELAB_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2lab, lch2lab, lab2lch
    >>> img = data.astronaut()
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    >>> img_lab2 = lch2lab(img_lch)
    """
    lch = _prepare_lab_array(lch)

    c, h = lch[..., 1], lch[..., 2]
    lch[..., 1], lch[..., 2] = c * np.cos(h), c * np.sin(h)
    return lch


def _prepare_lab_array(arr, force_copy=True):
    """Ensure input for lab2lch and lch2lab is well-formed.

    Input array must be in floating point and have at least 3 elements in the
    last dimension. Returns a new array by default.
    """
    arr = np.asarray(arr)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError("Input image has less than 3 channels.")
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = img_as_float32
    else:
        _func = img_as_float64
    return _func(arr, force_copy=force_copy)
