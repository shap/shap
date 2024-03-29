# TODO: heapq in numba does not yet support Typed Lists so we can move to them yet...
import heapq

import numba.typed
import numpy as np
from numba import njit

from .._serializable import Deserializer, Serializer
from ..utils import assert_import, record_import_error, safe_isinstance
from ..utils._exceptions import DimensionError
from ._masker import Masker

try:
    import cv2
except ImportError as e:
    record_import_error("cv2", "cv2 could not be imported!", e)

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, kernel_size, channels=3, sigma=1, dim=3, numpy=False):
        self.is_numpy = numpy
        super(GaussianSmoothing, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.padding = "same"

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.'
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.

        Returns
        -------
            filtered (torch.Tensor): Filtered output.
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).to(device=self.dummy_param.device)
            self.is_numpy = True
        else:
            self.is_numpy = False
        out = self.conv(input, weight=self.weight, groups=self.groups, padding = self.padding)
        if self.is_numpy:
            return out.cpu().numpy()
        else:
            return out


class Video(Masker):
    """Masks out image regions with blurring or inpainting."""

    def __init__(self, mask_value, shape=None):
        """Build a new Image masker with the given masking value.

        Parameters
        ----------
        mask_value : np.array, "blur(kernel_tsize, kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.
        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
            (c, t, h, w)
        """
        if shape is None:
            if isinstance(mask_value, str):
                raise TypeError("When the mask_value is a string the shape parameter must be given!")
            self.input_shape = mask_value.shape # the (1,) is because we only return a single masked sample to average over
        else:
            self.input_shape = shape

        self.input_mask_value = mask_value

        # This is the shape of the masks we expect
        self.shape = (1, np.prod(self.input_shape)) # the (1, ...) is because we only return a single masked sample to average over

        self.image_data = True

        self.blur_kernel = None
        self._blur_value_cache = None
        if issubclass(type(mask_value), np.ndarray):
            self.mask_value = mask_value.flatten()
        elif isinstance(mask_value, str):
            assert_import("cv2")
            self.mask_value = mask_value
            if mask_value.startswith("blur("):
                self.blur_kernel = tuple(map(int, mask_value[5:-1].split(",")))
                self.gsblur = GaussianSmoothing(self.blur_kernel).to(device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.mask_value = np.ones(self.input_shape).flatten() * mask_value
        self.build_partition_tree()

        # note if this masker can use different background for different samples
        self.fixed_background = not isinstance(self.mask_value, str)

        #self.scratch_mask = np.zeros(self.input_shape[:-1], dtype=bool)
        self.last_xid = None

        # flag that we return outputs that will not get changed by later masking calls
        self.immutable_outputs = True

    def __call__(self, mask, x):

        if safe_isinstance(x, "torch.Tensor"):
            x = x.cpu().numpy()

        if np.prod(x.shape) != np.prod(self.input_shape):
            raise DimensionError("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker constructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.input_shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # we preserve flattened inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.ravel()

        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=bool)

        if isinstance(self.mask_value, str):
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    with torch.no_grad():
                        self._blur_value_cache = self.gsblur(x.reshape(self.input_shape)).ravel()
                    # self._blur_value_cache = cv2.blur(x.reshape(self.input_shape), self.blur_kernel).ravel()
                    #ToDo: Change to Conv3d using https://discuss.pytorch.org/t/use-conv2d-and-conv3d-as-blur-fillter-on-matrix/170026/4
                    self.last_xid = id(x)
                out = x.copy()
                out[~mask] = self._blur_value_cache[~mask]

            elif self.mask_value == "inpaint_telea":
                out = self.inpaint(x, ~mask, "INPAINT_TELEA")
            elif self.mask_value == "inpaint_ns":
                out = self.inpaint(x, ~mask, "INPAINT_NS")
        else:
            out = x.copy()
            out[~mask] = self.mask_value[~mask]

        return (out.reshape(1, *in_shape),)

    def inpaint(self, x, mask, method):
        reshaped_mask = mask.reshape(self.input_shape).astype(np.uint8).max(0)
        if reshaped_mask.sum() == np.prod(self.input_shape[1:]):
            out = x.reshape(self.input_shape).copy()
            out[:] = out.mean((1,2,3))
            return out.ravel()

        ## Note: input shape is H W C previously now its C T H W make sure things good
        back = x.reshape(self.input_shape).copy()
        imgs = []
        for i in range(self.input_shape[1]):
            imgs.append(
                cv2.inpaint(
                    np.moveaxis(back[:, i, :, :], 0, -1).astype(np.uint8),
                    reshaped_mask[i, :, :],
                    inpaintRadius=  3,
                    flags = getattr(cv2, method)
                ).astype(x.dtype).ravel()
            )
        return np.stack(imgs).ravel()

    def inpaint_img(self, x, mask, method):
        """Fill in the masked parts of the image through inpainting."""
        reshaped_mask = mask.reshape(self.input_shape).astype(np.uint8).max(2)
        if reshaped_mask.sum() == np.prod(self.input_shape[:-1]):
            out = x.reshape(self.input_shape).copy()
            out[:] = out.mean((0, 1))
            return out.ravel()

        return cv2.inpaint(
            x.reshape(self.input_shape).astype(np.uint8),
            reshaped_mask,
            inpaintRadius=3,
            flags=getattr(cv2, method)
        ).astype(x.dtype).ravel()

    def build_partition_tree(self):
        """This partitions an image into a herarchical clustering based on axis-aligned splits."""
        tmin = 0
        tmax = self.input_shape[1]
        xmin = 0
        xmax = self.input_shape[2]
        ymin = 0
        ymax = self.input_shape[3]
        zmin = 0
        zmax = self.input_shape[0]

        total_twidth = tmax - tmin
        total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        #numba.typed.List()
        q = numba.typed.List([(0, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, -1, False)])
        M = int((tmax - tmin) * (xmax - xmin) * (ymax - ymin) * (zmax - zmin))
        clustering = np.zeros((M - 1, 4))
        _jit_build_partition_tree(tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, total_xwidth, total_ywidth, total_zwidth, M, clustering, q)
        self.clustering = clustering

    def save(self, out_file):
        """Write a Image masker to a file stream."""
        super().save(out_file)

        # Increment the version number when the encoding changes!
        with Serializer(out_file, "shap.maskers.Image", version=0) as s:
            s.save("mask_value", self.input_mask_value)
            s.save("shape", self.input_shape)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """Load a Image masker from a file stream."""
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Image", min_version=0, max_version=0) as s:
            kwargs["mask_value"] = s.load("mask_value")
            kwargs["shape"] = s.load("shape")
        return kwargs

@njit
def _jit_build_partition_tree(tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, total_xwidth, total_ywidth, total_zwidth, M, clustering, q):
    """This partitions an image into a herarchical clustering based on axis-aligned splits."""
    # heapq.heappush(q, (0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))

    # q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))
    ind = len(clustering) - 1
    while len(q) > 0: # q.empty()
        _, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left =  heapq.heappop(q)
        # _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()

        if parent_ind >= 0:
            clustering[parent_ind, 0 if is_left else 1] = ind + M

        # make sure we line up with a flattened indexing scheme
        if ind < 0:
            assert -ind - 1 == tmin * total_xwidth * total_ywidth * total_zwidth + xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

        twidth = tmax - tmin
        xwidth = xmax - xmin
        ywidth = ymax - ymin
        zwidth = zmax - zmin
        if twidth == 1 and xwidth == 1 and ywidth == 1 and zwidth == 1:
            pass
        else:

            # by default our ranges remain unchanged
            ltmin = rtmin = tmin
            ltmax = rtmax = tmax
            lxmin = rxmin = xmin
            lxmax = rxmax = xmax
            lymin = rymin = ymin
            lymax = rymax = ymax
            lzmin = rzmin = zmin
            lzmax = rzmax = zmax

            # split the xaxis if it is the largest dimension

            ## The preference order changes the cluster
            ## For Video It was very difficult to understand if
            ## X,y should be done before t or
            ## The whole T before x,y as one would be mean preference towards
            ## Individual frame content and other would mean temporal undertanding
            ## Coming to a middle ground if T is pretty large lets say
            ## T>32 then it makes the below a good choice. not sure about 8
            # dimensions = {
            #     't': {'width': twidth, 'min': tmin},
            #     'x': {'width': xwidth, 'min': xmin},
            #     'y': {'width': ywidth, 'min': ymin},
            #     'z': {'width': zwidth, 'min': zmin}
            # }

            dimensions = (
                ('t', twidth, tmin),
                ('x', xwidth, xmin),
                ('y', ywidth, ymin),
                ('z', zwidth, zmin),
            )
            widths = (twidth, xwidth, ywidth, zwidth)
            max_width = max(widths[:-1])
            dim_select = 'z'
            z_mid = zmin + zwidth//2
            lmax = z_mid
            rmin = z_mid
            for dim in dimensions:
                if max_width==dim[1]:
                    if dim[1] > 1:
                        mid = dim[2] + dim[1]//2
                        lmax = mid
                        rmin = mid
                        dim_select = dim[0]
                    break
            # dimensions = sorted(dimensions[:-1], key=lambda x:x[1], reverse=True) + dimensions[-1]
            # dim = dimensions[0]
            # if dim[1] > 1:
            #     mid = dim[2] + dim[1]//2
            #     lmax = mid
            #     rmin = mid
            #     dim_select = dim[0]
            # else:
            #     dim = dimensions[-1]
            #     if dim[1] > 1:
            #         mid = dim[2] + dim[1]//2
            #         lmax = mid
            #         rmin = mid
            #         dim_select = dim[0]

            # for dim in sorted(list(dimensions.keys())[:-1], key=lambda d: dimensions[d]['width'], reverse=True) + ['z']:
            #     if dimensions[dim]['width'] > 1:
            #         mid = dimensions[dim]['min'] + dimensions[dim]['width'] // 2
            #         lmax = mid
            #         rmin = mid
            #         dim_select = dim
            #         # exec(f"l{dim}max = mid")
            #         # exec(f"r{dim}min = mid")
            #         break

            if dim_select == "t":
                ltmax = lmax
                rtmin = rmin
            elif dim_select == "x":
                lxmax = lmax
                rxmin = rmin
            elif dim_select == "y":
                lymax = lmax
                rymin = rmin
            else:
                lzmax = lmax
                rzmin = rmin


            # if twidth >= xwidth and twidth>=ywidth and twidth > 1:
            #     tmid = tmin + twidth // 2
            #     ltmax = tmid
            #     rtmin = tmid

            # if xwidth >= ywidth and xwidth > 1:
            #     xmid = xmin + xwidth // 2
            #     lxmax = xmid
            #     rxmin = xmid

            # # split the yaxis
            # elif ywidth > 1:
            #     ymid = ymin + ywidth // 2
            #     lymax = ymid
            #     rymin = ymid

            # # split the zaxis only when the other ranges are already width 1
            # else:
            #     zmid = zmin + zwidth // 2
            #     lzmax = zmid
            #     rzmin = zmid

            lsize = (ltmax - ltmin) * (lxmax - lxmin) * (lymax - lymin) * (lzmax - lzmin)
            rsize = (rtmax - rtmin) * (rxmax - rxmin) * (rymax - rymin) * (rzmax - rzmin)

            heapq.heappush(q, (-lsize, ltmin, ltmax, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
            heapq.heappush(q, (-rsize, rtmin, rtmax, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))
            # q.put((-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
            # q.put((-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))

        ind -= 1

    # fill in the group sizes
    for i in range(len(clustering)):
        li = int(clustering[i, 0])
        ri = int(clustering[i, 1])
        lsize = 1 if li < M else clustering[li-M, 3]
        rsize = 1 if ri < M else clustering[ri-M, 3]
        clustering[i, 3] = lsize + rsize
