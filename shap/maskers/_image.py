import queue
import numpy as np
from ..utils import assert_import, record_import_error
from ._masker import Masker
from .._serializable import Serializer, Deserializer

try:
    import cv2
except ImportError as e:
    record_import_error("cv2", "cv2 could not be imported!", e)


class Image(Masker):
    """ This masks out image regions with blurring or inpainting.
    """

    def __init__(self, mask_value, shape=None):
        """ Build a new Image masker with the given masking value.

        Parameters
        ----------
        mask_value : np.array, "blur(kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.

        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
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

        self.blur_kernel = None
        self._blur_value_cache = None
        if issubclass(type(mask_value), np.ndarray):
            self.mask_value = mask_value.flatten()
        elif isinstance(mask_value, str):
            assert_import("cv2")
            self.mask_value = mask_value
            if mask_value.startswith("blur("):
                self.blur_kernel = tuple(map(int, mask_value[5:-1].split(",")))
        else:
            self.mask_value = np.ones(self.input_shape).flatten() * mask_value
        self.build_partition_tree()

        # note if this masker can use different background for different samples
        self.fixed_background = not isinstance(self.mask_value, str)

        #self.scratch_mask = np.zeros(self.input_shape[:-1], dtype=bool)
        self.last_xid = None

    def __call__(self, mask, x):
        if np.prod(x.shape) != np.prod(self.input_shape):
            raise Exception("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker contructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.input_shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.flatten()

        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=bool)

        if isinstance(self.mask_value, str):
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    self._blur_value_cache = cv2.blur(x.reshape(self.input_shape), self.blur_kernel).flatten()
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
        """ Fill in the masked parts of the image through inpainting.
        """
        reshaped_mask = mask.reshape(self.input_shape).astype(np.uint8).max(2)
        if reshaped_mask.sum() == np.prod(self.input_shape[:-1]):
            out = x.reshape(self.input_shape).copy()
            out[:] = out.mean((0, 1))
            return out.flatten()

        return cv2.inpaint(
            x.reshape(self.input_shape).astype(np.uint8),
            reshaped_mask,
            inpaintRadius=3,
            flags=getattr(cv2, method)
        ).astype(x.dtype).flatten()

    def build_partition_tree(self):
        """ This partitions an image into a herarchical clustering based on axis-aligned splits.
        """

        xmin = 0
        xmax = self.input_shape[0]
        ymin = 0
        ymax = self.input_shape[1]
        zmin = 0
        zmax = self.input_shape[2]
        #total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        q = queue.PriorityQueue()
        M = int((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
        self.clustering = np.zeros((M - 1, 4))
        q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))
        ind = len(self.clustering) - 1
        while not q.empty():
            _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()

            if parent_ind >= 0:
                self.clustering[parent_ind, 0 if is_left else 1] = ind + M

            # make sure we line up with a flattened indexing scheme
            if ind < 0:
                assert -ind - 1 == xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

            xwidth = xmax - xmin
            ywidth = ymax - ymin
            zwidth = zmax - zmin
            if xwidth == 1 and ywidth == 1 and zwidth == 1:
                pass
            else:

                # by default our ranges remain unchanged
                lxmin = rxmin = xmin
                lxmax = rxmax = xmax
                lymin = rymin = ymin
                lymax = rymax = ymax
                lzmin = rzmin = zmin
                lzmax = rzmax = zmax

                # split the xaxis if it is the largest dimension
                if xwidth >= ywidth and xwidth > 1:
                    xmid = xmin + xwidth // 2
                    lxmax = xmid
                    rxmin = xmid

                # split the yaxis
                elif ywidth > 1:
                    ymid = ymin + ywidth // 2
                    lymax = ymid
                    rymin = ymid

                # split the zaxis only when the other ranges are already width 1
                else:
                    zmid = zmin + zwidth // 2
                    lzmax = zmid
                    rzmin = zmid

                lsize = (lxmax - lxmin) * (lymax - lymin) * (lzmax - lzmin)
                rsize = (rxmax - rxmin) * (rymax - rymin) * (rzmax - rzmin)

                q.put((-lsize, lxmin, lxmax, lymin, lymax, lzmin, lzmax, ind, True))
                q.put((-rsize, rxmin, rxmax, rymin, rymax, rzmin, rzmax, ind, False))

            ind -= 1

        # fill in the group sizes
        for i in range(len(self.clustering)):
            li = int(self.clustering[i, 0])
            ri = int(self.clustering[i, 1])
            lsize = 1 if li < M else self.clustering[li-M, 3]
            rsize = 1 if ri < M else self.clustering[ri-M, 3]
            self.clustering[i, 3] = lsize + rsize

    def save(self, out_file):
        """ Write a Image masker to a file stream.
        """
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.maskers.Image", version=0) as s:
            s.save("mask_value", self.input_mask_value)
            s.save("shape", self.input_shape)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a Image masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.Image", min_version=0, max_version=0) as s:
            kwargs["mask_value"] = s.load("mask_value")
            kwargs["shape"] = s.load("shape")
        return kwargs
