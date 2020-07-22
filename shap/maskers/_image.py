import numpy as np
import queue
from ..utils import assert_import, record_import_error
from ._masker import Masker
try:
    import cv2
except ImportError as e:
    record_import_error("cv2", "cv2 could not be imported!", e)


class Image(Masker):
    def __init__(self, mask_value, shape=None):
        """ This masks out image regions according to the given tokenizer. 
        
        Parameters
        ----------
        mask_value : np.array, "blur(kernel_xsize, kernel_xsize)", "inpaint_telea", or "inpaint_ns"
            The value used to mask hidden regions of the image.

        shape : None or tuple
            If the mask_value is an auto-generated masker instead of a dataset then the input
            image shape needs to be provided.
        """
        if shape is None:
            if type(mask_value) is str:
                raise TypeError("When the mask_value is a string the shape parameter must be given!")
            self.shape = mask_value.shape
        else:
            self.shape = shape
        
        self.blur_kernel = None
        if issubclass(type(mask_value), np.ndarray):
            self.mask_value = mask_value.flatten()
        elif type(mask_value) is str:
            assert_import("cv2")
            self.mask_value = mask_value
            if mask_value.startswith("blur("):
                self.blur_kernel = tuple(map(int, mask_value[5:-1].split(",")))
        else:
            self.mask_value = np.ones(self.shape).flatten() * mask_value
        self.build_partition_tree()

        # note if this masker can use different background for different samples
        self.variable_background = type(self.mask_value) is str
        
        self.scratch_mask = np.zeros(self.shape[:-1], dtype=np.bool)
        self.last_xid = None
    
    def __call__(self, x, mask=None):

        if np.prod(x.shape) != np.prod(self.shape):
            raise Exception("The length of the image to be masked must match the shape given in the " + \
                            "ImageMasker contructor: "+" * ".join([str(i) for i in x.shape])+ \
                            " != "+" * ".join([str(i) for i in self.shape]))

        # unwrap single element lists (which are how single input models look in multi-input format)
        if type(x) is list and len(x) == 1:
            x = x[0]
        
        # we preserve flattend inputs as flattened and full-shaped inputs as their original shape
        in_shape = x.shape
        if len(x.shape) > 1:
            x = x.flatten()
        
        # if mask is not given then we mask the whole image
        if mask is None:
            mask = np.zeros(np.prod(x.shape), dtype=np.bool)
            
        if type(self.mask_value) is str:
            if self.blur_kernel is not None:
                if self.last_xid != id(x):
                    self.blur_value = cv2.blur(x.reshape(self.shape), self.blur_kernel).flatten()
                    self.last_xid = id(x)
                out = x.copy()
                out[~mask] = self.blur_value[~mask]
                
            elif self.mask_value == "inpaint_telea":
                out = self.inpaint(x, ~mask, "INPAINT_TELEA")
            elif self.mask_value == "inpaint_ns":
                out = self.inpaint(x, ~mask, "INPAINT_NS")
        else:
            out = x.copy()
            out[~mask] = self.mask_value[~mask]

        return out.reshape(1, *in_shape)
        
    def blur(self, x, mask):
        cv2.blur()
        
    def inpaint(self, x, mask, method):
        reshaped_mask = mask.reshape(self.shape).astype(np.uint8).max(2)
        if reshaped_mask.sum() == np.prod(self.shape[:-1]):
            out = x.reshape(self.shape).copy()
            out[:] = out.mean((0,1))
            return out.flatten()
        else:
            return cv2.inpaint(
                x.reshape(self.shape).astype(np.uint8),
                reshaped_mask,
                inpaintRadius=3,
                flags=getattr(cv2, method)
            ).astype(x.dtype).flatten()

    def build_partition_tree(self):
        """ This partitions an image into a herarchical clustering based on axis-aligned splits.
        """
        
        xmin = 0
        xmax = self.shape[0]
        ymin = 0
        ymax = self.shape[1]
        zmin = 0
        zmax = self.shape[2]
        #total_xwidth = xmax - xmin
        total_ywidth = ymax - ymin
        total_zwidth = zmax - zmin
        q = queue.PriorityQueue()
        M = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        self.partition_tree = np.zeros((M - 1, 2))
        q.put((0, xmin, xmax, ymin, ymax, zmin, zmax, -1, False))
        ind = len(self.partition_tree) - 1
        while not q.empty():
            _, xmin, xmax, ymin, ymax, zmin, zmax, parent_ind, is_left = q.get()
            
            if parent_ind >= 0:
                self.partition_tree[parent_ind, 0 if is_left else 1] = ind

            # make sure we line up with a flattened indexing scheme
            if ind < 0:
                assert -ind - 1 ==  xmin * total_ywidth * total_zwidth + ymin * total_zwidth + zmin

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
        self.partition_tree += int(M)