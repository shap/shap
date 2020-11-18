from ._masker import Masker
import numpy as np


class FixedComposite(Masker):
    def __init__(self, masker):
        self.masker = masker
    
    def __call__(self, mask, x):
        masked_x = self.masker(mask, x)
        if isinstance(x,str):
            x = np.array([x])
        else:
            x = x.reshape(1, *x.shape)

        out = np.append(masked_x, x, axis=0)
        out = np.expand_dims(out, axis=0)
        
        return out