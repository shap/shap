import numpy as np
from ._masker import Masker

def createFixedCompositeMasker(cls, *args, **kwargs):
    class FixedComposite(cls):
        def __init__(self, *args, **kwargs):
            super(FixedComposite, self).__init__(*args, **kwargs)

        def __call__(self, mask, *args):
            masked_X = super(FixedComposite, self).__call__(mask, *args)
            wrapped_outputs = []
            for masked_x in masked_X:
                wrapped_outputs.append((masked_x,)+args)
            return np.array(wrapped_outputs)

    return FixedComposite(*args, **kwargs)