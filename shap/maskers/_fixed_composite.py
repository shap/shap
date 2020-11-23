import numpy as np
from ._masker import Masker

def createFixedCompositeClass(cls, *args, **kwargs):
    class FixedComposite(cls):
        def __init__(self, *args, **kwargs):
            super(FixedComposite, self).__init__(*args, **kwargs)

        def __call__(self, mask, *args):
            masked_X = super(FixedComposite, self).__call__(mask, *args)
            return masked_X

    return FixedComposite(*args, **kwargs)