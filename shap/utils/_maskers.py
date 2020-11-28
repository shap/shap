import numpy as np

def variants(masker, *args):
    variants, variants_column_sums, variants_row_inds = None, None, None
    # if the masker supports it, save what positions vary from the background
    if callable(getattr(masker, "invariants", None)):
        variants = ~masker.invariants(*args)
        variants_column_sums = variants.sum(0)
        variants_row_inds = [
            variants[:,i] for i in range(variants.shape[1])
        ]
    else:
        variants = None

    return variants, variants_column_sums, variants_row_inds

def shape(masker, *args):
    masker_rows, masker_cols = None, None
    # compute the length of the mask (and hence our length)
    if hasattr(masker, "shape"):
        if callable(masker.shape):
            mshape = masker.shape(*args)
            masker_rows = mshape[0]
            masker_cols = mshape[1]
        else:
            mshape = masker.shape
            masker_rows = mshape[0]
            masker_cols = mshape[1]
    else:
        masker_rows = None# # just assuming...
        masker_cols = sum(np.prod(a.shape) for a in args)

    return masker_rows, masker_cols