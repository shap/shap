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
