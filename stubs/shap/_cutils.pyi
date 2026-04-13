import numpy as np
import numpy.typing as npt

def compute_grey_code_row_values(
    row_values: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    inds: npt.NDArray[np.uint64],
    outputs: npt.NDArray[np.float64],
    shapley_coeff: npt.NDArray[np.float64],
    extended_delta_indexes: npt.NDArray[np.uint64],
    noop_code: int,
) -> None: ...
def compute_exp_val(
    nsamples_run: int,
    nsamples_added: int,
    D: int,
    N: int,
    weights: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ey: npt.NDArray[np.float64],
) -> int: ...
