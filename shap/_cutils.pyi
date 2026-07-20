from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

@overload
def compute_grey_code_row_values(
    row_values: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu")],
    mask: Annotated[NDArray[numpy.bool_], dict(shape=(None,), device="cpu")],
    inds: Annotated[NDArray[numpy.uint64], dict(shape=(None,), device="cpu")],
    outputs: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu")],
    shapley_coeff: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu")],
    extended_delta_indexes: Annotated[NDArray[numpy.uint64], dict(shape=(None,), device="cpu")],
    noop_code: int,
) -> None: ...
@overload
def compute_grey_code_row_values(
    row_values: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device="cpu")],
    mask: Annotated[NDArray[numpy.bool_], dict(shape=(None,), device="cpu")],
    inds: Annotated[NDArray[numpy.uint64], dict(shape=(None,), device="cpu")],
    outputs: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device="cpu")],
    shapley_coeff: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu")],
    extended_delta_indexes: Annotated[NDArray[numpy.uint64], dict(shape=(None,), device="cpu")],
    noop_code: int,
) -> None: ...
def compute_exp_val(
    nsamples_run: int,
    nsamples_added: int,
    D: int,
    N: int,
    weights: Annotated[NDArray[numpy.float64], dict(shape=(None,), device="cpu")],
    y: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device="cpu")],
    ey: Annotated[NDArray[numpy.float64], dict(shape=(None, None), device="cpu")],
) -> int:
    """Compute the expected value for the kernel explainer algorithm"""
    ...
