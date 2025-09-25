import numpy as np

def _exp_val(
    nsamples_run: int,
    nsamples_added: int,
    D: int,
    N: int,
    weights: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    ey: list[float] | np.ndarray,
) -> tuple[np.ndarray, int]: ...
