import functools
import sys
from pathlib import Path

# Remove the current working directory from sys.path to ensure tests import the
# installed shap package (with compiled C extensions) rather than the source tree.
# If this line is commented out, run pytest via `python -P -m pytest tests` instead.
sys.path[:] = [p for p in sys.path if p not in ("", ".")]

try:
    # On MacOS, the newer libomp versions that comes with Homebrew (version >= 12)
    # cause segfaults to occur when pytorch + lightgbm are imported (in that order).
    # The error does not occur when we import lightgbm first because lightgbm
    # distributes its own libomp which takes precedence.
    # cf. GH #3092 for more context.
    import lightgbm  # noqa: F401
except ImportError:
    pass

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--random-seed", action="store", help="Fix the random seed")


@pytest.fixture()
def random_seed(request) -> int:
    """Provides a test-specific random seed for reproducible "fuzz testing".

    Example use in a test:

        def test_thing(random_seed):

            # Numpy
            rs = np.random.RandomState(seed=random_seed)
            values = rs.randint(...)

            # Pytorch
            torch.manual_seed(random_seed)

            # Tensorflow
            tf.compat.v1.random.set_random_seed(random_seed)

    By default, a new seed is generated on each run of the tests. If a test
    fails, the random seed used will be displayed in the pytest logs.

    The seed can be fixed by providing a CLI option e.g:

        pytest --random-seed 123

    For numpy usage, note the legacy `RandomState` has stricter version-to-version
    compatibility guarantees than new-style `default_rng`:
    https://numpy.org/doc/stable/reference/random/compatibility.html

    """
    manual_seed = request.config.getoption("--random-seed")
    if manual_seed is not None:
        return int(manual_seed)
    else:
        # Otherwise, create a new seed for each test
        rs = np.random.RandomState()
        return rs.randint(0, 1000)


@pytest.fixture(autouse=True)
def global_random_seed():
    """Set the global numpy random seed before each test

    Nb. Tests that use random numbers should instantiate a local
    `np.random.RandomState` rather than use the global numpy random state.
    """
    np.random.seed(0)


@pytest.fixture(autouse=True)
def mpl_test_cleanup():
    """Run tests in a mpl context manager and close figures after each test."""
    plt.switch_backend("Agg")  # Non-interactive backend
    with plt.rc_context():
        yield
    plt.close("all")


def compare_numpy_outputs_against_baseline(*, func_file, baseline_dir=None, rtol=1e-4, atol=1e-6):
    if baseline_dir is None:
        baseline_dir = Path(__file__).parent / "shap_values_baselines"
    elif isinstance(baseline_dir, str):
        baseline_dir = Path(baseline_dir)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            base_func_name = f"{Path(func_file).stem}_{func.__name__}"
            baseline_file = baseline_dir / f"{base_func_name}_baseline.npz"
            if hasattr(output, "values"):
                arrays = {"values": output.values, "base_values": np.asarray(output.base_values)}
            else:
                arrays = {"values": output}
            if baseline_file.exists():
                baseline = np.load(baseline_file, allow_pickle=False)
                for key in arrays:
                    np.testing.assert_allclose(arrays[key], baseline[key], rtol=rtol, atol=atol)
            else:
                baseline_dir.mkdir(parents=True, exist_ok=True)
                np.savez(baseline_file, **arrays)
            return output

        return wrapper

    return decorator
