try:
    # On MacOS, the newer libomp versions that comes with Homebrew (version >= 12)
    # cause segfaults to occur when pytorch + lightgbm are imported (in that order).
    # The error does not occur when we import lightgbm first because lightgbm
    # distributes its own libomp which takes precedence.
    # cf. GH #3092 for more context.
    import lightgbm  # noqa: F401
except ImportError:
    pass

import functools
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Subprocess isolation for macOS torch tests (GH #4075)
# ---------------------------------------------------------------------------
# On macOS, torch tests segfault due to an OpenMP/libomp conflict between
# PyTorch and LightGBM (upstream: pytorch/pytorch#121101).
# Instead of skipping these tests, we run them in a fresh subprocess so that
# the OpenMP state is clean and any crash is contained.

_SUBPROCESS_TIMEOUT_SECONDS = 300  # 5 minutes per test
_SUBPROCESS_ENV_VAR = "_SHAP_SUBPROCESS_CHILD"  # env var to prevent recursion


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "subprocess_isolation: run test in a subprocess on macOS to avoid torch/OpenMP segfaults (GH #4075)",
    )


def pytest_collection_modifyitems(config, items):
    """Replace Darwin skip markers (GH #4075) with subprocess isolation on macOS."""
    if platform.system() != "Darwin":
        return

    # When running inside a subprocess child, do NOT replace the skip markers.
    # The child should run the test in-process (with clean OpenMP state).
    # The original skipif(Darwin) condition is True, but the child inherits a
    # clean process, so we strip it there too — see the second check below.
    if os.environ.get(_SUBPROCESS_ENV_VAR):
        # In the child: remove the #4075 skipif markers so the test actually
        # runs in-process (the whole point of subprocess isolation).
        for item in items:
            item.own_markers = [
                m for m in item.own_markers if not (m.name == "skipif" and "#4075" in m.kwargs.get("reason", ""))
            ]
        return

    subprocess_marker = pytest.mark.subprocess_isolation

    for item in items:
        # Look through all markers for the skipif(Darwin...#4075) pattern
        new_markers = []
        replaced = False
        for marker in item.iter_markers():
            if marker.name == "skipif" and marker.kwargs.get("reason", "") and "#4075" in marker.kwargs["reason"]:
                # Replace skip with subprocess isolation
                replaced = True
                continue
            new_markers.append(marker)

        if replaced:
            # Clear existing markers and re-add the non-skip ones + subprocess marker
            item.own_markers = [
                m for m in item.own_markers if not (m.name == "skipif" and "#4075" in m.kwargs.get("reason", ""))
            ]
            item.add_marker(subprocess_marker)


@pytest.hookimpl(tryfirst=True, wrapper=True)
def pytest_runtest_call(item):
    """Run subprocess-isolated tests in a fresh process.

    Uses the wrapper hook pattern so that when a test is run in a subprocess,
    the in-process test body is never executed.
    """
    if not any(item.iter_markers(name="subprocess_isolation")):
        # Not isolated — run normally
        return (yield)

    node_id = item.nodeid
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        node_id,
        "--no-header",
        "-rN",
        "--tb=short",
        "--override-ini=addopts=",  # clear addopts to avoid mpl baseline issues
    ]

    # Set env var so the child conftest knows it's inside subprocess isolation
    child_env = {**os.environ, _SUBPROCESS_ENV_VAR: "1"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Subprocess-isolated test timed out after {_SUBPROCESS_TIMEOUT_SECONDS}s "
            f"(likely an infinite hang from the torch/OpenMP issue).\n"
            f"Test: {node_id}"
        )

    if result.returncode != 0:
        # On Unix, negative return codes indicate signal death
        sig_msg = ""
        if result.returncode < 0:
            try:
                sig_name = signal.Signals(-result.returncode).name
            except (ValueError, AttributeError):
                sig_name = f"signal {-result.returncode}"
            sig_msg = f" (killed by {sig_name})"

        pytest.fail(
            f"Subprocess-isolated test failed with exit code {result.returncode}{sig_msg}.\n"
            f"Test: {node_id}\n"
            f"--- stdout ---\n{result.stdout[-2000:] if result.stdout else '(empty)'}\n"
            f"--- stderr ---\n{result.stderr[-2000:] if result.stderr else '(empty)'}"
        )

    # Subprocess passed — do NOT yield, so the in-process test body is never executed.
    # Returning without yielding skips the wrapped hook (the actual test function).


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
            # breakpoint()
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
