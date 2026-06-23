from shap.benchmark._compute import ComputeTime


class DummyExplanation:
    """Simple mock object to simulate an Explanation."""

    def __init__(self, compute_time, shape):
        self.compute_time = compute_time
        self.shape = shape


def test_compute_time_basic():
    """Test average compute time for multiple samples."""
    expl = DummyExplanation(compute_time=10, shape=(2,))
    ct = ComputeTime()

    result = ct(expl, "test")

    assert result.value == 5


def test_compute_time_single_sample():
    """Test compute time when only one sample is present."""
    expl = DummyExplanation(compute_time=10, shape=(1,))
    ct = ComputeTime()

    result = ct(expl, "test")

    assert result.value == 10


def test_compute_time_float():
    """Test compute time calculation with float values."""
    expl = DummyExplanation(compute_time=5.5, shape=(2,))
    ct = ComputeTime()

    result = ct(expl, "test")

    assert result.value == 2.75


def test_compute_time_name_propagation():
    """Ensure the benchmark name is correctly passed to the result."""
    expl = DummyExplanation(compute_time=10, shape=(2,))
    ct = ComputeTime()

    result = ct(expl, "my_test")

    assert result.name == "my_test"
