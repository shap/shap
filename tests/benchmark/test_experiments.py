import pickle
import subprocess
from unittest.mock import patch

import pytest

import shap
from shap.benchmark import experiments


def test_experiments_generator():
    """Test the experiment filtering logic."""
    # Test fetching all experiments
    all_exps = list(experiments.experiments())
    assert len(all_exps) > 0

    # Test filtering by all parameters
    filtered = list(
        experiments.experiments(dataset="corrgroups60", model="lasso", method="random", metric="local_accuracy")
    )
    assert len(filtered) == 1
    assert filtered[0] == ["corrgroups60", "lasso", "random", "local_accuracy"]


def test_gen_cache_id():
    """Ensure the cache ID string is formatted correctly."""
    exp = ["datasetX", "modelY", "methodZ", "metricW"]
    cid = experiments.__gen_cache_id(exp)
    assert cid == f"v{shap.__version__}__datasetX__modelY__methodZ__metricW"


@patch("shap.benchmark.experiments.datasets")
@patch("shap.benchmark.experiments.metrics")
@patch("shap.benchmark.experiments.models")
def test_run_experiment(mock_models, mock_metrics, mock_datasets, tmp_path):
    """Test running an experiment locally (both cache miss and cache hit)."""
    exp = ["dummy_data", "dummy_model", "dummy_method", "dummy_metric"]

    # 1. Setup mock returns for the dynamically loaded modules
    mock_datasets.dummy_data.return_value = ("X_mock", "y_mock")
    mock_models.dummy_data__dummy_model = "MockedModelObj"
    mock_metrics.dummy_metric.return_value = 0.95

    # 2. Run without cache (simulates full computation)
    score = experiments.run_experiment(exp, cache_dir=str(tmp_path))
    assert score == 0.95

    # Verify the underlying ML loading functions were called
    mock_datasets.dummy_data.assert_called_once()
    mock_metrics.dummy_metric.assert_called_once_with("X_mock", "y_mock", "MockedModelObj", "dummy_method")

    # 3. Run again, should hit the cache and NOT call the mocked datasets/metrics
    mock_datasets.dummy_data.reset_mock()
    score_cached = experiments.run_experiment(exp, cache_dir=str(tmp_path))
    assert score_cached == 0.95
    mock_datasets.dummy_data.assert_not_called()


@patch("shap.benchmark.experiments.run_experiment")
def test_run_experiments(mock_run_exp, tmp_path):
    """Test the local mapping function (single worker vs multi-worker pool)."""
    mock_run_exp.return_value = 42.0

    # Single worker (map)
    out_single = experiments.run_experiments(
        dataset="cric",
        model="lasso",
        method="random",
        metric="local_accuracy",
        nworkers=1,
        cache_dir=str(tmp_path),
    )
    assert len(out_single) == 1
    assert out_single[0][1] == 42.0

    # Multi-worker (Pool)
    out_multi = experiments.run_experiments(
        dataset="cric",
        model="lasso",
        method="random",
        metric="local_accuracy",
        nworkers=2,
        cache_dir=str(tmp_path),
    )
    assert len(out_multi) == 1
    assert out_multi[0][1] == 42.0


# We need a custom fake clock to test the Thread Worker rate limit without creating an infinite loop
_current_time = [0.0]


def fake_time():
    return _current_time[0]


def fake_sleep(secs):
    _current_time[0] += secs


@patch("time.time", side_effect=fake_time)
@patch("time.sleep", side_effect=fake_sleep)
@patch("subprocess.run")
@patch("subprocess.check_output")
def test_run_remote_experiments(mock_check_output, mock_run, mock_sleep, mock_time, tmp_path):
    """
    Test the complex multi-threaded remote SSH queue execution.
    We mock time.sleep and time.time to simulate throttling without actually waiting.
    """
    exp1 = ["cric", "lasso", "random", "local_accuracy"]
    exp2 = ["cric", "ridge", "random", "local_accuracy"]

    # We need to simulate the SCP command successfully creating the cache file locally
    def check_output_side_effect(args, **kwargs):
        if args[0] == "scp":
            dest_file = args[2]
            with open(dest_file, "wb") as f:
                pickle.dump(0.99, f)
        return b"success"

    mock_check_output.side_effect = check_output_side_effect

    # Run the remote batch. Using rate_limit=1 forces the thread worker
    # to hit the throttling logic ("sleep 5 seconds until 61 seconds have passed")
    # since we pass two experiments to the exact same host.
    experiments.run_remote_experiments([exp1, exp2], ["localhost:python"], rate_limit=1)

    # Verify our system calls fired
    assert mock_run.called  # the SSH pkill command fired
    assert mock_check_output.called  # the SSH and SCP routines fired


@patch("subprocess.check_output")
@patch("time.sleep", return_value=None)
def test_run_remote_experiment_exceptions(mock_sleep, mock_check_output, tmp_path):
    """Test gracefully handling SSH/SCP network failures."""
    exp = ["dummy_data", "dummy_model", "dummy_method", "dummy_metric"]
    initial_failures = experiments.total_failed

    # 1. Test subprocess failure (e.g., SSH disconnects)
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
    experiments.__run_remote_experiment(exp, "localhost", cache_dir=str(tmp_path))
    assert experiments.total_failed == initial_failures + 1

    # 2. Test FileNotFoundError (SCP executes but file doesn't actually exist locally)
    mock_check_output.side_effect = None
    with pytest.raises(FileNotFoundError):
        experiments.__run_remote_experiment(exp, "localhost", cache_dir=str(tmp_path))
