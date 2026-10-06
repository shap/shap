import json
import logging
import platform
import subprocess
from pathlib import Path
from time import time
import numpy as np
from sklearn.datasets import make_regression
from shap import TreeExplainer
from shap import GPUTreeExplainer
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

benchmark_dict = {}
run_id = 0

def benchmark_function(func):
    def inner(model, explainer_class, data):
        now = time()
        res = func(model, explainer_class, data)
        after = time()
        elapsed = after - now
        benchmark_dict[run_id] = {"time": elapsed, "class": explainer_class.__name__, "data_shape": data.shape}
        logger.info(
            "Run %d: %s with data shape %s took %.3f seconds",
            run_id, explainer_class.__name__, data.shape, elapsed,
        )
        return res
    return inner

@benchmark_function
def explain(model, explainer_class, data):
    exp = explainer_class(model)
    global run_id
    run_id += 1
    return exp(data)

def compare_by_size(model, X):
    shap_values = explain(model=model, explainer_class=TreeExplainer, data=X)
    shap_values_gpu = explain(model=model, explainer_class=GPUTreeExplainer, data=X)
    np.testing.assert_allclose(shap_values.values, shap_values_gpu.values, rtol=1e-4, atol=1e-4)


def hardware_names():
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        gpu = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        logger.warning("Could not determine GPU names using nvidia-smi")
        gpu = []
    return cpu, gpu


def run():
    # Feature count -> explanation batch sizes.
    # Add 10_000, 100_000, or 1_000_000 to each list for larger batches.
    sizes = {
        10: [1_000, 10_000, 100_000, 1_000_000],
        50: [1_000, 10_000, 100_000, 1_000_000],
        100: [1_000, 10_000, 100_000, 1_000_000],
    }

    for n_features, sample_sizes in sizes.items():
        # Keep training size fixed as explanation batches grow.
        logger.info("Training model with 800 samples and %d features", n_features)
        X_train, y_train = make_regression(
            n_samples=800, n_features=n_features, n_targets=1, noise=0.0, random_state=0
        )
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train[:1000, :], y_train[:1000])

        # Reuse nested batches from one independent explanation dataset.
        X_explain = np.random.default_rng(1).standard_normal((max(sample_sizes), n_features))
        for n_samples in sample_sizes:
            logger.info("Run with samples: %d and n_features: %d", n_samples, n_features)
            compare_by_size(model, X_explain[:n_samples])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
    cpu, gpu = hardware_names()
    results = {"benchmarks": benchmark_dict, "CPU": cpu, "GPU": gpu}
    output_path = Path(__file__).with_name("benchmarks.json")
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Saved benchmark results to %s", output_path)
