"""Run CPU/GPU benchmarks on complete synthetic trees with process timeouts.

Run from the repository root with the same environment as the original script:
    python gputree_benchmarks/benchmark_gputreeexplainer_timeout.py

Edit TIMEOUT_SECONDS and the workload grid in run() to change the workloads. Timings include
explainer construction and exp(X), but exclude process startup and data creation.
After a backend times out, subsequent workloads with more samples at the same
feature count and model configuration are recorded as timeouts without starting a worker.
"""

import logging
import multiprocessing as mp
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import wait
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 1_200
SETUP_TIMEOUT_SECONDS = 180
benchmark_dict = {}
run_id = 0


def explain(model, explainer_class, data):
    exp = explainer_class(model, feature_perturbation="tree_path_dependent")
    return exp(data)


def explanation_worker(connection, model, class_name, n_samples, n_features):
    try:
        import numpy as np
        import shap

        # Identical nested batches for CPU/GPU, without sending large arrays
        # through multiprocessing or retaining them in the parent process.
        X = np.random.default_rng(1).standard_normal((n_samples, n_features))
        explainer_class = getattr(shap, class_name)
        connection.send({"status": "ready"})
        start = perf_counter()
        result = explain(model, explainer_class, X)
        elapsed = perf_counter() - start
        connection.send({
            "status": "success",
            "time": elapsed,
            # Only return a small subset for correctness comparison.
            "values": result.values[:100].tolist(),
        })
    except Exception as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def receive_message(connection, process, timeout):
    ready = wait([connection, process.sentinel], timeout=timeout)
    if connection in ready or connection.poll():
        try:
            return connection.recv()
        except EOFError:
            pass
    if process.sentinel in ready:
        process.join()
        return {"status": "error", "error": f"Worker exited with code {process.exitcode}"}
    return None


def run_worker(target, args, timeout_seconds=TIMEOUT_SECONDS, setup_timeout_seconds=SETUP_TIMEOUT_SECONDS):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        message = receive_message(receiver, process, setup_timeout_seconds)
        if message is None:
            return {"status": "timeout", "phase": "setup", "timeout_seconds": setup_timeout_seconds}
        if message["status"] != "ready":
            return message
        message = receive_message(receiver, process, timeout_seconds)
        if message is None:
            return {"status": "timeout", "phase": "explanation", "timeout_seconds": timeout_seconds}
        return message
    finally:
        if started:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        sender.close()
        receiver.close()


def build_complete_forest(max_depth, n_trees, n_features, seed=0):
    """Construct balanced trees whose leaves are all exactly max_depth deep.

    Each level uses a distinct feature along a path, with threshold zero.
    Unit leaf cover gives equal branch probability and consistent node cover.
    Independent random leaf values are scaled to represent an ensemble average.
    Trees share the same topology but rotate the feature assignment by tree.
    """
    import numpy as np

    if not 1 <= max_depth <= n_features or n_trees < 1:
        raise ValueError("Require 1 <= max_depth <= n_features and n_trees >= 1")
    n_leaves = 2**max_depth
    n_nodes = 2 * n_leaves - 1
    n_internal = n_leaves - 1
    indices = np.arange(n_nodes)
    depths = np.floor(np.log2(indices + 1)).astype(np.int32)
    children_left = np.full(n_nodes, -1, dtype=np.int32)
    children_right = children_left.copy()
    children_left[:n_internal] = 2 * indices[:n_internal] + 1
    children_right[:n_internal] = 2 * indices[:n_internal] + 2
    weights = np.exp2(max_depth - depths)
    rng = np.random.default_rng(seed)
    trees = []
    for tree_id in range(n_trees):
        features = np.full(n_nodes, -1, dtype=np.int32)
        features[:n_internal] = (depths[:n_internal] + tree_id) % n_features
        values = np.zeros((n_nodes, 1), dtype=np.float64)
        values[n_internal:, 0] = rng.standard_normal(n_leaves) / n_trees
        trees.append({
            "children_left": children_left,
            "children_right": children_right,
            "children_default": children_left,
            "features": features,
            "thresholds": np.zeros(n_nodes, dtype=np.float64),
            "values": values,
            "node_sample_weight": weights,
        })
    return {
        "trees": trees,
        "tree_output": "raw_value",
        "base_offset": 0.0,
        "benchmark": {
            "max_depth": max_depth,
            "n_trees": n_trees,
            "min_tree_depth": max_depth,
            "max_tree_depth": max_depth,
            "mean_tree_depth": float(max_depth),
            "mean_leaves_per_tree": float(n_leaves),
            "total_nodes": n_nodes * n_trees,
        },
    }


def benchmark_function(model, class_name, n_samples, n_features):
    model_details = model["benchmark"]
    global run_id
    run_id += 1
    logger.info("Starting run %d: %s with data shape (%d, %d)", run_id, class_name, n_samples, n_features)
    for previous_run_id, previous in benchmark_dict.items():
        if (
            previous["class"] == class_name
            and previous["status"] == "timeout"
            and previous["phase"] == "explanation"
            and all(previous.get(key) == value for key, value in model_details.items())
            and n_samples > previous["data_shape"][0]
            and n_features == previous["data_shape"][1]
        ):
            result = {
                "status": "timeout",
                "phase": previous["phase"],
                "timeout_seconds": previous["timeout_seconds"],
                "skipped": True,
                "timeout_run_id": previous.get("timeout_run_id", previous_run_id),
            }
            break
    else:
        result = run_worker(explanation_worker, (model, class_name, n_samples, n_features))
    values = result.pop("values", None)
    result.update({"class": class_name, "data_shape": [n_samples, n_features], **model_details})
    benchmark_dict[run_id] = result
    if result["status"] == "success":
        logger.info("Run %d: %s took %.3f seconds", run_id, class_name, result["time"])
    elif result.get("skipped"):
        logger.warning(
            "Run %d: skipping %s with data shape (%d, %d) because run %d timed out",
            run_id, class_name, n_samples, n_features, result["timeout_run_id"],
        )
    elif result["status"] == "timeout":
        logger.warning("Run %d: %s timed out during %s after %s seconds", run_id, class_name, result["phase"], result["timeout_seconds"])
    else:
        logger.error("Run %d: %s failed: %s", run_id, class_name, result["error"])
    return values


def compare_by_size(model, n_samples, n_features):
    import numpy as np

    cpu_values = benchmark_function(model, "TreeExplainer", n_samples, n_features)
    gpu_values = benchmark_function(model, "GPUTreeExplainer", n_samples, n_features)
    if cpu_values is not None and gpu_values is not None:
        try:
            np.testing.assert_allclose(cpu_values, gpu_values, rtol=1e-4, atol=1e-4)
        except AssertionError as exc:
            logger.error("CPU/GPU correctness comparison failed: %s", exc)
            for result_id in (run_id - 1, run_id):
                benchmark_dict[result_id]["validation"] = "failed"
        else:
            for result_id in (run_id - 1, run_id):
                benchmark_dict[result_id]["validation"] = "passed"
    else:
        logger.info("Skipping this batch's correctness comparison: both backends must finish")


def markdown_cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    ).replace("|", "&#124;").replace("\r", " ").replace("\n", "<br>")


def hardware_abbreviation(name):
    name = re.sub(r"\((?:R|TM)\)|[®™]", "", str(name), flags=re.IGNORECASE)
    name = name.split("@", 1)[0]
    name = re.sub(
        r"\b(?:Intel|AMD|NVIDIA|GeForce|CPU|Processor)\b|\b\d+-Core\b",
        "", name, flags=re.IGNORECASE,
    )
    return " ".join(name.split()) or "unknown"


def save_results(cpu, gpu, metadata=None):
    metadata = {
        "CPU": cpu,
        "GPU": gpu,
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "Explanation timeout (seconds)": TIMEOUT_SECONDS,
        "Setup timeout (seconds)": SETUP_TIMEOUT_SECONDS,
        "Timing scope": "Explainer construction and exp(X); excludes process startup and data creation",
        "Skip rule": "More samples at the same feature count and model configuration, for the same explainer type, after an explanation timeout",
        **(metadata or {}),
    }
    gpu_names = [gpu] if isinstance(gpu, str) else gpu
    gpu_label = ", ".join(hardware_abbreviation(name) for name in gpu_names) or "unknown"
    title = (
        f"# Tree explainer benchmark — CPU: {markdown_cell(hardware_abbreviation(cpu))}"
        f" | GPU: {markdown_cell(gpu_label)}"
    )
    lines = [title, "", "## Metadata", ""]
    lines.extend(f"- **{key}:** {markdown_cell(value)}" for key, value in metadata.items())
    lines.extend([
        "", "## Results", "",
        "| run_id | explainer_type | n_samples | n_features | max_depth | n_trees | min–max tree depth | mean tree depth | mean leaves/tree | total_nodes | time (s) | status | validation | details |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for result_id, result in benchmark_dict.items():
        details = ""
        if result.get("skipped"):
            details = f"Skipped because run {result['timeout_run_id']} timed out"
        elif result["status"] == "timeout":
            details = f"{result['phase']} exceeded {result['timeout_seconds']} s"
        elif result["status"] == "error":
            details = result["error"]
        cells = [
            result_id, result["class"], *result["data_shape"],
            result["max_depth"], result["n_trees"],
            f"{result['min_tree_depth']}–{result['max_tree_depth']}",
            f"{result['mean_tree_depth']:.2f}", f"{result['mean_leaves_per_tree']:.2f}",
            result["total_nodes"],
            f"{result['time']:.6f}" if result["status"] == "success" else "—",
            result["status"], result.get("validation", "—"), details,
        ]
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    output_path = Path(__file__).with_name("benchmarks_timeout.md")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved benchmark results to %s", output_path)


def run():
    from benchmark_gputreeexplainer import hardware_names

    # Hold the data shape fixed to isolate tree depth and ensemble size.
    n_features = 50
    n_samples = 20_000
    max_depths = [12, 20, 28, 36]
    tree_counts = [10, 100, 500]
    cpu, gpu = hardware_names()
    metadata = {
        "Started (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Model": "Synthetic complete balanced binary forest; all leaves at the requested depth",
        "Depth convention": "Root depth is 0; max_depth is exact for every tree and every leaf",
        "Tree depth statistics": "Min, max and mean of each tree's maximum path depth across the forest",
        "Training data": "None; trees are constructed directly using SHAP custom model dictionaries",
        "Tree construction": "Thresholds=0; distinct features per path, rotated by tree; unit leaf cover; normal leaf values, seed=0, averaged across trees",
        "Interpretation": "Synthetic scaling benchmark; leaves per tree = 2**depth, so depth also changes model size",
        "Explanation data": "Standard normal, numpy default_rng seed=1",
        "Feature perturbation": "tree_path_dependent",
        "Validation": "First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4",
        "Exact tree depths": max_depths,
        "Tree counts": tree_counts,
        "Explanation shape (samples, features)": [n_samples, n_features],
        "Initial comparison samples": 100,
    }
    for package in ("shap", "numpy", "scikit-learn"):
        try:
            metadata[f"{package} version"] = version(package)
        except PackageNotFoundError:
            metadata[f"{package} version"] = "unavailable"
    for max_depth in max_depths:
        for n_trees in tree_counts:
            logger.info("Building complete forest with depth=%d and n_trees=%d", max_depth, n_trees)
            model = build_complete_forest(max_depth, n_trees, n_features)
            # Keep a small correctness comparison even if the full batch times out.
            compare_by_size(model, 100, n_features)
            save_results(cpu, gpu, metadata)
            compare_by_size(model, n_samples, n_features)
            save_results(cpu, gpu, metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
