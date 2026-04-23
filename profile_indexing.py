import shap
import numpy as np
import time
import os
import psutil
import gc
import traceback

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_3d_benchmark():
    print("\n=== 1. 3D Ellipsis Benchmark (Crash Test) ===")
    N, D, C = 10000, 50, 2 
    values = np.random.randn(N, D, C)
    data = np.random.randn(N, D)
    feature_names = [f"Feature_{i}" for i in range(D)]
    output_names = [["Class_0", "Class_1"]] * N
    
    exp = shap.Explanation(
        values=values,
        data=data,
        feature_names=feature_names,
        output_names=output_names
    )
    
    print(f"Base Explanation shape: {exp.shape}")
    print("Attempting 3D slice: exp[..., 'Class_0']")
    
    start_time = time.perf_counter()
    try:
        subset = exp[..., "Class_0"]
        end_time = time.perf_counter()
        print(f"✅ SUCCESS: Sliced in {(end_time - start_time):.6f} seconds")
        print(f"Subset Shape: {subset.shape} (Expected: ({N}, {D}))")
    except Exception as e:
        print(f"❌ CRASH DETECTED (Expected on old master branch)")
        print(f"Error: {type(e).__name__}: {str(e).splitlines()[0]}")

def run_2d_benchmark():
    print("\n=== 2. 2D Memory Benchmark ===")
    N, D = 100000, 1000 
    values = np.random.randn(N, D)
    data = np.random.randn(N, D)
    feature_names = [f"Feature_{i}" for i in range(D)]

    exp = shap.Explanation(
        values=values,
        data=data,
        feature_names=feature_names
    )

    print(f"Base Explanation shape: {exp.shape}")

    # Execution Time
    gc.collect() 
    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        _ = exp[:, "Feature_0"]
    end_time = time.perf_counter()
    print(f"Average time per 2D slice: {(end_time - start_time) / iterations:.6f} seconds")

    # Memory Footprint
    gc.collect()
    mem_before = get_memory_mb()

    hold_memory = []
    for _ in range(100):
        hold_memory.append(exp[:, "Feature_0"])
        
    mem_after = get_memory_mb()
    subset = hold_memory[0]

    # Check view status safely
    is_view = getattr(subset.values, "base", None) is not None

    print(f"Are internal arrays NumPy Views?: {is_view}")
    print(f"Memory Spike (100 slices): {mem_after - mem_before:.2f} MB")

if __name__ == "__main__":
    print("Starting SHAP Explanation Indexing Benchmarks...")
    run_3d_benchmark()
    run_2d_benchmark()
    print("\n=== Benchmarks Complete ===")