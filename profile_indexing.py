import shap
import numpy as np
import time
import os
import psutil
import gc

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

print("=== SHAP 2D Baseline Benchmark (Master Branch) ===")

# 1. 2D Data (Master branch crashes on 3D ellipsis)
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

# 2. Execution Time
print("\n--- 1. Execution Time ---")
gc.collect() 
start_time = time.perf_counter()

iterations = 1000
for _ in range(iterations):
    # Standard 2D slice that master branch won't crash on
    _ = exp[:, "Feature_0"]
    
end_time = time.perf_counter()
print(f"Average time per slice: {(end_time - start_time) / iterations:.6f} seconds")

# 3. Memory Footprint
print("\n--- 2. Memory Footprint ---")
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
print("\n=== Benchmark Complete ===")