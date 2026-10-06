# Tree explainer benchmark — CPU: Ryzen 5 3600 | GPU: RTX 2080 Ti

## Metadata

- **CPU:** AMD Ryzen 5 3600 6-Core Processor
- **GPU:** ['NVIDIA GeForce RTX 2080 Ti']
- **OS:** Linux-7.0.0-31-generic-x86_64-with-glibc2.39
- **Python:** 3.13.14
- **Explanation timeout (seconds):** 1200
- **Setup timeout (seconds):** 180
- **Timing scope:** Explainer construction and exp(X); excludes process startup and data creation
- **Skip rule:** More samples at the same feature count and model configuration, for the same explainer type, after an explanation timeout
- **Started (UTC):** 2026-10-06T21:18:24+00:00
- **Model:** RandomForestRegressor(max_depth=grid, n_estimators=grid, random_state=0); other parameters use sklearn defaults
- **Depth convention:** Root depth is 0; max_depth is a configured limit, actual depths are measured after fitting
- **Tree depth statistics:** Min, max and mean of each tree's maximum path depth across the forest
- **Training data:** One shared make_regression invocation: 20000 samples, 50 features, 10 informative features, 1 target, noise=0.0, random_state=0
- **Explanation data:** Standard normal, numpy default_rng seed=1
- **Feature perturbation:** tree_path_dependent
- **Validation:** First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4
- **Max depth limits:** [12, 20, 28, 36]
- **Tree counts:** [10, 100, 500]
- **Explanation shape (samples, features):** [1000, 50]
- **Initial comparison samples:** 100
- **shap version:** 0.53.0rc1.dev45
- **numpy version:** 2.3.5
- **scikit-learn version:** 1.8.0

## Results

| run_id | explainer_type | n_samples | n_features | max_depth | n_trees | min–max tree depth | mean tree depth | mean leaves/tree | total_nodes | time (s) | status | validation | details |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | TreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 1.059360 | success | passed |  |
| 2 | GPUTreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.355230 | success | passed |  |
| 3 | TreeExplainer | 1000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 10.583430 | success | passed |  |
| 4 | GPUTreeExplainer | 1000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.377726 | success | passed |  |
| 5 | TreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 10.824266 | success | passed |  |
| 6 | GPUTreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1.254994 | success | passed |  |
