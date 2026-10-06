# Tree explainer benchmark — CPU: 13th Gen Core i7-13850HX | GPU: RTX 2000 Ada Generation Laptop GPU

## Metadata

- **CPU:** 13th Gen Intel(R) Core(TM) i7-13850HX
- **GPU:** ['NVIDIA RTX 2000 Ada Generation Laptop GPU']
- **OS:** Linux-7.0.0-34-generic-x86_64-with-glibc2.43
- **Python:** 3.13.14
- **Explanation timeout (seconds):** 1200
- **Setup timeout (seconds):** 60
- **Timing scope:** Explainer construction and exp(X); excludes process startup and data creation
- **Skip rule:** More samples at the same feature count, for the same explainer type
- **Started (UTC):** 2026-10-06T19:48:54+00:00
- **Model:** RandomForestRegressor(n_estimators=100, random_state=0); other parameters use sklearn defaults
- **Training data:** make_regression: 800 samples, 1 target, noise=0.0, random_state=0
- **Explanation data:** Standard normal, numpy default_rng seed=1
- **Feature perturbation:** tree_path_dependent
- **Validation:** First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4
- **Workloads (features → samples):** {10: [1000, 10000, 100000, 1000000], 50: [1000, 10000, 100000, 1000000], 100: [1000, 10000, 100000, 1000000]}
- **Initial comparison samples:** 100
- **shap version:** 0.53.0rc1.dev44
- **numpy version:** 2.3.5
- **scikit-learn version:** 1.8.0

## Results

| run_id | explainer_type | n_samples | n_features | time (s) | status | validation | details |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | TreeExplainer | 100 | 10 | 0.922441 | success | passed |  |
| 2 | GPUTreeExplainer | 100 | 10 | 0.267484 | success | passed |  |
| 3 | TreeExplainer | 1000 | 10 | 8.839536 | success | passed |  |
| 4 | GPUTreeExplainer | 1000 | 10 | 0.436340 | success | passed |  |
