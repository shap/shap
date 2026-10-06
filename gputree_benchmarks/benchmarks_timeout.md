# Tree explainer benchmark — CPU: Ryzen 5 3600 | GPU: RTX 2080 Ti

## Metadata

- **CPU:** AMD Ryzen 5 3600 6-Core Processor
- **GPU:** ['NVIDIA GeForce RTX 2080 Ti']
- **OS:** Linux-7.0.0-31-generic-x86_64-with-glibc2.39
- **Python:** 3.13.14
- **Explanation timeout (seconds):** 1200
- **Setup timeout (seconds):** 60
- **Timing scope:** Explainer construction and exp(X); excludes process startup and data creation
- **Skip rule:** More samples at the same feature count, for the same explainer type
- **Started (UTC):** 2026-10-06T20:21:15+00:00
- **Model:** RandomForestRegressor(n_estimators=100, random_state=0); other parameters use sklearn defaults
- **Training data:** make_regression: 800 samples, 1 target, noise=0.0, random_state=0
- **Explanation data:** Standard normal, numpy default_rng seed=1
- **Feature perturbation:** tree_path_dependent
- **Validation:** First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4
- **Workloads (features → samples):** {10: [1000, 10000, 100000, 1000000], 50: [1000, 10000, 100000, 1000000], 100: [1000, 10000, 100000, 1000000]}
- **Initial comparison samples:** 100
- **shap version:** 0.53.0rc1.dev45
- **numpy version:** 2.3.5
- **scikit-learn version:** 1.8.0

## Results

| run_id | explainer_type | n_samples | n_features | time (s) | status | validation | details |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | TreeExplainer | 100 | 10 | 1.232860 | success | passed |  |
| 2 | GPUTreeExplainer | 100 | 10 | 0.449615 | success | passed |  |
| 3 | TreeExplainer | 1000 | 10 | 12.265858 | success | passed |  |
| 4 | GPUTreeExplainer | 1000 | 10 | 0.459425 | success | passed |  |
| 5 | TreeExplainer | 10000 | 10 | 121.968134 | success | passed |  |
| 6 | GPUTreeExplainer | 10000 | 10 | 0.950597 | success | passed |  |
| 7 | TreeExplainer | 100000 | 10 | — | timeout | — | explanation exceeded 1200 s |
| 8 | GPUTreeExplainer | 100000 | 10 | 4.495710 | success | — |  |
| 9 | TreeExplainer | 1000000 | 10 | — | timeout | — | Skipped because run 7 timed out |
| 10 | GPUTreeExplainer | 1000000 | 10 | 41.491121 | success | — |  |
| 11 | TreeExplainer | 100 | 50 | 1.823056 | success | passed |  |
| 12 | GPUTreeExplainer | 100 | 50 | 0.420380 | success | passed |  |
| 13 | TreeExplainer | 1000 | 50 | 18.264983 | success | passed |  |
| 14 | GPUTreeExplainer | 1000 | 50 | 0.540699 | success | passed |  |
| 15 | TreeExplainer | 10000 | 50 | 181.863226 | success | passed |  |
| 16 | GPUTreeExplainer | 10000 | 50 | 1.056217 | success | passed |  |
