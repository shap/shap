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
| 7 | TreeExplainer | 1000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 108.397951 | success | passed |  |
| 8 | GPUTreeExplainer | 1000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1.508793 | success | passed |  |
| 9 | TreeExplainer | 100 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 54.569840 | success | passed |  |
| 10 | GPUTreeExplainer | 100 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 5.054335 | success | passed |  |
| 11 | TreeExplainer | 1000 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 539.167821 | success | passed |  |
| 12 | GPUTreeExplainer | 1000 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 7.073218 | success | passed |  |
| 13 | TreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 6.786257 | success | passed |  |
| 14 | GPUTreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 0.735354 | success | passed |  |
| 15 | TreeExplainer | 1000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 67.906196 | success | passed |  |
| 16 | GPUTreeExplainer | 1000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 0.938989 | success | passed |  |
| 17 | TreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 67.891035 | success | passed |  |
| 18 | GPUTreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 5.678551 | success | passed |  |
| 19 | TreeExplainer | 1000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 677.987598 | success | passed |  |
| 20 | GPUTreeExplainer | 1000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 7.667189 | success | passed |  |
| 21 | TreeExplainer | 100 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | 339.676040 | success | — |  |
| 22 | GPUTreeExplainer | 100 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | error | — | Worker exited with code -6 |
| 23 | TreeExplainer | 1000 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | timeout | — | explanation exceeded 1200 s |
| 24 | GPUTreeExplainer | 1000 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | error | — | Worker exited with code -6 |
| 25 | TreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 7.076970 | success | passed |  |
| 26 | GPUTreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 0.749676 | success | passed |  |
| 27 | TreeExplainer | 1000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 70.861994 | success | passed |  |
| 28 | GPUTreeExplainer | 1000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 0.968124 | success | passed |  |
| 29 | TreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 70.700641 | success | passed |  |
| 30 | GPUTreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 5.928134 | success | passed |  |
| 31 | TreeExplainer | 1000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 704.319163 | success | passed |  |
| 32 | GPUTreeExplainer | 1000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 7.965353 | success | passed |  |
| 33 | TreeExplainer | 100 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | 352.915928 | success | — |  |
| 34 | GPUTreeExplainer | 100 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | error | — | Worker exited with code -6 |
| 35 | TreeExplainer | 1000 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | timeout | — | explanation exceeded 1200 s |
| 36 | GPUTreeExplainer | 1000 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | error | — | Worker exited with code -6 |
| 37 | TreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 7.081516 | success | passed |  |
| 38 | GPUTreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 0.756111 | success | passed |  |
| 39 | TreeExplainer | 1000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 70.763583 | success | passed |  |
| 40 | GPUTreeExplainer | 1000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 0.935083 | success | passed |  |
| 41 | TreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 70.800637 | success | passed |  |
| 42 | GPUTreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 5.888630 | success | passed |  |
| 43 | TreeExplainer | 1000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 704.415363 | success | passed |  |
| 44 | GPUTreeExplainer | 1000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 7.916146 | success | passed |  |
| 45 | TreeExplainer | 100 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | 353.429282 | success | — |  |
| 46 | GPUTreeExplainer | 100 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | error | — | Worker exited with code -6 |
| 47 | TreeExplainer | 1000 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | timeout | — | explanation exceeded 1200 s |
| 48 | GPUTreeExplainer | 1000 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | error | — | Worker exited with code -6 |
