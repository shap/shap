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
- **Started (UTC):** 2026-10-06T23:59:00+00:00
- **Model:** RandomForestRegressor(max_depth=grid, n_estimators=grid, random_state=0); other parameters use sklearn defaults
- **Depth convention:** Root depth is 0; max_depth is a configured limit, actual depths are measured after fitting
- **Tree depth statistics:** Min, max and mean of each tree's maximum path depth across the forest
- **Training data:** One shared make_regression invocation: 20000 samples, 50 features, 10 informative features, 1 target, noise=0.0, random_state=0
- **Explanation data:** Standard normal, numpy default_rng seed=1
- **Feature perturbation:** tree_path_dependent
- **Validation:** First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4
- **Max depth limits:** [12, 20, 28, 36]
- **Tree counts:** [10, 100, 500]
- **Explanation shape (samples, features):** [10000, 50]
- **Initial comparison samples:** 100
- **shap version:** 0.53.0rc1.dev45
- **numpy version:** 2.3.5
- **scikit-learn version:** 1.8.0

## Results

| run_id | explainer_type | n_samples | n_features | max_depth | n_trees | min–max tree depth | mean tree depth | mean leaves/tree | total_nodes | time (s) | status | validation | details |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | TreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 1.060995 | success | passed |  |
| 2 | GPUTreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.360192 | success | passed |  |
| 3 | TreeExplainer | 10000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 105.464093 | success | passed |  |
| 4 | GPUTreeExplainer | 10000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.728198 | success | passed |  |
| 5 | TreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 10.803278 | success | passed |  |
| 6 | GPUTreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1.267907 | success | passed |  |
| 7 | TreeExplainer | 10000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1074.477273 | success | passed |  |
| 8 | GPUTreeExplainer | 10000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 5.111447 | success | passed |  |
| 9 | TreeExplainer | 100 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 53.987438 | success | passed |  |
| 10 | GPUTreeExplainer | 100 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 5.146116 | success | passed |  |
| 11 | TreeExplainer | 10000 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | — | timeout | — | explanation exceeded 1200 s |
| 12 | GPUTreeExplainer | 10000 | 50 | 12 | 500 | 12–12 | 12.00 | 3036.54 | 3036040 | 25.365643 | success | — |  |
| 13 | TreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 6.816745 | success | passed |  |
| 14 | GPUTreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 0.729977 | success | passed |  |
| 15 | TreeExplainer | 10000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 677.720483 | success | passed |  |
| 16 | GPUTreeExplainer | 10000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 2.883020 | success | passed |  |
| 17 | TreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 67.570846 | success | passed |  |
| 18 | GPUTreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 5.772333 | success | passed |  |
| 19 | TreeExplainer | 10000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | — | timeout | — | explanation exceeded 1200 s |
| 20 | GPUTreeExplainer | 10000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 27.585694 | success | — |  |
| 21 | TreeExplainer | 100 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | 339.740217 | success | — |  |
| 22 | GPUTreeExplainer | 100 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | error | — | Worker exited with code -6 |
| 23 | TreeExplainer | 10000 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | timeout | — | explanation exceeded 1200 s |
| 24 | GPUTreeExplainer | 10000 | 50 | 20 | 500 | 20–20 | 20.00 | 12379.94 | 12379442 | — | error | — | Worker exited with code -6 |
| 25 | TreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 7.055933 | success | passed |  |
| 26 | GPUTreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 0.749848 | success | passed |  |
| 27 | TreeExplainer | 10000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 706.595103 | success | passed |  |
| 28 | GPUTreeExplainer | 10000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 2.990825 | success | passed |  |
| 29 | TreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 70.362166 | success | passed |  |
| 30 | GPUTreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 5.791929 | success | passed |  |
| 31 | TreeExplainer | 10000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | — | timeout | — | explanation exceeded 1200 s |
| 32 | GPUTreeExplainer | 10000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 28.540274 | success | — |  |
| 33 | TreeExplainer | 100 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | 353.499948 | success | — |  |
| 34 | GPUTreeExplainer | 100 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | error | — | Worker exited with code -6 |
| 35 | TreeExplainer | 10000 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | timeout | — | explanation exceeded 1200 s |
| 36 | GPUTreeExplainer | 10000 | 50 | 28 | 500 | 23–28 | 26.20 | 12646.97 | 12646468 | — | error | — | Worker exited with code -6 |
| 37 | TreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 7.164095 | success | passed |  |
| 38 | GPUTreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 0.776792 | success | passed |  |
| 39 | TreeExplainer | 10000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 707.376079 | success | passed |  |
| 40 | GPUTreeExplainer | 10000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 3.000875 | success | passed |  |
| 41 | TreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 70.634867 | success | passed |  |
| 42 | GPUTreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 5.856650 | success | passed |  |
| 43 | TreeExplainer | 10000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | — | timeout | — | explanation exceeded 1200 s |
| 44 | GPUTreeExplainer | 10000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 28.574012 | success | — |  |
| 45 | TreeExplainer | 100 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | 353.062163 | success | — |  |
| 46 | GPUTreeExplainer | 100 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | error | — | Worker exited with code -6 |
| 47 | TreeExplainer | 10000 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | timeout | — | explanation exceeded 1200 s |
| 48 | GPUTreeExplainer | 10000 | 50 | 36 | 500 | 23–33 | 26.29 | 12647.13 | 12646626 | — | error | — | Worker exited with code -6 |
