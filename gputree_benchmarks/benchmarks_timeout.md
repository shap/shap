# Tree explainer benchmark — CPU: Ryzen 5 3600 | GPU: unknown

## Metadata

- **CPU:** AMD Ryzen 5 3600 6-Core Processor
- **GPU:** []
- **OS:** Linux-7.0.0-34-generic-x86_64-with-glibc2.39
- **Python:** 3.15.0rc3
- **Explanation timeout (seconds):** 1200
- **Setup timeout (seconds):** 180
- **Timing scope:** Explainer construction and exp(X); excludes process startup and data creation
- **Skip rule:** More samples at the same feature count and model configuration, for the same explainer type, after an explanation timeout
- **Started (UTC):** 2026-10-11T20:48:47+00:00
- **Model:** RandomForestRegressor(max_depth=grid, n_estimators=grid, random_state=0); other parameters use sklearn defaults
- **Depth convention:** Root depth is 0; max_depth is a configured limit, actual depths are measured after fitting
- **Tree depth statistics:** Min, max and mean of each tree's maximum path depth across the forest
- **Training data:** One shared make_regression invocation: 20000 samples, 50 features, 10 informative features, 1 target, noise=0.0, random_state=0
- **Explanation data:** Standard normal, numpy default_rng seed=1
- **Feature perturbation:** tree_path_dependent
- **Validation:** First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4
- **Max depth limits:** [12, 20, 28, 36]
- **Tree counts:** [10, 50, 100]
- **Explanation sample counts:** [100, 1000, 10000]
- **Explanation feature count:** 50
- **shap version:** 0.53.0rc1.dev48
- **numpy version:** 2.4.6
- **scikit-learn version:** 1.8.0

## Results

| run_id | explainer_type | n_samples | n_features | max_depth | n_trees | min–max tree depth | mean tree depth | mean leaves/tree | total_nodes | time (s) | status | validation | details |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | TreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 1.030029 | success | passed |  |
| 2 | GPUTreeExplainer | 100 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.332311 | success | passed |  |
| 3 | TreeExplainer | 1000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 10.045078 | success | passed |  |
| 4 | GPUTreeExplainer | 1000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.367382 | success | passed |  |
| 5 | TreeExplainer | 10000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 100.758563 | success | passed |  |
| 6 | GPUTreeExplainer | 10000 | 50 | 12 | 10 | 12–12 | 12.00 | 3039.30 | 60776 | 0.722338 | success | passed |  |
| 7 | TreeExplainer | 100 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 5.069914 | success | passed |  |
| 8 | GPUTreeExplainer | 100 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 0.717574 | success | passed |  |
| 9 | TreeExplainer | 1000 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 50.767530 | success | passed |  |
| 10 | GPUTreeExplainer | 1000 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 0.922504 | success | passed |  |
| 11 | TreeExplainer | 10000 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 508.749261 | success | passed |  |
| 12 | GPUTreeExplainer | 10000 | 50 | 12 | 50 | 12–12 | 12.00 | 3041.98 | 304148 | 2.565245 | success | passed |  |
| 13 | TreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 10.256870 | success | passed |  |
| 14 | GPUTreeExplainer | 100 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1.212190 | success | passed |  |
| 15 | TreeExplainer | 1000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 102.602743 | success | passed |  |
| 16 | GPUTreeExplainer | 1000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1.561042 | success | passed |  |
| 17 | TreeExplainer | 10000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 1020.098531 | success | passed |  |
| 18 | GPUTreeExplainer | 10000 | 50 | 12 | 100 | 12–12 | 12.00 | 3041.49 | 608198 | 4.941707 | success | passed |  |
| 19 | TreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 6.409079 | success | passed |  |
| 20 | GPUTreeExplainer | 100 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 0.684632 | success | passed |  |
| 21 | TreeExplainer | 1000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 64.112309 | success | passed |  |
| 22 | GPUTreeExplainer | 1000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 0.898356 | success | passed |  |
| 23 | TreeExplainer | 10000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 642.966197 | success | passed |  |
| 24 | GPUTreeExplainer | 10000 | 50 | 20 | 10 | 20–20 | 20.00 | 12365.40 | 247298 | 2.868405 | success | passed |  |
| 25 | TreeExplainer | 100 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | 32.045977 | success | passed |  |
| 26 | GPUTreeExplainer | 100 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | 2.757589 | success | passed |  |
| 27 | TreeExplainer | 1000 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | 321.217299 | success | passed |  |
| 28 | GPUTreeExplainer | 1000 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | 3.778700 | success | passed |  |
| 29 | TreeExplainer | 10000 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | — | timeout | — | explanation exceeded 1200 s |
| 30 | GPUTreeExplainer | 10000 | 50 | 20 | 50 | 20–20 | 20.00 | 12372.48 | 1237198 | 13.593839 | success | — |  |
| 31 | TreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 64.694526 | success | passed |  |
| 32 | GPUTreeExplainer | 100 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 5.236929 | success | passed |  |
| 33 | TreeExplainer | 1000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 643.206509 | success | passed |  |
| 34 | GPUTreeExplainer | 1000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 7.235985 | success | passed |  |
| 35 | TreeExplainer | 10000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | — | timeout | — | explanation exceeded 1200 s |
| 36 | GPUTreeExplainer | 10000 | 50 | 20 | 100 | 20–20 | 20.00 | 12369.47 | 2473794 | 27.201626 | success | — |  |
| 37 | TreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 6.699831 | success | passed |  |
| 38 | GPUTreeExplainer | 100 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 0.708059 | success | passed |  |
| 39 | TreeExplainer | 1000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 66.919986 | success | passed |  |
| 40 | GPUTreeExplainer | 1000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 0.918387 | success | passed |  |
| 41 | TreeExplainer | 10000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 671.036212 | success | passed |  |
| 42 | GPUTreeExplainer | 10000 | 50 | 28 | 10 | 25–28 | 26.60 | 12644.50 | 252880 | 2.974616 | success | passed |  |
| 43 | TreeExplainer | 100 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | 33.633977 | success | passed |  |
| 44 | GPUTreeExplainer | 100 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | 2.842406 | success | passed |  |
| 45 | TreeExplainer | 1000 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | 335.806754 | success | passed |  |
| 46 | GPUTreeExplainer | 1000 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | 3.794819 | success | passed |  |
| 47 | TreeExplainer | 10000 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | — | timeout | — | explanation exceeded 1200 s |
| 48 | GPUTreeExplainer | 10000 | 50 | 28 | 50 | 24–28 | 26.38 | 12641.58 | 1264108 | 14.125569 | success | — |  |
| 49 | TreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 66.855448 | success | passed |  |
| 50 | GPUTreeExplainer | 100 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 5.428264 | success | passed |  |
| 51 | TreeExplainer | 1000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 671.438436 | success | passed |  |
| 52 | GPUTreeExplainer | 1000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 7.478184 | success | passed |  |
| 53 | TreeExplainer | 10000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | — | timeout | — | explanation exceeded 1200 s |
| 54 | GPUTreeExplainer | 10000 | 50 | 28 | 100 | 24–28 | 26.25 | 12642.14 | 2528328 | 28.075126 | success | — |  |
| 55 | TreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 6.675933 | success | passed |  |
| 56 | GPUTreeExplainer | 100 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 0.722924 | success | passed |  |
| 57 | TreeExplainer | 1000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 67.054771 | success | passed |  |
| 58 | GPUTreeExplainer | 1000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 0.921688 | success | passed |  |
| 59 | TreeExplainer | 10000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 671.704858 | success | passed |  |
| 60 | GPUTreeExplainer | 10000 | 50 | 36 | 10 | 25–29 | 26.70 | 12644.60 | 252882 | 2.967417 | success | passed |  |
| 61 | TreeExplainer | 100 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | 33.380680 | success | passed |  |
| 62 | GPUTreeExplainer | 100 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | 2.802596 | success | passed |  |
| 63 | TreeExplainer | 1000 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | 336.544205 | success | passed |  |
| 64 | GPUTreeExplainer | 1000 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | 3.780809 | success | passed |  |
| 65 | TreeExplainer | 10000 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | — | timeout | — | explanation exceeded 1200 s |
| 66 | GPUTreeExplainer | 10000 | 50 | 36 | 50 | 24–32 | 26.52 | 12641.88 | 1264138 | 14.121842 | success | — |  |
| 67 | TreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 67.101751 | success | passed |  |
| 68 | GPUTreeExplainer | 100 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 5.411811 | success | passed |  |
| 69 | TreeExplainer | 1000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 673.044910 | success | passed |  |
| 70 | GPUTreeExplainer | 1000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 7.459132 | success | passed |  |
| 71 | TreeExplainer | 10000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | — | timeout | — | explanation exceeded 1200 s |
| 72 | GPUTreeExplainer | 10000 | 50 | 36 | 100 | 24–33 | 26.40 | 12642.45 | 2528390 | 28.156919 | success | — |  |
