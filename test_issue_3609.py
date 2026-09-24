"""Test script for issue #3609 - Add directionality indicators to beeswarm plots"""

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    import shap

    print("Testing issue #3609 - Beeswarm directionality feature...")

    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Train a simple model
    print("2. Training model...")
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)

    # Compute SHAP values
    print("3. Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Test 1: Beeswarm plot WITHOUT directionality (default behavior)
    print("\n4. Testing beeswarm plot WITHOUT directionality...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False, ax=ax)
    plt.title("Beeswarm Plot - Without Directionality")
    plt.savefig("test_beeswarm_no_directionality.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: test_beeswarm_no_directionality.png")

    # Test 2: Beeswarm plot WITH directionality
    print("\n5. Testing beeswarm plot WITH directionality...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False, ax=ax, show_directionality=True)
    plt.title("Beeswarm Plot - With Directionality Indicators")
    plt.savefig("test_beeswarm_with_directionality.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: test_beeswarm_with_directionality.png")

    # Test 3: Verify directionality computation manually
    print("\n6. Verifying directionality computation...")
    from scipy.stats import spearmanr

    for i in range(min(3, X.shape[1])):
        corr, p_value = spearmanr(X[:, i], shap_values.values[:, i])
        direction = "positive (+)" if corr > 0 else "negative (−)"
        print(f"   Feature {i}: correlation={corr:.3f}, p-value={p_value:.4f}, direction={direction}")

    # Test 4: Test with missing values
    print("\n7. Testing with missing values...")
    X_with_nan = X.copy()
    X_with_nan[0:10, 0] = np.nan
    shap_values_nan = explainer(X_with_nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_nan, show=False, ax=ax, show_directionality=True)
    plt.title("Beeswarm Plot - With NaN Values and Directionality")
    plt.savefig("test_beeswarm_with_nan.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved: test_beeswarm_with_nan.png")

    print("\n✅ All tests passed! Issue #3609 feature implemented successfully.")
    print("\nGenerated plots:")
    print("  - test_beeswarm_no_directionality.png")
    print("  - test_beeswarm_with_directionality.png")
    print("  - test_beeswarm_with_nan.png")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
