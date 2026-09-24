file_path = "tests/plots/test_summary.py"
with open(file_path) as f:
    content = f.read()

# Replace np.random.seed(0) with rs = np.random.RandomState(0)
content = content.replace("np.random.seed(0)", "rs = np.random.RandomState(0)")

# Replace np.random.randn with rs.randn
content = content.replace("np.random.randn", "rs.randn")

# Replace np.random.randint with rs.randint
content = content.replace("np.random.randint", "rs.randint")

# Manually replace the shap.summary_plot calls in tests to include rng=rs.
replacements = [
    ("shap.summary_plot(rs.randn(20, 5), show=False)", "shap.summary_plot(rs.randn(20, 5), show=False, rng=rs)"),
    (
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 5), show=False)",
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 5), show=False, rng=rs)",
    ),
    (
        "shap.summary_plot([rs.randn(20, 5) for i in range(3)], rs.randn(20, 5), show=False)",
        "shap.summary_plot([rs.randn(20, 5) for i in range(3)], rs.randn(20, 5), show=False, rng=rs)",
    ),
    (
        "shap.summary_plot(\n        [rs.randn(20, 5) for i in range(3)], rs.randn(20, 5), show=False, show_values_in_legend=True\n    )",
        "shap.summary_plot(\n        [rs.randn(20, 5) for i in range(3)], rs.randn(20, 5), show=False, show_values_in_legend=True, rng=rs\n    )",
    ),
    (
        "show_values_in_legend=True,",
        "show_values_in_legend=True,\n        rng=rs,",
    ),  # For test_summary_multi_class_legend
    ('plot_type="bar", show=False', 'plot_type="bar", show=False, rng=rs'),
    ('plot_type="dot", show=False', 'plot_type="dot", show=False, rng=rs'),
    ('plot_type="compact_dot", show=False', 'plot_type="compact_dot", show=False, rng=rs'),
    ('plot_type="violin", show=False', 'plot_type="violin", show=False, rng=rs'),
    (
        'plot_type="layered_violin",\n        show=False,\n    )',
        'plot_type="layered_violin",\n        show=False,\n        rng=rs,\n    )',
    ),
    ("use_log_scale=True, show=False", "use_log_scale=True, show=False, rng=rs"),
    ('feature_names=["foo", "bar", "baz"], show=False', 'feature_names=["foo", "bar", "baz"], show=False, rng=rs'),
    ("feature_names=feature_names, show=False", "feature_names=feature_names, show=False, rng=rs"),
    ("shap.summary_plot(shap_values, X)\n", "shap.summary_plot(shap_values, X, rng=rs)\n"),
    (
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 4), show=False)",
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 4), show=False, rng=rs)",
    ),
    (
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 1), show=False)",
        "shap.summary_plot(rs.randn(20, 5), rs.randn(20, 1), show=False, rng=rs)",
    ),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(file_path, "w") as f:
    f.write(content)

print("Updated test_summary.py")
