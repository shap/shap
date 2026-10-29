import re

with open("shap/plots/_violin.py") as f:
    content = f.read()

# 1. Update signature
content = content.replace("    use_log_scale=False,\n):", "    use_log_scale=False,\n    ax=None,\n):")

# 2. Update docstring
content = content.replace(
    "    use_log_scale : bool, optional\n        Whether to use a symmetric log scale for the x-axis.\n",
    "    use_log_scale : bool, optional\n        Whether to use a symmetric log scale for the x-axis.\n    ax : matplotlib.axes.Axes, optional\n        Axes object to draw the plot onto, otherwise uses the current Axes.\n",
)

# 3. Add `ax = ax or plt.gca()` at the start of the function body
content = content.replace(
    "    if title is not None:\n        warnings.warn",
    "    if ax is None:\n        ax = plt.gca()\n\n    if title is not None:\n        warnings.warn",
)

# 4. Replace plotting commands
# Use regex to replace plt.something with ax.something where appropriate
content = re.sub(r"plt\.xscale\((.*?)\)", r"ax.set_xscale(\1)", content)
content = re.sub(r"plt\.gcf\(\)\.set_size_inches\((.*?)\)", r"ax.get_figure().set_size_inches(\1)", content)
content = re.sub(r"plt\.axvline\((.*?)\)", r"ax.axvline(\1)", content)
content = re.sub(r"plt\.axhline\((.*?)\)", r"ax.axhline(\1)", content)
content = re.sub(r"plt\.scatter\((.*?)\)", r"ax.scatter(\1)", content)
content = re.sub(r"plt\.fill_between\((.*?)\)", r"ax.fill_between(\1)", content)
content = re.sub(r"plt\.violinplot\((.*?)\)", r"ax.violinplot(\1)", content)
content = re.sub(r"plt\.xlim\((.*?)\)", r"ax.set_xlim(\1)", content)
content = re.sub(r"plt\.ylim\((.*?)\)", r"ax.set_ylim(\1)", content)
content = re.sub(r"plt\.xlabel\((.*?)\)", r"ax.set_xlabel(\1)", content)
content = re.sub(r"plt\.yticks\((.*?)\)", r"ax.set_yticks(\1)", content)

# 5. Fix specific yticks usage (plt.yticks takes ticks and labels)
# It was: plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
# It should be: ax.set_yticks(range(len(feature_order))); ax.set_yticklabels([feature_names[i] for i in feature_order], fontsize=13)
content = content.replace(
    "ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)",
    "ax.set_yticks(range(len(feature_order)))\n    ax.set_yticklabels([feature_names[i] for i in feature_order], fontsize=13)",
)

# 6. Clean up plt.gca() and plt.gcf()
content = content.replace("plt.gca()", "ax")
content = content.replace("plt.gcf()", "ax.get_figure()")

# Wait, `cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)`
# plt.colorbar is fine, but it was `plt.colorbar(m, ax=plt.gca(), ...)`
# We replaced `plt.gca()` with `ax` above, so it is `ax=ax`.
# Wait, ax.get_figure().colorbar is more object-oriented, but plt.colorbar is acceptable if ax is passed. Let's use ax.get_figure().colorbar
content = content.replace(
    "cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)",
    "cb = ax.get_figure().colorbar(m, ax=ax, ticks=[0, 1], aspect=80)",
)

with open("shap/plots/_violin.py", "w") as f:
    f.write(content)
