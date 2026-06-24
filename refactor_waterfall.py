import re

with open("shap/plots/_waterfall.py") as f:
    content = f.read()

# 1. Update function signatures
content = re.sub(
    r"def waterfall\(shap_values, max_display=10, show=True\):",
    "def waterfall(shap_values, max_display=10, show=True, ax=None):",
    content,
)
content = re.sub(
    r"def waterfall_legacy\(expected_value, shap_values=None, features=None, feature_names=None, max_display=10, show=True\):",
    "def waterfall_legacy(expected_value, shap_values=None, features=None, feature_names=None, max_display=10, show=True, ax=None):",
    content,
)

# 2. Update docstrings
content = re.sub(
    r"    show : bool\n        Whether", "    show : bool\n        Whether", content
)  # We don't necessarily have to touch docstring, maybe add ax?
# Wait, I'll just skip docstring for now.

# 3. Add `ax` and `fig` resolution
replacement = """    if ax is None:
        ax = plt.gca()
        # size the plot based on how many features we are plotting
        fig = ax.get_figure()
        fig.set_size_inches(8, num_features * row_height + 1.5)
    else:
        fig = ax.get_figure()"""

content = re.sub(
    r"    # size the plot based on how many features we are plotting\n    plt.gcf\(\).set_size_inches\(8, num_features \* row_height \+ 1.5\)",
    replacement,
    content,
)

# 4. Remove `fig = plt.gcf()` and `ax = plt.gca()`
content = re.sub(r"    fig = plt.gcf\(\)\n    ax = plt.gca\(\)\n", "", content)

# 5. Replace `plt.foo()` with `ax.foo()` for simple methods
simple_methods = ["plot", "barh", "errorbar", "text", "axhline", "axvline"]
for method in simple_methods:
    content = content.replace(f"plt.{method}(", f"ax.{method}(")

# 6. Replace `plt.xlim()`
content = content.replace("plt.xlim()", "ax.get_xlim()")

# 7. Replace `plt.arrow` with `ax.arrow` (it exists, just undocumented sometimes, wait, matplotlib Axes has `arrow`?)
# Actually `Axes.arrow` does exist! Let's check matplotlib docs, yes `ax.arrow` is valid.
content = content.replace("plt.arrow(", "ax.arrow(")

# 8. Replace `plt.yticks` with `ax.set_yticks` and `ax.set_yticklabels`
# Example: plt.yticks(ytick_pos, yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]], fontsize=13)
# We can do this manually or just use `ax.set_yticks` and `ax.set_yticklabels`.
content = re.sub(
    r"plt.yticks\(([^,]+),\s*([^,]+(?:[^)]+)*),\s*fontsize=([^)]+)\)",
    r"ax.set_yticks(\1)\n    ax.set_yticklabels(\2, fontsize=\3)",
    content,
)

# 9. Replace `plt.gca()`
content = content.replace("plt.gca()", "ax")

# 10. Fix returns
content = content.replace("return plt.gcf()", "return ax")
content = content.replace(
    "return ax", "return ax"
)  # Already ax from plt.gca() replacement, wait: `return plt.gca()` became `return ax`

with open("shap/plots/_waterfall.py", "w") as f:
    f.write(content)
