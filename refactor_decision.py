import re

with open("shap/plots/_decision.py") as f:
    content = f.read()

# 1. Update `decision` signature
content = content.replace(
    '    legend_labels=None,\n    legend_location="best",\n) -> DecisionPlotResult | None:',
    '    legend_labels=None,\n    legend_location="best",\n    ax=None,\n) -> DecisionPlotResult | None:',
)

# 2. Update `decision` docstring
content = content.replace(
    "    legend_location : str\n        The matplotlib legend location string.\n",
    "    legend_location : str\n        The matplotlib legend location string.\n\n    ax : matplotlib.axes.Axes, optional\n        Axes object to draw the plot onto, otherwise uses the current Axes.\n",
)

# 3. Update `decision` passing `ax` to `__decision_plot_matplotlib`
content = content.replace(
    "        title,\n        show,\n        legend_labels,\n        legend_location,\n    )",
    "        title,\n        show,\n        legend_labels,\n        legend_location,\n        ax=ax,\n    )",
)

# 4. Update `__decision_plot_matplotlib` signature
content = content.replace(
    "    title,\n    show,\n    legend_labels,\n    legend_location,\n):",
    "    title,\n    show,\n    legend_labels,\n    legend_location,\n    ax=None,\n):",
)

# 5. Initialize `ax` inside `__decision_plot_matplotlib` and replace `plt.gcf().set_size_inches`
content = content.replace(
    "    # image size\n    row_height = 0.4\n    if auto_size_plot:\n        plt.gcf().set_size_inches(8, feature_display_count * row_height + 1.5)",
    "    if ax is None:\n        ax = plt.gca()\n\n    # image size\n    row_height = 0.4\n    if auto_size_plot:\n        ax.get_figure().set_size_inches(8, feature_display_count * row_height + 1.5)",
)

# 6. Replace specific plotting calls with regex
content = re.sub(r"plt\.axvline\((.*?)\)", r"ax.axvline(\1)", content)
content = re.sub(r"plt\.axhline\((.*?)\)", r"ax.axhline(\1)", content)

# 7. Replace `ax = plt.gca()` because we already initialized `ax` at the top
content = content.replace(
    "    # plot each observation's cumulative SHAP values.\n    ax = plt.gca()\n",
    "    # plot each observation's cumulative SHAP values.\n",
)

# 8. Replace `plt.plot` with `ax.plot`
content = re.sub(r"plt\.plot\((.*?)\)", r"ax.plot(\1)", content)

# 9. Replace `plt.gcf().canvas...` with `ax.get_figure().canvas...`
content = content.replace("plt.gcf().canvas", "ax.get_figure().canvas")
content = content.replace("plt.gca()", "ax")

# 10. Replace ticks, limits, labels
content = re.sub(r"plt\.yticks\((.*?)\)", r"ax.set_yticks(\1)", content)
content = content.replace(
    "ax.set_yticks(np.arange(feature_display_count) + 0.5, feature_names, fontsize=fontsize)",
    "ax.set_yticks(np.arange(feature_display_count) + 0.5)\n    ax.set_yticklabels(feature_names, fontsize=fontsize)",
)

content = re.sub(r"plt\.ylim\((.*?)\)", r"ax.set_ylim(\1)", content)
content = re.sub(r"plt\.xlabel\((.*?)\)", r"ax.set_xlabel(\1)", content)

# 11. Colorbar refactor
content = content.replace(
    'cb = plt.colorbar(m, ticks=[0, 1], orientation="horizontal", cax=ax_cb)',
    'cb = ax.get_figure().colorbar(m, ticks=[0, 1], orientation="horizontal", cax=ax_cb)',
)

# 12. Remove `plt.sca(ax)`
content = content.replace("    # re-activate the main axis for drawing.\n        plt.sca(ax)\n", "")

# 13. Replace title and invert_yaxis
content = content.replace("plt.title(title)", "ax.set_title(title)")
content = content.replace("plt.gca().invert_yaxis()", "ax.invert_yaxis()")

# 14. Return ax instead of None if show=False
# The old code just ends at:
#     if show:
#         plt.show()
# We should return ax if show is False?
# Wait, `decision()` explicitly returns a `DecisionPlotResult` object.
# We shouldn't change the return type of `decision()`. `__decision_plot_matplotlib()` doesn't return anything. It modifies the axes. That is perfectly fine, we don't need to return `ax` since the user has access to it via the argument they passed, or if they used `plt.gca()`.
# For waterfall, we returned `plt.gcf()`. For violin, we returned `ax`.
# But `decision()` returns `DecisionPlotResult`. We shouldn't change that.
# So I won't change the return value of `__decision_plot_matplotlib()`.

with open("shap/plots/_decision.py", "w") as f:
    f.write(content)
