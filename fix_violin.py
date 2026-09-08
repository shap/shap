import nbformat
from nbclient import NotebookClient

nb_path = "notebooks/api_examples/plots/violin.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        source = cell.source

        # Replace the model setup
        if "xgboost.train" in source:
            source = source.replace(
                'xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)',
                "xgboost.XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=0).fit(X, y)",
            )
            source = source.replace("bst", "model")

        if "shap.TreeExplainer" in source:
            source = source.replace("shap.TreeExplainer(model).shap_values(X)", "shap.Explainer(model)(X)")

        if "feat_names = list(X.columns)" in source:
            source = source.replace("feat_names = list(X.columns)\n", "")

        source = source.replace(", feature_names=feat_names", "")
        source = source.replace(", features=X", "")
        source = source.replace("features=X, ", "")

        cell.source = source.strip()

    if cell.cell_type == "markdown":
        source = cell.source
        if "Providing the feature names as a list" in source:
            cell.source = "With the modern `shap.Explanation` API, feature names are automatically extracted from the dataframe and included in the summary plot for readability:"
        if "We pass as parameters:" in source:
            cell.source = """Let us take the diabetes example :

We want to plot a layered violin summary plot based on our computed `shap_values` object.

We pass as parameters:
- our `shap_values` object (which inherently contains the features and their names)
- and the `plot_type` of interest : "layered_violin\""""

client = NotebookClient(nb, timeout=600, kernel_name="python3")
try:
    client.execute()
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    print("Successfully updated and executed notebook")
except Exception as e:
    print(f"Error executing notebook: {e}")
