import sys
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def configure_pyspark_python(monkeypatch):
    monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.mark.skipif(sys.platform == "win32", reason="PySpark OOM issues on Windows")
def test_shap_values_distributed_tree(configure_pyspark_python):
    pyspark = pytest.importorskip("pyspark")
    pytest.importorskip("pyspark.ml")
    pytest.importorskip("sklearn")

    import shap
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from sklearn.datasets import load_iris

    try:
        spark = (
            pyspark.sql.SparkSession.builder
            .config("spark.master", "local[2]")
            .getOrCreate()
        )
    except Exception:
        pytest.skip("Could not start local Spark session")

    iris = load_iris()
    pdf = pd.DataFrame(iris.data, columns=iris.feature_names)
    pdf["label"] = iris.target.astype(float)

    sdf = spark.createDataFrame(pdf)
    sdf = VectorAssembler(
        inputCols=list(iris.feature_names), outputCol="features"
    ).transform(sdf)

    model = RandomForestClassifier(
        labelCol="label", featuresCol="features", numTrees=10
    ).fit(sdf)

    explainer = shap.TreeExplainer(model)
    result = shap.utils.shap_values_distributed(
        explainer, sdf, features_col="features", spark=spark
    )

    assert "shap_values" in result.columns

    shap_pdf = result.select("shap_values").toPandas()
    assert len(shap_pdf) == len(pdf)
    assert len(shap_pdf["shap_values"].iloc[0]) == len(iris.feature_names)

    spark.stop()