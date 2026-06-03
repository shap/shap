from __future__ import annotations

import pickle

import numpy as np


def shap_values_distributed(explainer, spark_df, features_col: str, spark=None):
    try:
        import pyspark.sql.functions as F
        from pyspark.sql import SparkSession
        from pyspark.sql.types import ArrayType, DoubleType
    except ImportError as e:
        raise ImportError(
            "PySpark is required for shap_values_distributed. Install it with: pip install pyspark"
        ) from e

    try:
        pickle.dumps(explainer)
    except Exception as e:
        raise ValueError(
            f"The explainer must be picklable to broadcast to Spark workers. Pickling failed with: {e}"
        ) from e

    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    explainer_bc = spark.sparkContext.broadcast(explainer)

    @F.udf(returnType=ArrayType(DoubleType()))
    def _shap_udf(features):
        if features is None:
            return None
        exp = explainer_bc.value
        x = np.array(features.toArray()).reshape(1, -1)
        vals = exp.shap_values(x)
        if isinstance(vals, list):
            return vals[0][0].tolist()
        return vals[0].tolist()

    return spark_df.withColumn("shap_values", _shap_udf(F.col(features_col)))
