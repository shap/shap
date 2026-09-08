"""Utilities for running pyspark-based explainers in constrained CI environments."""


def create_resource_safe_spark_session(pyspark):
    """Create a Spark session with conservative defaults to reduce OOM risk in CI."""
    conf = (
        pyspark.SparkConf()
        .set("spark.master", "local[2]")
        .set("spark.driver.memory", "2g")
        .set("spark.driver.maxResultSize", "1g")
        .set("spark.sql.shuffle.partitions", "8")
        .set("spark.default.parallelism", "8")
        .set("spark.python.worker.reuse", "false")
        .set("spark.ui.enabled", "false")
    )
    return pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
