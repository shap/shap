from shap.explainers._pyspark_utils import create_resource_safe_spark_session


def test_create_resource_safe_spark_session_uses_guardrails():
    conf_values = {}

    class DummySparkConf:
        def set(self, key, value):
            conf_values[key] = value
            return self

    class DummyBuilder:
        def config(self, conf):
            return self

        def getOrCreate(self):
            return "dummy-spark-session"

    class DummySparkSession:
        builder = DummyBuilder()

    class DummySQL:
        SparkSession = DummySparkSession

    class DummyPySpark:
        SparkConf = DummySparkConf
        sql = DummySQL

    session = create_resource_safe_spark_session(DummyPySpark())

    assert session == "dummy-spark-session"
    assert conf_values["spark.master"] == "local[2]"
    assert conf_values["spark.driver.memory"] == "2g"
    assert conf_values["spark.sql.shuffle.partitions"] == "8"
    assert conf_values["spark.default.parallelism"] == "8"
    assert conf_values["spark.ui.enabled"] == "false"
    assert conf_values["spark.driver.maxResultSize"] == "1g"
    assert conf_values["spark.python.worker.reuse"] == "false"
