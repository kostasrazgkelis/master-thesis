from django.conf import settings
from pyspark.sql import SparkSession


def get_spark_session(app_name, pipeline_id):
    config = settings.SPARK_CONFIG
    builder = SparkSession.builder.appName(f"{app_name}-{pipeline_id}")
    builder = builder.master(config["master"])

    for key, value in config["configs"].items():
        builder = builder.config(key, value)

    return builder.getOrCreate()
