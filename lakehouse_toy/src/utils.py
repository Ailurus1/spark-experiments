import os
import shutil
from pathlib import Path

import mlflow
import mlflow.spark
from pyspark.ml import PipelineModel
import logging

logger = logging.getLogger(__name__)


def setup_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit_scoring")


def log_model_metrics(model: PipelineModel, metrics: dict, feature_cols: list):
    with mlflow.start_run():
        mlflow.spark.log_model(
            model,
            "model",
            conda_env={
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.12",
                ],
            },
        )

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("feature_names", feature_cols)

        model_params = model.stages[-1].extractParamMap()
        for param, value in model_params.items():
            mlflow.log_param(param.name, value)

        logger.info("Model and metrics logged to MLflow successfully")


def create_directories(root: str) -> None:
    for dir_path in [
        Path(root) / "data/bronze",
        Path(root) / "data/silver",
        Path(root) / "data/gold",
    ]:
        os.makedirs(dir_path, exist_ok=True)


def clean_directories(root: str) -> None:
    for dir_path in [
        Path(root) / "data/bronze",
        Path(root) / "data/silver",
        Path(root) / "data/gold",
    ]:
        shutil.rmtree(dir_path)


def load_raw_data(spark, input_path, bronze_path):
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.write.format("delta").mode("overwrite").save(bronze_path)
