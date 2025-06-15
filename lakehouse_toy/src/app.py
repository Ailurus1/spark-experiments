from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

import logging

from utils import create_directories, clean_directories, setup_mlflow, log_model_metrics
from etl_pipeline import create_etl_pipeline
from train import load_and_prepare_data, train_model, calculate_metrics, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session():
    spark = (
        SparkSession.builder.appName("CreditScoreETL")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
    )
    return configure_spark_with_delta_pip(spark).getOrCreate()


def main():
    spark = create_spark_session()

    create_directories(".")

    input_path = "data/data.csv"
    gold_path = "data/gold/credit_data"
    root_dir = "."

    setup_mlflow()

    try:
        etl = create_etl_pipeline(spark, root_dir, do_repartition=True)

        etl(input_path)

        logger.info("ETL pipeline completed successfully!")

        df, feature_cols = load_and_prepare_data(spark, gold_path)
        model = train_model(df, feature_cols)

        metrics = calculate_metrics(model, df)

        logger.info("Model Evaluation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        log_model_metrics(model, metrics, feature_cols)

        save_model(model, "credit_score_gbt")

        logger.info("Training completed successfully!")

    except Exception as exc:
        logger.error(f"Error: {exc}")
        clean_directories(".")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
