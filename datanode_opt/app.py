from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import time
import argparse
import logging

from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_spark_session(optimized: bool = False) -> SparkSession:
    builder = SparkSession.builder.master("local").appName("spark_opt_lab")

    if optimized:
        builder = (
            builder.config("spark.sql.shuffle.partitions", "100")
            .config("spark.default.parallelism", "100")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
            .config("spark.sql.files.maxPartitionBytes", "134217728")
            .config("spark.sql.autoBroadcastJoinThreshold", "10485760")
        )

    return builder.getOrCreate()


def load_data(spark: SparkSession) -> DataFrame:
    data_path = Path("data/final_animedataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Please run download_dataset.py first"
        )

    logger.info("Loading dataset from local storage")
    return spark.read.format("csv").load(str(data_path), header=True, inferSchema=True)


def prepare_data(df: DataFrame) -> DataFrame:
    return df.select(
        col("user_id").cast("integer"),
        col("anime_id").cast("integer"),
        col("my_score").cast("float"),
    )


def split_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    logger.info("Splitting dataset into train and test sets")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Train set size: {train_df.count()}, Test set size: {test_df.count()}")
    return train_df, test_df


def train_model(train_df: DataFrame) -> ALS:
    logger.info("Starting model training")
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="user_id",
        itemCol="anime_id",
        ratingCol="my_score",
        coldStartStrategy="drop",
    )
    model = als.fit(train_df)
    logger.info("Model training completed")
    return model


def evaluate_model(model: ALS, test_df: DataFrame) -> Tuple[float, float]:
    logger.info("Starting model evaluation")
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="my_score", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)

    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)

    logger.info("Model evaluation completed")

    return rmse, mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", action="store_true", help="Enable Spark optimizations")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    opt = args.opt  # type: ignore

    logger.info(
        f"Starting Spark session with optimizations={'enabled' if opt else 'disabled'}"
    )
    spark = create_spark_session(optimized=opt)

    start_time = time.time()

    df = load_data(spark)
    prepared_df = prepare_data(df)
    train_df, test_df = split_data(prepared_df)

    model = train_model(train_df)
    rmse, mae = evaluate_model(model, test_df)

    end_time = time.time()

    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logger.info(f"Root Mean Square Error: {rmse:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
