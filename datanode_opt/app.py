from pyspark.sql import SparkSession
import kagglehub
from pathlib import Path
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import time
import argparse

def create_spark_session(optimized=False):
    builder = (SparkSession
              .builder
              .master('local')
              .appName('spark_opt_lab'))
    
    if optimized:
        builder = (builder
                  .config("spark.sql.shuffle.partitions", "100")
                  .config("spark.default.parallelism", "100")
                  .config("spark.memory.fraction", "0.8")
                  .config("spark.memory.storageFraction", "0.3")
                  .config("spark.sql.adaptive.enabled", "true")
                  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                  .config("spark.dynamicAllocation.enabled", "true")
                  .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
                  .config("spark.sql.files.maxPartitionBytes", "134217728")
                  .config("spark.sql.autoBroadcastJoinThreshold", "10485760"))
    
    return builder.getOrCreate()

def load_data(spark):
    path = Path(kagglehub.dataset_download("dbdmobile/myanimelist-dataset"))
    table_name = "final_animedataset.csv"
    return spark.read.format("csv").load(path / table_name, header=True, inferSchema=True)

def prepare_data(df):
    return df.select(
        col("user_id").cast("integer"),
        col("anime_id").cast("integer"),
        col("my_score").cast("float")
    )

def split_data(df):
    return df.randomSplit([0.8, 0.2], seed=42)

def train_model(train_df):
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="user_id",
        itemCol="anime_id",
        ratingCol="my_score",
        coldStartStrategy="drop"
    )
    return als.fit(train_df)

def evaluate_model(model, test_df):
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="my_score",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)
    
    return rmse, mae

def parse_args():
    parser = argparse.ArgumentParser(description='Anime Recommendations using PySpark')
    parser.add_argument('--opt', action='store_true', help='Enable Spark optimizations')
    return parser.parse_args()

def main():
    args = parse_args()
    spark = create_spark_session(optimized=args.opt)
    
    start_time = time.time()
    
    df = load_data(spark)
    prepared_df = prepare_data(df)
    train_df, test_df = split_data(prepared_df)
    
    model = train_model(train_df)
    rmse, mae = evaluate_model(model, test_df)
    
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    spark.stop()

if __name__ == "__main__":
    main()