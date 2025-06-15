from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, regexp_extract, lit, avg, count, 
    stddev, percentile_approx, expr, abs as spark_abs,
    collect_list, struct
)
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from typing import List, Tuple, TypeVar, Callable
from functools import wraps
from pathlib import Path
import logging

T = TypeVar('T', DataFrame, Tuple[DataFrame, ...])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def repartition(spark: SparkSession):
    def repartition_output(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = func(*args, **kwargs)
                    
            if isinstance(result, DataFrame):
                return result.repartition(spark.sparkContext.defaultParallelism)
            
            elif isinstance(result, tuple):
                return tuple(
                    df.repartition(spark.sparkContext.defaultParallelism) 
                    if isinstance(df, DataFrame) else df 
                    for df in result
                )
            
            return result
        
        return wrapper
    return repartition_output

class ETLPipeline:
    def __init__(self, spark: SparkSession, root_dir: str):
        self.spark = spark
        self.bronze_path = f"{root_dir}/data/bronze/credit_data"
        self.silver_path = f"{root_dir}/data/silver/credit_data"
        self.gold_path = f"{root_dir}/data/gold/credit_data"
    
    def extract(self, input_path: str) -> None:
        df = self.spark.read.csv(input_path, header=True, inferSchema=False)
        
        def detect_column_type(df: DataFrame, col_name: str) -> str:
            sample = df.select(col_name).distinct().collect()
            values = [row[0] for row in sample if row[0] is not None]
            
            if not values:
                return "string"
            
            type_counts = {
                "int": 0,
                "double": 0,
                "boolean": 0,
                "date": 0
            }
            
            for value in values:
                try:
                    int(value)
                    type_counts["int"] += 1
                except:
                    pass
                    
                try:
                    float(value)
                    type_counts["double"] += 1
                except:
                    pass
                    
                try:
                    if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
                        type_counts["boolean"] += 1
                except:
                    pass
                    
                try:
                    from datetime import datetime
                    datetime.strptime(value, '%Y-%m-%d')
                    type_counts["date"] += 1
                except:
                    pass
            
            max_type = max(type_counts.items(), key=lambda x: x[1])
            
            if max_type[1] < len(values) * 0.5:
                return "string"
                
            return max_type[0]
    
        for col_name in df.columns:
            detected_type = detect_column_type(df, col_name)
            
            if detected_type != "string":
                df = df.withColumn(
                    col_name,
                    when(col(col_name).cast(detected_type).isNotNull(), 
                        col(col_name).cast(detected_type))
                    .otherwise(None)
                )
    
        df.write.format("delta").mode("overwrite").save(self.bronze_path)
    
    def transform(self) -> Tuple[DataFrame, DataFrame]:
        silver_df = self._process_silver_layer()
        
        gold_df = self._process_gold_layer(silver_df)
        
        return silver_df, gold_df
    
    def load(self, silver_df: DataFrame, gold_df: DataFrame) -> None:
        silver_df.write.format("delta")\
            .mode("overwrite")\
            .option("overwriteSchema", "true")\
            .save(self.silver_path)
        
        gold_df.write.format("delta")\
            .mode("overwrite")\
            .option("overwriteSchema", "true")\
            .save(self.gold_path)
    
    def _convert_credit_history_to_months(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "Credit_History_Months",
            when(col("Credit_History_Age").contains("Years and"),
                 expr("cast(regexp_extract(Credit_History_Age, '(\\d+) Years and', 1) as int) * 12 + " +
                      "cast(regexp_extract(Credit_History_Age, 'and (\\d+) Months', 1) as int)"))
            .when(col("Credit_History_Age").contains("Years"),
                  expr("cast(regexp_extract(Credit_History_Age, '(\\d+) Years', 1) as int) * 12"))
            .when(col("Credit_History_Age").contains("Months"),
                  expr("cast(regexp_extract(Credit_History_Age, '(\\d+) Months', 1) as int)"))
            .otherwise(None)
        )
    
    def _remove_outliers(self, df: DataFrame, numeric_cols: List[str], threshold: int = 3) -> DataFrame:
        for col_name in numeric_cols:
            mean = df.select(avg(col_name)).collect()[0][0]
            std = df.select(stddev(col_name)).collect()[0][0]
            df = df.filter(
                spark_abs((col(col_name) - mean) / std) < threshold
            )
        return df
    
    def _process_silver_layer(self) -> DataFrame:
        df = self.spark.read.format("delta").load(self.bronze_path)
        
        columns_to_remove = ["ID", "Name"]
        df = df.drop(*columns_to_remove)
        
        df = self._convert_credit_history_to_months(df)
        
        numeric_cols = [field.name for field in df.schema.fields 
                    if field.dataType in [IntegerType(), DoubleType()]]
        categorical_cols = [field.name for field in df.schema.fields 
                        if field.dataType not in [IntegerType(), DoubleType()]]
        
        for col_name in numeric_cols:
            try:
                quantiles = df.approxQuantile(col_name, [0.5], 0.01)
                if quantiles and len(quantiles) > 0:
                    median = quantiles[0]
                    df = df.fillna(median, subset=[col_name])
                else:
                    df = df.fillna(0, subset=[col_name])
            except Exception as e:
                logger.warning(f"Error while processing numeric column {col_name}: {str(e)}")
                df = df.fillna(0, subset=[col_name])
        
        for col_name in categorical_cols:
            try:
                mode = df.groupBy(col_name).count().orderBy("count", ascending=False).first()
                if mode:
                    df = df.fillna(mode[0], subset=[col_name])
                else:
                    df = df.fillna("unknown", subset=[col_name])
            except Exception as e:
                logger.warning(f"Error while processing categorical column {col_name}: {str(e)}")
                df = df.fillna("unknown", subset=[col_name])
        
        # df = self._remove_outliers(df, numeric_cols)
        logger.info(f"Silver Dataframe number of rows: {df.count()}")
        
        return df
    
    def _process_gold_layer(self, silver_df: DataFrame) -> DataFrame:
        logger.info("Processing gold layer...")
        
        numeric_cols = [field.name for field in silver_df.schema.fields 
                       if isinstance(field.dataType, (IntegerType, DoubleType))]
        categorical_cols = [field.name for field in silver_df.schema.fields 
                          if isinstance(field.dataType, StringType)]
        
        if "Customer_ID" in categorical_cols:
            categorical_cols.remove("Customer_ID")
        
        stats_df = silver_df.groupBy("Customer_ID").agg(
            *[avg(col_name).alias(f"avg_{col_name}") for col_name in numeric_cols],
            *[stddev(col_name).alias(f"std_{col_name}") for col_name in numeric_cols]
        )
        
        for col_name in numeric_cols:
            try:
                percentiles = silver_df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
                if percentiles and len(percentiles) == 3:
                    stats_df = stats_df.withColumn(
                        f"{col_name}_q1", lit(percentiles[0])
                    ).withColumn(
                        f"{col_name}_median", lit(percentiles[1])
                    ).withColumn(
                        f"{col_name}_q3", lit(percentiles[2])
                    )
            except Exception as e:
                logger.warning(f"Could not calculate percentiles for {col_name}: {str(e)}")
        
        gold_df = silver_df.join(stats_df, on="Customer_ID", how="left")
        
        for col_name in categorical_cols:
            if not col_name or col_name.strip() == "":  # Skip empty column names
                logger.warning("Skipping empty column name in categorical encoding")
                continue
            
            try:
                indexer = StringIndexer(
                    inputCol=col_name,
                    outputCol=f"{col_name}_idx",
                    handleInvalid="keep"
                )
                gold_df = indexer.fit(gold_df).transform(gold_df)
                
                encoder = OneHotEncoder(
                    inputCol=f"{col_name}_idx",
                    outputCol=f"{col_name}_encoded",
                    dropLast=True  # to prevent perfect multicollinearity
                )
                gold_df = encoder.fit(gold_df).transform(gold_df)
                
                gold_df = gold_df.drop(f"{col_name}_idx")
                
            except Exception as e:
                logger.error(f"Error encoding categorical column {col_name}: {str(e)}")
                continue
        
        gold_df = gold_df.drop("Customer_ID")
        
        logger.info(f"Gold Dataframe number of rows: {gold_df.count()}")
        
        return gold_df
    
    def __call__(self, raw_data_path: str) -> None:
        logger.info("Running Extract")
        self.extract(raw_data_path)

        logger.info("Running Transform")
        silver_df, gold_df = self.transform()

        logger.info("Running Load")
        self.load(silver_df, gold_df)

def create_etl_pipeline(spark: SparkSession, root_dir: str, do_repartition: bool = False):
    etl = ETLPipeline(spark, root_dir)
    
    if do_repartition:
        etl.transform = repartition(spark)(etl.transform)
        etl._process_silver_layer = repartition(spark)(etl._process_silver_layer)
        etl._process_gold_layer = repartition(spark)(etl._process_gold_layer)

    return etl