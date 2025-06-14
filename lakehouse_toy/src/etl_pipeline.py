from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, regexp_extract, lit, avg, count, 
    stddev, percentile_approx, expr
)
from pyspark.sql.types import IntegerType, DoubleType
from typing import List, Tuple

class ETLPipeline:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.bronze_path = "data/bronze/credit_data"
        self.silver_path = "data/silver/credit_data"
        self.gold_path = "data/gold/credit_data"
    
    def extract(self, input_path: str) -> None:
        df = self.spark.read.csv(input_path, header=True, inferSchema=True)
        df.write.format("delta").mode("overwrite").save(self.bronze_path)
    
    def transform(self) -> Tuple[DataFrame, DataFrame]:
        silver_df = self._process_silver_layer()
        
        gold_df = self._process_gold_layer(silver_df)
        
        return silver_df, gold_df
    
    def load(self, silver_df: DataFrame, gold_df: DataFrame) -> None:
        silver_df.write.format("delta").mode("overwrite").save(self.silver_path)
        gold_df.write.format("delta").mode("overwrite").save(self.gold_path)
    
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
                abs((col(col_name) - mean) / std) < threshold
            )
        return df
    
    def _process_silver_layer(self) -> DataFrame:
        df = self.spark.read.format("delta").load(self.bronze_path)
        
        columns_to_remove = ["ID", "Customer_ID", "Name"]
        df = df.drop(*columns_to_remove)
        
        df = self._convert_credit_history_to_months(df)
        
        numeric_cols = [field.name for field in df.schema.fields 
                       if field.dataType in [IntegerType(), DoubleType()]]
        categorical_cols = [field.name for field in df.schema.fields 
                           if field.dataType not in [IntegerType(), DoubleType()]]
        
        for col_name in numeric_cols:
            median = df.approxQuantile(col_name, [0.5], 0.01)[0]
            df = df.fillna(median, subset=[col_name])
        
        for col_name in categorical_cols:
            mode = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
            df = df.fillna(mode, subset=[col_name])
        
        df = self._remove_outliers(df, numeric_cols)
        
        return df
    
    def _process_gold_layer(self, silver_df: DataFrame) -> DataFrame:
        aggregated_df = silver_df.groupBy("Credit_Score").agg(
            avg("Monthly_Inhand_Salary").alias("avg_monthly_salary"),
            avg("Num_Bank_Accounts").alias("avg_bank_accounts"),
            avg("Credit_History_Months").alias("avg_credit_history"),
            count("*").alias("customer_count")
        )
        
        numeric_cols = ["Monthly_Inhand_Salary", "Num_Bank_Accounts", "Credit_History_Months"]
        for col_name in numeric_cols:
            percentiles = silver_df.approxQuantile(col_name, [0.25, 0.5, 0.75], 0.01)
            aggregated_df = aggregated_df.withColumn(
                f"{col_name}_q1", lit(percentiles[0])
            ).withColumn(
                f"{col_name}_median", lit(percentiles[1])
            ).withColumn(
                f"{col_name}_q3", lit(percentiles[2])
            )
        
        return aggregated_df

    def __call__(self, raw_data_path: str) -> None:
        self.extract(raw_data_path)
        silver_df, gold_df = self.transform()
        self.load(silver_df, gold_df)