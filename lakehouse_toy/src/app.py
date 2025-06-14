from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

from utils import create_directories
from etl_pipeline import create_etl_pipeline

def create_spark_session():
    spark = SparkSession.builder.appName("CreditScoreETL")\
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    return configure_spark_with_delta_pip(spark).getOrCreate()

def main():
    spark = create_spark_session()
    
    create_directories('.')
    
    input_path = "data/data.csv"
    
    etl = create_etl_pipeline(spark)

    etl(input_path)

    print("ETL pipeline completed successfully!")

if __name__ == "__main__":
    main()