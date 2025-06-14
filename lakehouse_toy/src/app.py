from pyspark.sql import SparkSession

from utils import create_directories, load_raw_data
from etl_pipeline import ETLPipeline

def create_spark_session():
    return (SparkSession.builder
            .appName("CreditScoreETL")
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())

def main():
    spark = create_spark_session()
    
    create_directories()
    
    input_path = "data/data.csv"
    
    etl = ETLPipeline(spark)

    etl(input_path)

    print("ETL pipeline completed successfully!")

if __name__ == "__main__":
    main()