import os

def create_directories():
    for dir_path in ['data/bronze', 'data/silver', 'data/gold']:
        os.makedirs(dir_path, exist_ok=True)

def load_raw_data(spark, input_path, bronze_path):
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.write.format("delta").mode("overwrite").save(bronze_path)