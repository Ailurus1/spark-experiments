import os
import shutil
from pathlib import Path

def create_directories(root: str) -> None:
    for dir_path in [Path(root) / 'data/bronze', Path(root) / 'data/silver', Path(root) / 'data/gold']:
        os.makedirs(dir_path, exist_ok=True)

def clean_directories(root: str) -> None:
    for dir_path in [Path(root) / 'data/bronze', Path(root) / 'data/silver', Path(root) / 'data/gold']:
        shutil.rmtree(dir_path)

def load_raw_data(spark, input_path, bronze_path):
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.write.format("delta").mode("overwrite").save(bronze_path)