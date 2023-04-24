"""
Calls all other methods to clean data, train model, and assess accuracy

"""
import os
import pyspark

from d01_data.read_parquet import clean_parquet_df
from d02_intermediate.to_parquet import clean_write_parquet
from d03_processing.input_model import transform_info_vector
from d05_model_evaluatoin import create_predictions, calculate_metrics

TRAIN_MODEL = True

sc = pyspark.SparkContext("local[*]")
spark = pyspark.sql.SparkSession(sc)


# Create parquet if it doesn't yet exist
if not os.path.exists("../data/02_intermediate/nfip-flood-policies.parquet"):
    clean_write_parquet(spark)

# Read parquet if it exists
if os.path.exists("../data/02_intermediate/nfip-flood-policies.parquet"):
    df = clean_parquet_df(spark)

if TRAIN_MODEL:
    train_df, test_df, validation_df = transform_info_vector(spark, df)


df.show()
