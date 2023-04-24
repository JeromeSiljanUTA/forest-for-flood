"""
Calls all other methods to clean data, train model, and assess accuracy

"""
import os
import pyspark

from d01_data.to_parquet import clean_write_parquet, new

sc = pyspark.SparkContext("local[*]")
spark = pyspark.sql.SparkSession(sc)


# Create parquet if it doesn't yet exist
if not os.path.exists("../data/02_intermediate/nfip-flood-policies.parquet"):
    clean_write_parquet(spark)
