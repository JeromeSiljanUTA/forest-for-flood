"""
Reads parquet file and cleans data
"""

import pyspark
from pyspark.sql.types import DateType, IntegerType, LongType, DoubleType
from pyspark.sql.functions import col


def clean_parquet_df(spark):
    """Reads parquet, drops unnecessary dolumns and casts column types.

    :param spark: Spark session from __init__.py
    :type spark: pyspark.SparkSession
    :returns: pyspark DataFrame of parquet data

    """

    df = spark.read.parquet("data/02_intermediate/nfip-flood-policies.parquet")

    # Cast types
    df = (
        df.withColumn("censustract", col("censustract").cast(LongType()))
        .withColumn("crsdiscount", col("crsdiscount").cast(DoubleType()))
        .withColumn(
            "elevationdifference", col("elevationdifference").cast(IntegerType())
        )
        .withColumn("federalpolicyfee", col("federalpolicyfee").cast(IntegerType()))
        .withColumn("hfiaasurcharge", col("hfiaasurcharge").cast(IntegerType()))
        .withColumn("latitude", col("latitude").cast(DoubleType()))
        .withColumn("longitude", col("longitude").cast(DoubleType()))
        .withColumn(
            "numberoffloorsininsuredbuilding",
            col("numberoffloorsininsuredbuilding").cast(IntegerType()),
        )
        .withColumn("policycost", col("policycost").cast(IntegerType()))
        .withColumn("policycount", col("policycount").cast(IntegerType()))
        .withColumn(
            "totalbuildinginsurancecoverage",
            col("totalbuildinginsurancecoverage").cast(IntegerType()),
        )
        .withColumn(
            "totalcontentsinsurancecoverage",
            col("totalcontentsinsurancecoverage").cast(IntegerType()),
        )
        .withColumn(
            "totalinsurancepremiumofthepolicy",
            col("totalinsurancepremiumofthepolicy").cast(IntegerType()),
        )
    )

    return df
