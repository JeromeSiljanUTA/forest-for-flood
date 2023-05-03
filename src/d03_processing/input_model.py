"""
Converts Dataframe into vectors for model, splits data into training, test, validation sets
"""

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


def transform_into_vector(spark, df):
    """Transforms features from DataFrame into vector column, then returns train, test, validation split.

    :param spark: Spark session from __init__.py
    :type spark: pyspark.SparkSession
    :param df: DataFrame of cleaned data
    :type df: DataFrame
    :returns: test, train, validation DataFrames (0.6/0.2/0.2 split)

    """
    assembler = VectorAssembler(
        # Exclude string columns
        inputCols=[
            "censustract",
            # "crsdiscount", lots of 0 values
            "elevationdifference",
            "federalpolicyfee",
            # "hfiaasurcharge", lots of 0 values
            "latitude",
            "longitude",
            "numberoffloorsininsuredbuilding",
            # "policycost", this is calculated by FEMA
            # "policycount", all 1
        ],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_df = output.select(["features", "totalinsurancepremiumofthepolicy"])

    return final_df.randomSplit([0.6, 0.2, 0.2], seed=0)
