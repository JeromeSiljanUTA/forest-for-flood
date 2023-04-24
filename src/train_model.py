import numpy as np
import pyspark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, IntegerType, LongType, DoubleType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib import RegressionMetrics
import pickle

try:
    sc = pyspark.SparkContext("local[*]")
    spark = pyspark.sql.SparkSession(sc)
except ValueError:
    pass

df = spark.read.parquet(
    "../data/02_intermediate/nfip-flood-policies.parquet", inferSchema=True
)

df = df.drop(*["policyeffectivedate", "policyterminationdate", "policytermindicator"])

df.printSchema()

df = (
    df.withColumn("censustract", col("censustract").cast(LongType()))
    .withColumn("crsdiscount", col("crsdiscount").cast(DoubleType()))
    .withColumn("elevationdifference", col("elevationdifference").cast(IntegerType()))
    .withColumn("federalpolicyfee", col("federalpolicyfee").cast(IntegerType()))
    .withColumn("hfiaasurcharge", col("hfiaasurcharge").cast(IntegerType()))
    .withColumn("latitude", col("latitude").cast(DoubleType()))
    .withColumn("longitude", col("longitude").cast(DoubleType()))
    .withColumn(
        "numberoffloorsininsuredbuilding",
        col("numberoffloorsininsuredbuilding").cast(IntegerType()),
    )
    .withColumn(
        "originalconstructiondate", col("originalconstructiondate").cast(DateType())
    )
    .withColumn("originalnbdate", col("originalnbdate").cast(DateType()))
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

assembler = VectorAssembler(
    inputCols=[
        "censustract",
        "crsdiscount",
        "elevationdifference",
        "federalpolicyfee",
        "hfiaasurcharge",
        "latitude",
        "longitude",
        "numberoffloorsininsuredbuilding",
        "policycost",
        "policycount",
    ],
    outputCol="features",
)
output = assembler.transform(df)
final_df = output.select(["features", "totalinsurancepremiumofthepolicy"])

train, test, validate = final_df.randomSplit([0.6, 0.2, 0.2], seed=0)

model = RandomForestRegressor(labelCol="totalinsurancepremiumofthepolicy")
model.setSeed(0)

print("Started fitting model")

trained_model = model.fit(train)

predictions = trained_model.transform(test)

val_pred_df = predictions.select(["totalinsurancepremiumofthepolicy", "prediction"])
val_pred_df = val_pred_df.withColumn(
    "totalinsurancepremiumofthepolicy",
    val_pred_df["totalinsurancepremiumofthepolicy"].cast(DoubleType()),
)

val_pred = val_pred_df.rdd.map(tuple)

metrics = RegressionMetrics(val_pred)

metrics_dict = {}

metrics_dict["r2"] = metrics.r2
metrics_dict["explainedVariance"] = metrics.explainedVariance
metrics_dict["meanAbsoluteError"] = metrics.meanAbsoluteError
metrics_dict["meanSquaredError"] = metrics.meanSquaredError
metrics_dict["rootMeanSquaredError"] = metrics.rootMeanSquaredError
