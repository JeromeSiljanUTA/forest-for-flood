import numpy as np
import pyspark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col
from pyspark.sql.types import DateType, IntegerType, LongType, DoubleType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pickle

sc = pyspark.SparkContext("local[*]")
spark = pyspark.sql.SparkSession(sc)


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

model.fit(train)

with open('pickled_model', 'wb') as file:
    pickle.dump(model, file,pickle.HIGHEST_PROTOCOL)

predictions = model.transform(test)

test.write.save("test.csv", mode="overwrite")
predictions.write.save("predictions.csv", mode="overwrite")
