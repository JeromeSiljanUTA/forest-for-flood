import pandas as pd
import numpy as np
import pyspark

sc = pyspark.SparkContext("local[*]")
spark = pyspark.sql.SparkSession(sc)

df = spark.read.csv(
    "../data/01_raw/NFIP/nfip-flood-policies.csv", header=True, inferSchema=True
)

df.printSchema()


df = df.drop(
    *[
        "basefloodelevation",
        "cancellationdateoffloodpolicy",
        "deductibleamountincontentscoverage",
        "elevationcertificateindicator",
        "houseofworshipindicator",
        "locationofcontents",
        "lowestadjacentgrade",
        "lowestfloorelevation",
        "nonprofitindicator",
        "obstructiontype",
        "smallbusinessindicatorbuilding",
    ]
).dropna()

df.write.save("../data/02_intermediate/nfip-flood-policies.parquet", mode="overwrite")
