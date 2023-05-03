"""
Reads data from csv, cleans and outputs as parquet
"""

import pyspark


def clean_write_parquet(spark):
    """
    Creates spark session, drops appropriate columns, and writes data to parquet file

    """

    df = spark.read.csv(
        # Go back out of d01_data, then src
        "data/01_raw/NFIP/nfip-flood-policies.csv",
        header=True,
        inferSchema=True,
    )

    # Drop columns that have a lot of null/date values, then all rows with null values
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
            "originalconstructiondate",
            "originalnbdate",
            "policyeffectivedate",
            "policyterminationdate",
            "policytermindicator",
            "smallbusinessindicatorbuilding",
        ]
    ).dropna()

    df.write.save("data/02_intermediate/nfip-flood-policies.parquet", mode="overwrite")
