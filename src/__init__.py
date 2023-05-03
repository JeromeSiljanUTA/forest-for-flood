"""
Calls all other methods to clean data, train model, and assess accuracy

"""
import logging
import os
import pickle
import pyspark

from d01_data.read_parquet import clean_parquet_df
from d02_intermediate.to_parquet import clean_write_parquet
from d03_processing.input_model import transform_into_vector
from d04_modelling.train_model import fit_model
from d05_model_evaluation.metrics import create_predictions, calculate_metrics

logging.basicConfig(level=logging.INFO)

# If not called from toplevel of project,
if os.getcwd().split("/")[-1] != "forest-for-flood":
    os.chdir("..")
    if os.getcwd().split("/")[-1] != "forest-for-flood":
        raise ValueError(
            "Make sure to start program from toplevel (eg. python src/__init__.py)"
        )

TRAIN_MODEL = True
LOAD_MODEL = False

spark = (
    pyspark.sql.SparkSession.builder.master("local[*]")
    .config("spark.driver.memory", "10g")
    .getOrCreate()
)

sc = spark.sparkContext

# Create parquet if it doesn't yet exist
if not os.path.exists("data/02_intermediate/nfip-flood-policies.parquet"):
    print("Creating parquet file")
    clean_write_parquet(spark)

# Read parquet if it exists
if os.path.exists("data/02_intermediate/nfip-flood-policies.parquet"):
    df = clean_parquet_df(spark)

if TRAIN_MODEL:
    train_df, test_df, validation_df = transform_into_vector(spark, df)
    depths = range(13, 20)
    for depth in depths:
        trained_model = fit_model(train_df, depth, save=True)


if LOAD_MODEL:
    continue_load = True

    train_df, test_df, validation_df = transform_into_vector(spark, df)
    while continue_load:
        print("Which model would you like to load?")
        for model in os.listdir("data/04_models"):
            print(model)

        print(" > ")
        model_name = input()

        if model_name == "quit":
            continue_load = False
        else:
            trained_model = pyspark.ml.regression.RandomForestRegressionModel.load(
                f"data/04_models/{model_name}"
            )

            predictions_df = create_predictions(test_df, trained_model)
            metrics_dict = calculate_metrics(predictions_df)
            print(f"{model_name} metrics:")
            print(metrics_dict)
