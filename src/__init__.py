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
    # Set to 90G free for server
    .config("spark.driver.memory", "90g").getOrCreate()
)

sc = spark.sparkContext

# Create parquet if it doesn't yet exist
if not os.path.exists("data/02_intermediate/nfip-flood-policies.parquet"):
    # Create directory of it doesn't yet exist
    if not os.path.exists("data/02_intermediate/"):
        os.mkdir("data/02_intermediate/")

    logging.info("Creating parquet file")
    clean_write_parquet(spark)

# Load data into dataframe
df = clean_parquet_df(spark)

# Max depth of RandomForestRegressor goes to 30
depths = range(5, 31)
trees = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

metric_arr = []

if TRAIN_MODEL:
    logging.info("Starting to train models")
    train_df, test_df, validation_df = transform_into_vector(spark, df)
    for num_trees in trees:
        for depth in depths:
            # Check if model has been trained; helps if only few models were trained
            # because the program crashed
            if not os.path.exists(f"data/04_models/model_{depth}_{num_trees}"):
                trained_model = fit_model(train_df, depth, num_trees, save=True)
                predictions_df = create_predictions(test_df, trained_model)
                metrics_dict = calculate_metrics(predictions_df)
                metrics_dict["model"] = f"{depth}, {num_trees}"
                metric_arr.append(metrics_dict)
            else:
                logging.info(f"model_{depth}_{num_trees} exists, not training")

    file_path = "data/06_reporting/metrics.pkl"
    logging.info(f"Writing metrics to {file_path}")
    with open(file_path, "ab") as f:
        pickle.dump(metric_arr, f)


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
