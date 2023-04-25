"""
Evaluate model against test set, provide metrics
"""

from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import DoubleType


def create_predictions(test, trained_model):
    predictions_df = trained_model.transform(test)
    return predictions_df


def calculate_metrics(predictions_df):
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

    return metrics_dict
