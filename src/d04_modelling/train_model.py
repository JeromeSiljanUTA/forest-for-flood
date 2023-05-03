"""
Train model and save to file
"""

import logging

from pyspark.ml.regression import RandomForestRegressor


def create_model(depth):
    """Creates model, sets label column

    :returns: Returns model object

    """
    model = RandomForestRegressor(
        labelCol="totalinsurancepremiumofthepolicy", maxDepth=depth
    )
    model.setSeed(0)
    return model


def fit_model(train_df, depth, save=False):
    """Fits model to training data

    :param train_df: Training data
    :type train_df: DataFrame
    :returns: RandomForestRegressor object of trained model

    """
    model = create_model(depth)
    logging.info("Started fitting model")
    trained_model = model.fit(train_df)

    model_name = f"data/04_models/model_depth_{depth}"

    if save:
        trained_model.save(model_name)
        logging.info(f"Successfully trained model with depth {depth}")

    return trained_model
