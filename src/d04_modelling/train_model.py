"""
Train model and save to file
"""

import pickle
import logging

from pyspark.ml.regression import RandomForestRegressor


def create_model():
    """Creates model, sets label column

    :returns: Returns model object

    """
    model = RandomForestRegressor(labelCol="totalinsurancepremiumofthepolicy")
    model.setSeed(0)
    return model


def fit_model(model, train_df):
    """Fits model to training data

    :param model: Model for training
    :type model: RandomForestRegressor
    :param train_df: Training data
    :type train_df: DataFrame
    :returns: RandomForestRegressor object of trained model

    """
    create_model()
    logging.info("Started fitting model")
    trained_model = model.fit(train_df)
    return trained_model
