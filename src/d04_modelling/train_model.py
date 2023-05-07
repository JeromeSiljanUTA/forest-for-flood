"""
Train model and save to file
"""

import logging
from pyspark.ml.regression import RandomForestRegressor


def create_model(depth, num_trees):
    """Creates model, sets label column

    :param depth: Set maximum depth of trees in forest
    :type depth: int
    :param num_trees: Set number of trees in forest
    :type num_trees: int
    :returns: model object

    """
    model = RandomForestRegressor(
        labelCol="totalinsurancepremiumofthepolicy", maxDepth=depth, numTrees=num_trees
    )
    model.setSeed(0)

    return model


def fit_model(train_df, depth, num_trees, save=False):
    """Fits model to training data

    :param train_df: Training data
    :type train_df: DataFrame
    :param depth: Set maximum depth of trees in forest
    :type depth: int
    :param num_trees: Set number of trees in forest
    :type num_trees: int
    :param save: Set to save model to file if True
    :type save: bool
    :returns: RandomForestRegressor object of trained model

    """
    model = create_model(depth, num_trees)
    logging.info(f"Started fitting model {depth}, {num_trees}")
    trained_model = model.fit(train_df)

    if save:
        model_name = f"data/04_models/model_{depth}_{num_trees}"

        try:
            trained_model.save(model_name)
            logging.info(
                f"Successfully trained model with depth {depth} and {num_trees} trees"
            )
        except Py4JJavaError:
            logging.warning(
                f"Unable to save trained model with depth {depth} and {num_trees} trees\nCheck if existing model is present"
                )
    return trained_model
