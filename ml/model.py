"""
This script implements methods to train and measure the model on the census
data.

Script originally taken from https://github.com/udacity/nd0821-c3-starter-code.
Modified by Luis Alfredo León
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from ml.data import process_data


def train_model(
    X_train: np.array, y_train: np.array
) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.array, preds: np.array) -> Tuple:
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.array) -> np.array:
    """
    Run model inferences and return the predictions.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def compute_performance_on_slice(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    slice_by: str = "sex",
    cat_features: list = [],
) -> Tuple:
    """
    Yields the metrics of some sliced version of the dataset.
    :param model: Trained model to compute metrics on.
    :param df: pd.DataFrame with observations
    :param encoder: OneHotEncoder for preprocessing
    :param lb: LabelBinarizer for preprocessing
    :param slice_by: Category to slice by.
    :param cat_features: Categorical features list.
    :return:
    """
    values = df[slice_by].unique()
    for value in values:
        X, y, encoder, lb = process_data(
            df[df[slice_by] == value],
            cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds_value = model.predict(X)
        precision, recall, fbeta = compute_model_metrics(
            y=y, preds=preds_value
        )
        yield value, precision, recall, fbeta


def log_slice_performance(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    slice_by: str = "sex",
    output_path: str = "slice_output.txt",
    cat_features: list = [],
) -> None:
    """
    Logs the slice performance into a file.
    :param model: Trained model to compute metrics on.
    :param df: pd.DataFrame with observations
    :param encoder: OneHotEncoder for preprocessing
    :param lb: LabelBinarizer for preprocessing
    :param slice_by: Category to slice by.
    :param output_path: Path to output the results
    :param cat_features: Categorical features list.
    :return:
    """
    with open(output_path, "w") as output:
        output.write("value,\tprecision,\trecall,\tfbeta\n")
        for value, precision, recall, fbeta in compute_performance_on_slice(
            model, df, encoder, lb, slice_by, cat_features
        ):
            output.write(f"{value},\t{precision},\t{recall},\t{fbeta}\n")
