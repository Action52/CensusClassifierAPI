"""
This script implements methods to interact with the census dataset like
preprocessing the data.

Script originally taken from https://github.com/udacity/nd0821-c3-starter-code.
Modified by Luis Alfredo León
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X: pd.DataFrame,
    categorical_features: list = [],
    label: str = None,
    training: bool = True,
    encoder: OneHotEncoder = None,
    lb: LabelBinarizer = None,
) -> Tuple:
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.
    :param X: pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`.
    :param categorical_features: List containing the names of the categorical
    features (default=[]).
    :param label: Name of the label column in `X`. If None, then an empty array
    will be returned for y (default=None)
    :param training: Indicator if training mode or inference/validation mode.
    :param encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    :param lb: Trained sklearn LabelBinarizer, only used if training=False.
    :return: Tuple of:
            X : np.array
                Processed data.
            y : np.array
                Processed labels if labeled=True, otherwise empty np.array.
            encoder : sklearn.preprocessing._encoders.OneHotEncoder
                Trained OneHotEncoder if training is True, otherwise returns the
                encoder passed in.
            lb : sklearn.preprocessing._label.LabelBinarizer
                Trained LabelBinarizer if training is True, otherwise returns
                the binarizer passed in.
    """

    if categorical_features is None:
        categorical_features = []
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
