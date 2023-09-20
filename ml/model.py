from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
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


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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


def inference(model: RandomForestClassifier, X):
    """ Run model inferences and return the predictions.

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


def compute_performance_on_slice(model, df, encoder, lb, slice_by="sex", cat_features=[]):
    values = df[slice_by].unique()
    for value in values:
        X, y, encoder, lb = process_data(df[df[slice_by] == value], cat_features, label="salary",
                                         training=False, encoder=encoder, lb=lb)
        preds_value = model.predict(X)
        precision, recall, fbeta = compute_model_metrics(y=y, preds=preds_value)
        yield value, precision, recall, fbeta


def log_slice_performance(model, df, encoder, lb, slice_by="sex", output_path="slice_output.txt", cat_features=[]):
    with open(output_path, "w") as output:
        output.write("value,\tprecision,\trecall,\tfbeta\n")
        for value, precision, recall, fbeta in compute_performance_on_slice(model, df, encoder, lb, slice_by, cat_features):
            output.write(f"{value},\t{precision},\t{recall},\t{fbeta}\n")