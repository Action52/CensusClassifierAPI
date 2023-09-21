"""
This script trains the model and saves it.

Script originally taken from https://github.com/udacity/nd0821-c3-starter-code.
Modified by Luis Alfredo León
"""

# Script to train machine learning model.
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import compute_model_metrics, log_slice_performance, train_model

home_path = Path(__file__).cwd()

# Add code to load in the data.
data = pd.read_csv(f"{home_path}/data/clean_census.csv", index_col=0)

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


# Train and save a model.
model = train_model(X_train, y_train)
model_path = f"{home_path}/models/random_forest.joblib"
encoder_path = f"{home_path}/models/encoder.joblib"
lb_path = f"{home_path}/models/label_binarizer.joblib"
joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)

# Log the performance on a particular slice
log_slice_performance(
    model, test, encoder, lb, slice_by="race", cat_features=cat_features
)

# Print the test results
fbeta, precision, recall = compute_model_metrics(y_test, model.predict(X_test))
print(f"Test set metrics: {fbeta, precision, recall}")
