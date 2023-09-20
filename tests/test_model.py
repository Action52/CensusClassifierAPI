import pytest
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics

home_path = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture(scope="module")
def loaded_data(cat_features):
    data = pd.read_csv(f"{home_path}/data/clean_census.csv", index_col=0)
    return data


@pytest.fixture(scope="module")
def loaded_model():
    return joblib.load(f"{home_path}/models/random_forest.joblib")


@pytest.fixture(scope="module")
def loaded_encoder():
    return joblib.load(f"{home_path}/models/encoder.joblib")


@pytest.fixture(scope="module")
def loaded_binarizer():
    return joblib.load(f"{home_path}/models/label_binarizer.joblib")


def test_gender_slice(loaded_data, loaded_model: RandomForestClassifier, cat_features, loaded_encoder, loaded_binarizer):
    _, test = train_test_split(loaded_data, test_size=0.20)

    male_data = test[test['sex'] == 'Male']
    female_data = test[test['sex'] == 'Female']

    X_male, y_male, encoder_male, lb_male = process_data(male_data, cat_features, label="salary", training=False, encoder=loaded_encoder, lb=loaded_binarizer)
    X_female, y_female, encoder_female, lb_female = process_data(female_data, cat_features, label="salary", training=False, encoder=loaded_encoder, lb=loaded_binarizer)

    male_preds = loaded_model.predict(X_male)
    female_preds = loaded_model.predict(X_female)

    fbeta_male, precision_male, recall_male = compute_model_metrics(y_male, male_preds)
    fbeta_female, precision_female, recall_female = compute_model_metrics(y_female, female_preds)

    try:
        assert fbeta_male > 0.5
        assert fbeta_female > 0.5
    except AssertionError as e:
        raise AssertionError("The data sliced by sex shows weird fbeta scores!")


def test_dataframe_columns(loaded_data, cat_features):
    for cat in cat_features:
        assert cat in loaded_data


def test_dataframe_target_col(loaded_data, target_col='salary'):
    assert target_col in loaded_data