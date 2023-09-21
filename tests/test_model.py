"""
This script tests the inference model and the data.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from ml.data import process_data
from ml.model import compute_model_metrics

home_path = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def cat_features() -> list:
    """
    Fixture to get the categorical features of the dataset.
    :return: list of categorical features
    """
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
def loaded_data(cat_features: list) -> pd.DataFrame:
    """
    Fixture to load the dataset as a pandas DataFrame.
    :param cat_features: list
    :return:
    """
    data = pd.read_csv(f"{home_path}/data/clean_census.csv", index_col=0)
    return data


@pytest.fixture(scope="module")
def loaded_model() -> RandomForestClassifier:
    """
    Loads the model.
    :return: The loaded model from .joblib file
    """
    return joblib.load(f"{home_path}/models/random_forest.joblib")


@pytest.fixture(scope="module")
def loaded_encoder() -> OneHotEncoder:
    """
    Loads the encoder
    :return: The loaded OneHotEncoder from .joblib file
    """
    return joblib.load(f"{home_path}/models/encoder.joblib")


@pytest.fixture(scope="module")
def loaded_binarizer() -> LabelBinarizer:
    """
    Loads the lb
    :return: The loaded LabelBinarizer from .joblib file
    """
    return joblib.load(f"{home_path}/models/label_binarizer.joblib")


def test_gender_slice(
    loaded_data: pd.DataFrame,
    loaded_model: RandomForestClassifier,
    cat_features: list,
    loaded_encoder: OneHotEncoder,
    loaded_binarizer: LabelBinarizer,
) -> None:
    """
    Tests a slice of the data and checks if the precision on both slices looks
    ok.
    :param loaded_data: pd.DataFrame with observations.
    :param loaded_model: The trained model.
    :param cat_features: List of categorical features.
    :param loaded_encoder: The loaded encoder for preprocessing.
    :param loaded_binarizer: The loaded lb for preprocessing.
    :return:
    """
    _, test = train_test_split(loaded_data, test_size=0.20)

    male_data = test[test["sex"] == "Male"]
    female_data = test[test["sex"] == "Female"]

    X_male, y_male, encoder_male, lb_male = process_data(
        male_data,
        cat_features,
        label="salary",
        training=False,
        encoder=loaded_encoder,
        lb=loaded_binarizer,
    )
    X_female, y_female, encoder_female, lb_female = process_data(
        female_data,
        cat_features,
        label="salary",
        training=False,
        encoder=loaded_encoder,
        lb=loaded_binarizer,
    )

    male_preds = loaded_model.predict(X_male)
    female_preds = loaded_model.predict(X_female)

    fbeta_male, precision_male, recall_male = compute_model_metrics(
        y_male, male_preds
    )
    fbeta_female, precision_female, recall_female = compute_model_metrics(
        y_female, female_preds
    )

    try:
        assert fbeta_male > 0.5
        assert fbeta_female > 0.5
    except AssertionError as e:
        raise AssertionError(
            "The data sliced by sex shows weird fbeta scores!"
        )


def test_dataframe_columns(
    loaded_data: pd.DataFrame, cat_features: list
) -> None:
    """
    Checks that all the categorical features exist in the DataFrame
    :param loaded_data: The pd.DataFrame with observations
    :param cat_features: list of categorical features
    :return:
    """
    for cat in cat_features:
        assert cat in loaded_data


def test_dataframe_target_col(
    loaded_data: pd.DataFrame, target_col: str = "salary"
) -> None:
    """
    Asserts that the target column exists in the training DataFrame
    :param loaded_data: The pd.DataFrame with observations
    :param target_col: The target column name
    :return:
    """
    assert target_col in loaded_data
