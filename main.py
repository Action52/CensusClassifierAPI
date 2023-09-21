"""
This script implements an API with FastAPI to be able to perform inference using
our saved model.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml.model import process_data

app = FastAPI()

home_path = Path(__file__).cwd()
model_path = f"{home_path}/models/random_forest.joblib"
lb_path = f"{home_path}/models/label_binarizer.joblib"
encoder_path = f"{home_path}/models/encoder.joblib"

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


class CensusRow(BaseModel):
    """
    Pydantic wrapper for an observation of the census.
    """

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def hello() -> dict:
    """
    Simple get method to acknowledge the landing url of the api.
    :return: dict with greeting.
    """
    return {"greeting": "Welcome to the census data ML API."}


@app.post("/infer")
async def inference(body: CensusRow) -> dict:
    """
    Post method to perform inference on an observation.
    :param body: CensusRow with the columns of an observation X.
    :return: dict with the prediction and with the original observation (body)
    """
    with open(model_path, "rb") as model_file:
        model = joblib.load(model_file)
    with open(lb_path, "rb") as lb_file:
        lb = joblib.load(lb_file)
    with open(encoder_path, "rb") as encoder_file:
        encoder = joblib.load(encoder_file)
    X = pd.DataFrame(data=dict(body), index=[0])
    X, y, encoder, lb = process_data(
        X,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    y_pred = model.predict(X)
    y_pred_label = lb.inverse_transform(y_pred)
    return {"prediction": y_pred_label[0], "body": body}
