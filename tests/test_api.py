"""
This script tests the API calls for the welcome and the inference.
"""

import json
import sys

from fastapi.testclient import TestClient

from main import app

sys.path.append("..")


client = TestClient(app)


def test_welcome() -> None:
    """
    Tests greeting message on the home path of the API server.
    :return:
    """
    r = client.get("/")
    content = json.loads(r.content)
    assert r.status_code == 200
    assert "greeting" in content
    assert content["greeting"] == "Welcome to the census data ML API."


def test_inference_lower() -> None:
    """
    Tests a call to the inference API that should return a prediction of '<=50k'
    :return:
    """
    values = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    data = json.dumps(values)
    r = client.post("/infer", data=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_inference_upper() -> None:
    """
    Tests a call to the inference API that should return a prediction of '>50k'
    :return:
    """
    values = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    data = json.dumps(values)
    r = client.post("/infer", data=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"
