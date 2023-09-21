"""
This script runs a call to the deployed API.
"""

import json
import requests


def call_welcome_api(base_url: str) -> None:
    """
    Calls the base URL and prints the welcome message.
    """
    response = requests.get(f"{base_url}/")
    if response.status_code == 200:
        print("Welcome API Response:", response.json())
        print("Status code", response.status_code)

    else:
        print("Error in Welcome API:", response.status_code)


def call_inference_api(base_url: str, payload: dict) -> None:
    """
    Calls the /infer endpoint and prints the prediction result.
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{base_url}/infer", data=json.dumps(payload),
                             headers=headers)

    if response.status_code == 200:
        print("Inference API Response:", response.json())
        print("Status code", response.status_code)
    else:
        print("Error in Inference API:", response.status_code)


if __name__ == "__main__":
    base_url = "https://censusclassifierapi.onrender.com"

    # Sample payload for the /infer endpoint
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

    # Call the APIs
    call_welcome_api(base_url)
    call_inference_api(base_url, values)
