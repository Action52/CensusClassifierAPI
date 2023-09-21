# CensusClassifierAPI 📊

## Description

The CensusClassifier API predicts whether a person will earn a salary of >=50k per year based on their demographics. This project uses census data to train the underlying model and is served using FastAPI.
Github link: https://github.com/Action52/CensusClassifierAPI.git
Deployed API endpoint: https://censusclassifierapi.onrender.com/
## Installation 🛠️

1. Clone the repository
    ```bash
    git clone https://github.com/Action52/CensusClassifierAPI.git
    ```
2. Install the requirements into an environment
    ```bash
    pip install -r requirements.txt
    ## or
    conda env create -f environment.yml
    ```

## Usage 🚀

1. Navigate to the project folder
    ```bash
    cd CensusClassifier
    ```

2. Run the API
    ```bash
    uvicorn main:app --reload
    ```

Now the API is running at `http://127.0.0.1:8000/`. Use the `/infer` endpoint to make your predictions. Check the docs for more specific usage.

## Disclaimer ⚠️

This repository includes and modifies code from the starter kit repo provided in the Udacity Machine Learning DevOps Nanodegree program.

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
