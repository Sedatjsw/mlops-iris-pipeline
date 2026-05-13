import os
import joblib
import numpy as np


def test_model_file_exists():
    assert os.path.exists("models/iris_model.pkl")


def test_target_names_file_exists():
    assert os.path.exists("models/target_names.pkl")


def test_model_prediction():
    model = joblib.load("models/iris_model.pkl")
    target_names = joblib.load("models/target_names.pkl")

    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction_index = model.predict(sample_input)[0]
    prediction = target_names[prediction_index]

    assert prediction in ["setosa", "versicolor", "virginica"]