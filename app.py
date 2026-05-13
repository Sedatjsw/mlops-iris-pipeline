from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/iris_model.pkl")
target_names = joblib.load("models/target_names.pkl")

descriptions = {
    "setosa": "Setosa is usually a small iris flower with short petals.",
    "versicolor": "Versicolor is a medium-sized iris flower with balanced petal and sepal measurements.",
    "virginica": "Virginica is usually a larger iris flower with longer petals."
}

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Prediction</title>
</head>
<body>
    <h1>Iris Flower Prediction App</h1>

    <p>
        This application uses a machine learning model to predict the Iris flower species
        based on four flower measurements.
    </p>

    <h3>Please enter realistic Iris flower measurements:</h3>
    <ul>
        <li>Sepal Length: 4.0 - 8.0</li>
        <li>Sepal Width: 2.0 - 4.5</li>
        <li>Petal Length: 1.0 - 7.0</li>
        <li>Petal Width: 0.1 - 2.5</li>
    </ul>

    <p><strong>Example:</strong> 5.1, 3.5, 1.4, 0.2</p>

    <form method="POST" action="/predict">
        <label>Sepal Length:</label><br>
        <input type="number" step="0.1" min="4.0" max="8.0" name="sepal_length" required><br><br>

        <label>Sepal Width:</label><br>
        <input type="number" step="0.1" min="2.0" max="4.5" name="sepal_width" required><br><br>

        <label>Petal Length:</label><br>
        <input type="number" step="0.1" min="1.0" max="7.0" name="petal_length" required><br><br>

        <label>Petal Width:</label><br>
        <input type="number" step="0.1" min="0.1" max="2.5" name="petal_width" required><br><br>

        <button type="submit">Predict Flower Species</button>
    </form>

    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
        <p>{{ description }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction_index = model.predict(input_data)[0]
    prediction = target_names[prediction_index]
    description = descriptions[prediction]

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        description=description
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)