from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/iris_model.pkl")
joblib.dump(iris.target_names, "models/target_names.pkl")

print("Model trained successfully.")
print(f"Model accuracy: {accuracy:.2f}")
print("Model saved to models/iris_model.pkl")