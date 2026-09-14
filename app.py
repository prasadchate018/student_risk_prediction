import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the logistic model from the current directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route("/")
def home():
    return jsonify(
        status="success", message="Logistic Regression Model API is running on Vercel!"
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Model file could not be loaded on startup."), 500

    try:
        data = request.get_json(force=True)

        # Allow passing key-value JSON or an ordered array of 10 values
        if "features" in data and isinstance(data["features"], list):
            feature_values = data["features"]
        else:
            # Match strict feature order from the pickle metadata
            feature_names = [
                "attendance",
                "study_hours",
                "past_failures",
                "assignments_completed_pct",
                "parental_education",
                "family_income",
                "extracurricular",
                "internet_access",
                "previous_grade",
                "final_score",
            ]
            feature_values = [data.get(name) for name in feature_names]

        if len(feature_values) != 10 or any(v is None for v in feature_values):
            return (
                jsonify(
                    error="Expected 10 numeric inputs matching features: attendance, study_hours, past_failures, assignments_completed_pct, parental_education, family_income, extracurricular, internet_access, previous_grade, final_score."
                ),
                400,
            )

        input_data = np.array(feature_values, dtype=float).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify(prediction=str(prediction))

    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
