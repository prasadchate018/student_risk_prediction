import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained scikit-learn LogisticRegression model from file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route("/")
def home():
    return jsonify(
        status="success",
        message="Logistic Regression Model API is running on Render!",
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="Model file could not be loaded on startup."), 500

    data = request.get_json(force=True)

    # Expected features based on your model metadata:
    # [attendance, study_hours, past_failures, assignments_completed_pct,
    #  parental_education, family_income, extracurricular, internet_access,
    #  previous_grade, final_score]
    try:
        features = data.get("features")
        if not features or len(features) != 10:
            return (
                jsonify(
                    error="Expected a 'features' array with exactly 10 numeric values."
                ),
                400,
            )

        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify(prediction=str(prediction))

    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
