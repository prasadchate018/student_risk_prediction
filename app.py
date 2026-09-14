import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained scikit-learn model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'logistic.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML Template with inline CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Status Predictor</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #f4f7f6; margin: 0; padding: 40px 20px; color: #333; }
        .container { max-width: 650px; background: #ffffff; margin: 0 auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 25px; }
        .form-group { margin-bottom: 18px; }
        label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 14px; }
        input[type="number"], select { width: 100%; padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; transition: border 0.3s; }
        input[type="number"]:focus, select:focus { border-color: #3498db; outline: none; }
        button { width: 100%; background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.3s; margin-top: 10px; }
        button:hover { background-color: #2980b9; }
        .result-card { margin-top: 25px; padding: 15px; border-radius: 6px; text-align: center; font-weight: bold; font-size: 18px; }
        .At-Risk { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .High-Risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .Safe { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Level Predictor</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label>Attendance (%)</label>
                <input type="number" step="any" name="attendance" required placeholder="0 - 100">
            </div>
            <div class="form-group">
                <label>Study Hours (per week)</label>
                <input type="number" step="any" name="study_hours" required placeholder="e.g. 15">
            </div>
            <div class="form-group">
                <label>Past Failures</label>
                <input type="number" name="past_failures" required placeholder="e.g. 0">
            </div>
            <div class="form-group">
                <label>Assignments Completed (%)</label>
                <input type="number" step="any" name="assignments_completed_pct" required placeholder="0 - 100">
            </div>
            <div class="form-group">
                <label>Parental Education</label>
                <select name="parental_education" required>
                    <option value="0">High School</option>
                    <option value="1">Associate Degree</option>
                    <option value="2">Bachelor's Degree</option>
                    <option value="3">Master's or Higher</option>
                </select>
            </div>
            <div class="form-group">
                <label>Family Income Level</label>
                <select name="family_income" required>
                    <option value="0">Low</option>
                    <option value="1">Medium</option>
                    <option value="2">High</option>
                </select>
            </div>
            <div class="form-group">
                <label>Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Internet Access</label>
                <select name="internet_access" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Previous Grade</label>
                <input type="number" step="any" name="previous_grade" required placeholder="0 - 100">
            </div>
            <div class="form-group">
                <label>Final Score</label>
                <input type="number" step="any" name="final_score" required placeholder="0 - 100">
            </div>
            <button type="submit">Predict Status</button>
        </form>

        {% if prediction %}
        <div class="result-card {{ prediction }}">
            Predicted Student Status: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features matching model order
        features = [
            float(request.form['attendance']),
            float(request.form['study_hours']),
            float(request.form['past_failures']),
            float(request.form['assignments_completed_pct']),
            float(request.form['parental_education']),
            float(request.form['family_income']),
            float(request.form['extracurricular']),
            float(request.form['internet_access']),
            float(request.form['previous_grade']),
            float(request.form['final_score'])
        ]
        
        # Reshape for model input
        input_data = np.array([features])
        prediction_index = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction_index)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

# Vercel Serverless Function entry point
app_instance = app

if __name__ == '__main__':
    app.run(debug=True)
