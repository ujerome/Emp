from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load model and encoder
model = joblib.load('saved_models/best_xgboost_pipeline_optimized.pkl')
label_encoder = joblib.load('saved_models/label_encoder.pkl')

# Input feature columns (must match model training)
input_columns = ["Educational_level", "Field_of_education", "TVT2", "Sex", "Age_group"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form  # use .get for safety
        input_data = pd.DataFrame([{
            "Educational_level": data.get("education"),
            "Field_of_education": data.get("field"),
            "TVT2": data.get("tvt"),
            "Sex": data.get("sex"),
            "Age_group": data.get("age_group")
        }])

        prediction_encoded = model.predict(input_data)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
