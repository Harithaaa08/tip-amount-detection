"""
Flask API for Tip Prediction

- Loads trained Random Forest model
- Exposes /predict endpoint
- Handles errors gracefully
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------
# Load the trained model
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")  # make sure this matches your saved file

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}. Train and save your model first.")
    raise

# ---------------------------
# Define /predict endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from Streamlit
        data = request.get_json()

        # Validate required fields
        required_fields = ["total_bill", "sex", "smoker", "day", "time", "size"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Convert to DataFrame (column names must match training)
        input_df = pd.DataFrame([{
            "total_bill": float(data["total_bill"]),
            "sex": data["sex"],
            "smoker": data["smoker"],
            "day": data["day"],
            "time": data["time"],
            "size": int(data["size"])
        }])

        # Make prediction
        prediction = model.predict(input_df)

        return jsonify({"predicted_tip": float(prediction[0])})

    except Exception as e:
        # Return error as JSON
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
