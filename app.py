import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

artifact = joblib.load("models/california_housing_model.pkl")
pipeline = artifact["pipeline"]
metrics = artifact["metrics"]
features = artifact["features"]

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "California Housing Price Predictor",
        "status": "running",
        "endpoints": ["/", "/health", "/predict", "/features"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "metrics": metrics
    })

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({
        "features": [{"name": feature, "type": "float"} for feature in features]
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Request body debe ser JSON"}), 400

        missing = [feature for feature in features if feature not in data]
        if missing:
            return jsonify({
                "error": "Faltan campos requeridos",
                "missing_fields": missing
            }), 400

        row = {}
        for feature in features:
            value = data[feature]
            try:
                row[feature] = float(value)
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"El campo '{feature}' debe ser numérico"
                }), 400

        X_new = pd.DataFrame([row])
        pred_log = pipeline.predict(X_new)[0]
        pred_original = float(np.expm1(pred_log))

        return jsonify({
            "predicted_value": pred_original
        })

    except Exception as e:
        return jsonify({
            "error": "Error interno del servidor",
            "detail": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)