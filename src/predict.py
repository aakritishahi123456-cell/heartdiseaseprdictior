"""
predict.py
Loads the trained model and makes predictions with threshold support.
Can be used as a module or run standalone for quick CLI predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"
DEFAULT_THRESHOLD = 0.4


def load_model(path=None):
    """Load the trained model pipeline."""
    if path is None:
        path = MODEL_PATH
    return joblib.load(path)


def predict(model, input_data, threshold=DEFAULT_THRESHOLD):
    """
    Make a prediction with the trained model.

    Parameters:
        model: trained sklearn Pipeline
        input_data: pd.DataFrame with 13 clinical features
        threshold: classification threshold (default 0.4)

    Returns:
        dict with prediction, probability, and risk label
    """
    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)
    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "risk_label": risk_label,
        "threshold": threshold,
    }


def predict_from_values(
    model,
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal,
    threshold=DEFAULT_THRESHOLD
):
    """Make a prediction from individual feature values."""
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])
    return predict(model, input_data, threshold)


if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")

    # Example: typical high-risk patient
    result = predict_from_values(
        model,
        age=63, sex=1, cp=3, trestbps=145, chol=233,
        fbs=1, restecg=0, thalach=150, exang=0,
        oldpeak=2.3, slope=0, ca=0, thal=1,
    )

    print(f"\nPrediction:  {result['risk_label']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Threshold:   {result['threshold']}")
