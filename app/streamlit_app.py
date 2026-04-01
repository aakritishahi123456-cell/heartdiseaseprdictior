import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Constants
# -----------------------------
THRESHOLD = 0.4
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Title and description
# -----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    """
This app predicts whether a patient is at **high risk** or **low risk** of heart disease
using a trained **Logistic Regression** model.

**Model details**
- Final model: Logistic Regression
- Operational threshold: 0.4
- Output: predicted risk class + probability score

**Important**
This tool is for **educational and portfolio purposes only**.  
It is **not a medical diagnosis system** and must not be used for real clinical decisions.
"""
)

st.divider()

# -----------------------------
# Sidebar info
# -----------------------------
st.sidebar.header("About")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write(f"Threshold: {THRESHOLD}")
st.sidebar.write("Dataset: UCI Heart Disease")
st.sidebar.info(
    "Lower threshold improves recall, which is useful in screening-style medical tasks."
)

# -----------------------------
# User inputs
# -----------------------------
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=100, value=54)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        help="Encoded category from dataset"
    )
    trestbps = st.slider("Resting Blood Pressure (trestbps)", min_value=80, max_value=220, value=130)
    chol = st.slider("Cholesterol (chol)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    restecg = st.selectbox(
        "Resting ECG Results (restecg)",
        options=[0, 1, 2]
    )

with col2:
    thalach = st.slider("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", options=[0, 1, 2, 3])

# -----------------------------
# Build input dataframe
# -----------------------------
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
    "thal": thal
}])

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk", use_container_width=True):
    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= THRESHOLD)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Predicted Risk: HIGH")
    else:
        st.success("Predicted Risk: LOW")

    st.write(f"**Predicted probability of heart disease:** `{probability:.3f}`")
    st.write(f"**Decision threshold used:** `{THRESHOLD}`")

    # Confidence-style interpretation
    if probability >= 0.75:
        st.write("Interpretation: strong positive risk signal.")
    elif probability >= THRESHOLD:
        st.write("Interpretation: moderate positive risk signal.")
    else:
        st.write("Interpretation: below the selected operational threshold.")

    # Show raw input
    with st.expander("See input data"):
        st.dataframe(input_data, use_container_width=True)

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Educational ML project • Heart Disease Risk Prediction • Built with Streamlit and scikit-learn"
)