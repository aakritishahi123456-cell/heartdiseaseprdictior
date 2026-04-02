import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------- Page Config ----------
st.set_page_config(
    page_title="Heart Disease Risk Stratification",
    page_icon="❤️",
    layout="centered"
)

# ---------- Constants ----------
THRESHOLD = 0.4
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- Label Mappings ----------
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# ---------- Header ----------
st.title("❤️ Heart Disease Risk Stratification System")
st.markdown(
    """
This tool predicts whether a patient is at **high** or **low** risk of heart disease
using a trained **Logistic Regression** model.

**Important:** This is an educational machine learning project and **not** a clinical diagnosis tool.
"""
)

# ---------- Sidebar ----------
st.sidebar.header("Model Info")
st.sidebar.write("**Final Model:** Logistic Regression")
st.sidebar.write(f"**Threshold:** {THRESHOLD}")
st.sidebar.write("**Dataset:** UCI Heart Disease")
st.sidebar.info("Threshold 0.4 was selected to improve recall for screening-style use.")

# ---------- Patient Inputs ----------
st.subheader("Patient Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 54)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.slider("Resting Blood Pressure", 80, 220, 130)
    chol = st.slider("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Result", list(restecg_map.keys()))

with col2:
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox("ST Segment Slope", list(slope_map.keys()))
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Result", list(thal_map.keys()))

# ---------- Build Input DataFrame ----------
input_data = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp_map[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": restecg_map[restecg],
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": slope_map[slope],
    "ca": ca,
    "thal": thal_map[thal]
}])

# ---------- Prediction ----------
if st.button("🔍 Predict Risk", use_container_width=True):
    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= THRESHOLD)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Predicted Risk: **HIGH**")
    else:
        st.success("✅ Predicted Risk: **LOW**")

    st.metric("Predicted Probability", f"{probability:.2%}")
    st.progress(min(int(probability * 100), 100))

    if probability >= 0.75:
        st.write("**Interpretation:** Strong positive risk signal. Further clinical evaluation recommended.")
    elif probability >= THRESHOLD:
        st.write("**Interpretation:** Moderate positive risk signal. Consider follow-up testing.")
    else:
        st.write("**Interpretation:** Below the selected operational threshold. Low risk indicated.")

    with st.expander("📋 View Model Input Data"):
        st.dataframe(input_data, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Educational ML project • Not for real clinical decision-making • UCI Heart Disease Dataset")