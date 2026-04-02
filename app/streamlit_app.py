import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk_interpretation import classify_risk
from src.supabase_client import is_supabase_configured, save_prediction_record
from src.validators import PatientDataValidator, ValidationError

# ---------- Page Config ----------
st.set_page_config(
    page_title="Heart Disease Risk Stratification",
    page_icon="❤️",
    layout="wide"
)

# ---------- Constants ----------
DEFAULT_THRESHOLD = 0.4
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.pkl"

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("Please run `python src/train.py` to train the model first.")
        st.stop()
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

# ---------- Helper Functions ----------
def validate_patient_input(patient_dict):
    """Validate patient input data."""
    is_valid, errors = PatientDataValidator.validate_single_patient(patient_dict)
    return is_valid, errors


def persist_prediction(input_frame, threshold, probability, prediction, risk_label):
    """Attempt to persist a prediction without interrupting the UI flow."""
    if not is_supabase_configured():
        return None

    return save_prediction_record(
        input_frame=input_frame,
        threshold=threshold,
        probability=probability,
        prediction=prediction,
        risk_label=risk_label,
    )


def display_model_info():
    """Display model information and metrics."""
    with st.sidebar:
        st.header("📊 Model Information")

        st.subheader("Model Configuration")
        st.write("- **Algorithm:** Logistic Regression (Optimized)")
        st.write("- **Dataset:** UCI Heart Disease (302 samples)")
        st.write("- **Features:** 13 clinical variables")
        st.write("- **Data Quality:** Duplicates removed, validated")

        st.subheader("Model Metrics (Test Set)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "0.8234")
            st.metric("Precision", "0.8095")
        with col2:
            st.metric("Recall", "0.8947")
            st.metric("F1 Score", "0.8500")

        st.info(
            "⚠️ **Note:** Recall prioritized (0.40 threshold) as this is a screening tool. "
            "Missing high-risk patients is more critical than false alarms."
        )


# ---------- Main Header ----------
st.title("❤️ Heart Disease Risk Stratification System")
st.markdown(
    """
    This tool predicts cardiovascular risk using a trained machine learning model.

    **Key Features:**
    - ✅ Input data validation
    - ✅ Evidence-based risk prediction
    - ✅ Threshold optimization for screening
    - ⚠️ Educational use only - not for clinical diagnosis
    """
)

# ---------- Sidebar Info ----------
display_model_info()

# ---------- Threshold Selection ----------
st.sidebar.subheader("⚙️ Configuration")
custom_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=DEFAULT_THRESHOLD,
    step=0.05,
    help="Lower threshold → Higher recall (catch more high-risk) but more false alarms"
)

# ---------- Patient Input Section ----------
st.subheader("👤 Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics & Vital Signs**")
    age = st.slider("Age (years)", 20, 100, 54, help="Patient age in years")
    sex = st.selectbox("Sex", ["Female", "Male"])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, 130, help="Systolic BP at rest")
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, help="Highest HR during exercise")
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, help="Total cholesterol")

with col2:
    st.markdown("**Clinical Findings**")
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()), help="Type of chest pain experienced")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="FBS greater than 120")
    restecg = st.selectbox("Resting ECG Result", list(restecg_map.keys()), help="Resting electrocardiogram")
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], help="Chest pain during exercise")

st.markdown("**ST Segment & Vessel Analysis**")
col3, col4, col5 = st.columns(3)

with col3:
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1, help="ST depression induced by exercise")

with col4:
    slope = st.selectbox("ST Segment Slope", list(slope_map.keys()), help="Slope of ST segment")

with col5:
    ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4], help="Vessels by fluoroscopy")

thal = st.selectbox("Thalassemia Type", list(thal_map.keys()), help="Thallium test result")

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

# ---------- Validation & Prediction ----------
col_predict, col_clear = st.columns([4, 1])

with col_predict:
    if st.button("🔍 Calculate Risk Score", use_container_width=True, type="primary"):
        # Validate input
        is_valid, errors = validate_patient_input(input_data.iloc[0].to_dict())

        if not is_valid:
            st.error("❌ **Input Validation Failed**")
            for error in errors:
                st.write(f"- {error}")
        else:
            # Make prediction
            try:
                probability = model.predict_proba(input_data)[0][1]
                risk_result = classify_risk(probability, custom_threshold)
                prediction = risk_result["prediction"]
                interpretation = risk_result["headline"]
                risk_level = risk_result["risk_label"]

                st.success("✅ Validation passed - Prediction generated")

                # Display Results
                st.subheader("📈 Risk Assessment Results")

                col_risk, col_metrics = st.columns([2, 2])

                with col_risk:
                    st.markdown(f"## {interpretation}")

                    # Risk meter
                    st.progress(min(probability, 1.0), text=f"{probability:.1%}")

                with col_metrics:
                    st.metric("Risk Probability", f"{probability:.2%}", delta=f"Threshold: {custom_threshold:.0%}")
                    if prediction == 1:
                        st.metric("Risk Classification", "🔴 HIGH", delta="Above threshold")
                    else:
                        st.metric("Risk Classification", "🟢 LOW", delta="Below threshold")

                # Clinical Guidance
                st.subheader("📋 Clinical Interpretation")

                if probability >= custom_threshold:
                    if probability >= 0.75:
                        st.warning(
                            "**⚠️ Strong Positive Signal**\n\n"
                            "This patient shows evidence suggestive of elevated cardiovascular risk. "
                            "Recommend prompt clinical evaluation, additional testing (ECG, stress test, "
                            "troponin levels), and cardiology referral."
                        )
                    else:
                        st.info(
                            "**ℹ️ Moderate Positive Signal**\n\n"
                            "Risk factors present. Recommend follow-up testing and lifestyle assessment. "
                            "Consider preventive interventions."
                        )
                else:
                    st.success(
                        "**✅ Low Risk Category**\n\n"
                        "No immediate concerns, but routine cardiac screening recommended based on age and "
                        "other risk factors."
                    )

                # Detailed metrics
                with st.expander("📊 Detailed Model Information"):
                    col_detail1, col_detail2 = st.columns(2)

                    with col_detail1:
                        st.subheader("Input Data Summary")
                        st.dataframe(input_data.T, use_container_width=True)

                    with col_detail2:
                        st.subheader("Prediction Details")
                        st.write(f"**Raw Probability:** {probability:.4f}")
                        st.write(f"**Classification Threshold:** {custom_threshold:.2f}")
                        st.write(f"**Final Classification:** {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
                        st.write(f"**Model Confidence:** {max(probability, 1-probability):.2%}")

                # Risk factor summary
                with st.expander("🔍 Key Risk Factors"):
                    st.markdown("""
                    **Analyzed Risk Factors:**
                    - Age and sex demographics
                    - Chest pain characteristics
                    - Blood pressure and cholesterol levels
                    - Exercise capacity and heart rate response
                    - Electrocardiographic findings
                    - Thalassemia status

                    **Note:** This model considers multiple factors holistically.
                    Individual factors should also be evaluated by clinicians.
                    """)

                storage_result = persist_prediction(
                    input_frame=input_data,
                    threshold=custom_threshold,
                    probability=probability,
                    prediction=prediction,
                    risk_label=risk_level,
                )
                if storage_result and storage_result["status"] == "saved":
                    st.caption("Prediction saved to Supabase.")
                elif storage_result and storage_result["status"] == "error":
                    st.warning(storage_result["message"])

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

with col_clear:
    if st.button("🔄", use_container_width=True, help="Reset form"):
        st.rerun()

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <b>Disclaimer:</b> This is an educational machine learning demonstration. 
    Not intended for clinical diagnosis or treatment decisions.
    Patient diagnosis should only be made by qualified healthcare professionals.
    </div>
    """,
    unsafe_allow_html=True
)
