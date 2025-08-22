
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList  
except Exception:
   
    pass


import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    from pytorch_tabnet.tab_model import TabNetClassifier  
except Exception:
   
    pass

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction (TabNet)")

@st.cache_resource
def load_model(path="model.pkl"):
  
    try:
        bundle = joblib.load(path)
        return bundle
    except Exception as e:
        import sklearn, sys
        st.error(
            f"Failed to load '{path}'.\n\n"
            f"Python: {sys.version.split()[0]}\n"
            f"scikit-learn: {getattr(sklearn, '__version__', 'unknown')}\n"
            f"joblib: {getattr(joblib, '__version__', 'unknown')}\n\n"
            f"Error: {e}"
        )
        st.stop()

bundle = load_model()
raw_columns = bundle["raw_columns"]
preprocessor = bundle["preprocessor"]
sel_idx = bundle["selected_idx"]
tabnet = bundle["tabnet"]

st.markdown("Enter patient information and click **Predict**.")

with st.form("inp"):
    
    age = st.number_input("Age", 1, 120, 35)
    gender = st.selectbox("Gender", ["Male","Female"])
    pulse_rate = st.number_input("Pulse Rate (bpm)", 30, 200, 72)
    systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 260, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 30, 150, 80)
    glucose = st.number_input("Glucose (mmol/L)", 2.0, 30.0, 5.5, step=0.1, format="%.2f")
    height = st.number_input("Height (m)", 1.0, 2.5, 1.70, step=0.01, format="%.2f")
    weight = st.number_input("Weight (kg)", 20.0, 250.0, 70.0, step=0.1, format="%.1f")
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1, format="%.1f")
    family_diabetes = st.selectbox("Family Diabetes (0/1)", [0,1])
    hypertensive = st.selectbox("Hypertensive (0/1)", [0,1])
    family_hypertension = st.selectbox("Family Hypertension (0/1)", [0,1])
    cardiovascular_disease = st.selectbox("Cardiovascular Disease (0/1)", [0,1])
    stroke = st.selectbox("Stroke (0/1)", [0,1])

    submitted = st.form_submit_button("Predict")

if submitted:
    
    row = {
        "age": age, "gender": gender, "pulse_rate": pulse_rate,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "glucose": glucose, "height": height, "weight": weight, "bmi": bmi,
        "family_diabetes": family_diabetes, "hypertensive": hypertensive,
        "family_hypertension": family_hypertension,
        "cardiovascular_disease": cardiovascular_disease, "stroke": stroke
    }
    X = pd.DataFrame([row], columns=raw_columns)

    X_t = preprocessor.transform(X)
    X_sel = X_t[:, sel_idx].astype(np.float32)

    proba = tabnet.predict_proba(X_sel)[0, 1]
    pred = tabnet.predict(X_sel)[0]
    label = "Diabetic" if pred == 1 else "Non-Diabetic"

    st.subheader(f"Prediction: **{label}**")
    st.metric("Probability of Diabetes", f"{proba:.3f}")
