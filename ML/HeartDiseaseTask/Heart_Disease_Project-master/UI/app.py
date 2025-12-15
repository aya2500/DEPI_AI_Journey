# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import math

# -----------------------------
# Helpers
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent  # adjust if your structure differs
MODELS_DIR = PROJECT_DIR / "Models"
DATA_DIR = PROJECT_DIR / "data"

def load_model(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{e}. Try: pip install scikit-learn (match version used to save models).") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model {path}: {e}") from e

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def safe_predict_proba(model, X):
    """Return probability of positive class. Handles models without predict_proba."""
    try:
        probs = model.predict_proba(X)
        # Assume positive class at index 1 if binary
        if probs.shape[1] == 2:
            return probs[:, 1]
        # fallback if multiclass - return max prob
        return probs.max(axis=1)
    except AttributeError:
        # fallback: use decision_function and convert with sigmoid
        if hasattr(model, "decision_function"):
            df = model.decision_function(X)
            # decision_function may be shape (n_samples,) or (n_samples, n_classes)
            if df.ndim == 1:
                return sigmoid(df)
            else:
                # multiclass fallback: take max then sigmoid
                return sigmoid(df.max(axis=1))
        else:
            # last resort: use model.predict as 0/1 and give probability 1.0 (not ideal)
            preds = model.predict(X)
            return np.array([1.0 if p == 1 else 0.0 for p in preds])

# -----------------------------
# Load models (with clear errors)
# -----------------------------
try:
    rf_model = load_model(MODELS_DIR / 'Random_Forest_model.pkl')
    lr_model = load_model(MODELS_DIR / 'Logistic_Regression_model.pkl')
    dt_model = load_model(MODELS_DIR / 'Decision_Tree_model.pkl')
    svm_model = load_model(MODELS_DIR / 'SVM_model.pkl')
    scaler = load_model(MODELS_DIR / 'scaler.pkl')
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Heart Disease Prediction App ❤️")
st.markdown("Enter patient data below to predict the risk of heart disease.")

# FEATURES CSV: try default path, otherwise allow upload or manual fallback
features_path = Path("/home/aya/HeartDisease/Heart_Disease_Project-master/Data/heart_disease_selected_features.csv")
df_features = None
if features_path.exists():
    try:
        df_features = pd.read_csv(features_path)
    except Exception as e:
        st.warning(f"Could not read features file at {features_path}: {e}")

if df_features is None:
    st.warning("Selected-features CSV not found at expected path.")
    uploaded = st.file_uploader("Upload heart_disease_selected_features.csv (optional)", type=['csv'])
    if uploaded is not None:
        try:
            df_features = pd.read_csv(uploaded)
            st.success("Features loaded from uploaded file.")
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")
            st.stop()
    else:
        st.info("If you don't have the CSV, you can enter features manually using a small default set.")
        # default minimal feature list (replace with your real features)
        default_features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        df_features = pd.DataFrame(columns=default_features + ['target'])

# Build feature names (drop target if present)
feature_names = df_features.drop(columns=[c for c in ['target'] if c in df_features.columns], errors='ignore').columns

st.write("Feature inputs (fill values):")
user_input = {}
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    # show in two columns for compactness
    with cols[i % 2]:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")

input_df = pd.DataFrame([user_input])

# Choose model
model_choice = st.selectbox("Choose model for prediction:", ['Random Forest', 'Logistic Regression', 'Decision Tree', 'SVM'])

if st.button("Predict"):
    try:
        # If model expects scaled input (we assume linear models & SVM do), scale those
        if model_choice in ['Logistic Regression', 'SVM']:
            try:
                X_input = scaler.transform(input_df)
            except Exception as e:
                st.error(f"Scaler transform failed: {e}. Check scaler compatibility with input features.")
                st.stop()
        else:
            X_input = input_df.values

        model = {'Random Forest': rf_model,
                 'Logistic Regression': lr_model,
                 'Decision Tree': dt_model,
                 'SVM': svm_model}[model_choice]

        pred = model.predict(X_input)[0]
        # safe probability
        prob_pos = safe_predict_proba(model, X_input)[0]

        if pred == 1:
            st.error(f"⚠️ Patient predicted to have heart disease — P(disease) = {prob_pos:.2f}")
        else:
            st.success(f"✅ Patient predicted healthy — P(no disease) = {(1-prob_pos):.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

# Helpful tips for the user
# st.markdown("---")
# st.markdown("**Troubleshooting tips:**")
# st.markdown("- If you see `InconsistentVersionWarning` from sklearn, either re-save the models using the current scikit-learn or install the version used to create the pickles. Example: `pip install scikit-learn==1.6.1`.")
# st.markdown("- Run the app with `streamlit run app.py` (not `python app.py`).")
# st.markdown("- If SVM probabilities are wrong or `predict_proba` missing, retrain SVC with `probability=True` or use a calibrated classifier (`CalibratedClassifierCV`).")
