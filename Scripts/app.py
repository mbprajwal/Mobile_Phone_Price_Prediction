import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths (safe with fallback)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', 'airtifacts'))
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR', 'data'), "output")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")

# Load model and preprocessing pipeline
@st.cache_resource
def load_model_pipeline():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = joblib.load(f)
    return model, pipeline, feature_names

model, pipeline, feature_names = load_model_pipeline()

# Streamlit UI
st.title("📱 Mobile Price Prediction App")

st.markdown("Enter the *numerical features* to predict the price of the mobile phone.")

# Dynamically create input fields for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([user_input])
        input_transformed = pipeline.transform(input_df)
        predicted_price = model.predict(input_transformed)[0]
        st.success(f"💰 Predicted Price: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")