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
import streamlit as st
import pandas as pd

st.title("📱 Mobile Price Prediction App")
st.markdown("Enter the **features** to predict the price of the mobile phone.")

# Example feature list (modify these ranges/choices as per your dataset)
user_input = {}

# Features with a defined set of options
user_input["No_of_sim"] = st.selectbox("Number of SIMs", options=[1, 2])
user_input["Android_version"] = st.selectbox("Android Version", options=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
user_input["External_Memory"] = st.selectbox("External Memory (GB)", options=[0, 32, 64, 128, 256])

# Continuous/numerical features
user_input["Rating"] = st.number_input("Rating (0–5)", min_value=0.0, max_value=5.0, step=0.1)
user_input["Spec_score"] = st.number_input("Spec Score", min_value=0.0, max_value=100.0, step=0.5)
user_input["Ram"] = st.number_input("RAM (GB)", min_value=1.0, max_value=32.0, step=1.0)
user_input["Battery"] = st.number_input("Battery Capacity (mAh)", min_value=1000.0, max_value=6000.0, step=100.0)
user_input["Display"] = st.number_input("Display Size (inches)", min_value=4.0, max_value=7.5, step=0.1)
user_input["Camera"] = st.number_input("Camera (MP)", min_value=2.0, max_value=108.0, step=1.0)

# Predict button
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([user_input])
        input_transformed = pipeline.transform(input_df)
        predicted_price = model.predict(input_transformed)[0]
        st.success(f"Predicted Price: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
