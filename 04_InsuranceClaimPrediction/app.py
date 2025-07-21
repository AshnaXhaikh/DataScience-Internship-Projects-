import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Safely load the model from current directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# App title
st.title("🩺 Medical Insurance Cost Prediction App")
st.write("Predict the expected medical insurance charges based on input features.")

# User input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    submitted = st.form_submit_button("Predict")

# Preprocess input
if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # If preprocessing is needed (e.g., one-hot encoding), include that here
    # For this example, we assume the model pipeline handles preprocessing

    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Medical Charges: ${prediction[0]:,.2f}")
