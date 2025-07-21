import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

# Safe path to model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)
# Expected columns (from training)
expected_cols = ['age', 'bmi', 'children',
                 'sex_male', 'smoker_yes',
                 'region_northwest', 'region_southeast', 'region_southwest']

def preprocess_input(input_df):
    # One-hot encode input
    input_encoded = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)

    # Ensure all expected columns are present
    for col in expected_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0  # Add missing col as 0

    # Reorder columns
    input_encoded = input_encoded[expected_cols]
    return input_encoded

# Streamlit UI
st.title("Medical Insurance Cost Predictor")

age = st.number_input("Age", 18, 100)
bmi = st.number_input("BMI", 10.0, 60.0)
children = st.number_input("Number of Children", 0, 10)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict"):
    # Prepare input
    input_dict = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    }
    input_df = pd.DataFrame(input_dict)
    input_processed = preprocess_input(input_df)

    # Predict
    prediction = model.predict(input_processed)[0]
    st.success(f"Predicted Insurance Cost: ${prediction:,.2f}")
