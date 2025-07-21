import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

# App title and description
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")

st.title("🩺 Insurance Claim Prediction App")
st.markdown("""
Welcome! This app uses a trained Lasso Regression model to predict **insurance charges** based on user details.  
Fill in the details below and click **Predict** to get the result.
""")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This project was part of a data science internship.  
I used feature engineering, encoding, and regularized regression to build the model.

**Tech Stack:** Python, Scikit-learn, Streamlit  
**Created by:** Ashna Imtiaz
""")

# Input form
st.subheader("Enter your details:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert inputs to DataFrame
input_data = {
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == 'male' else 0],
    'smoker_yes': [1 if smoker == 'yes' else 0],
    'region_northwest': [1 if region == 'northwest' else 0],
    'region_southeast': [1 if region == 'southeast' else 0],
    'region_southwest': [1 if region == 'southwest' else 0]
}

input_df = pd.DataFrame(input_data)

# Predict button
if st.button("🚀 Predict"):
    prediction = model.predict(input_df)[0]
    threshold = 20000

    st.subheader("💰 Estimated Insurance Charges:")
    if prediction > threshold:
        st.markdown(f"<h3 style='color:red;'>${prediction:,.2f}</h3>", unsafe_allow_html=True)
        st.markdown("<span style='color:red;'>⚠️ High predicted charge!</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:green;'>${prediction:,.2f}</h3>", unsafe_allow_html=True)

# Optional footer
st.markdown("---")
st.markdown("© 2025 Ashna Imtiaz • Data Science Internship Project")

