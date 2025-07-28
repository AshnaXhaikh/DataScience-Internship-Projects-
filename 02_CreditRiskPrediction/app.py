import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load preprocessed model assets
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    encoder_ohe = joblib.load("encoder_ohe.pkl")
    encoder_age = joblib.load("encoder_age.pkl")
    return model, encoder_ohe, encoder_age

model, encoder_ohe, encoder_age = load_model()

# Title
st.set_page_config(page_title="Credit Default Predictor", layout="centered")
st.title("🏦 Credit Risk Prediction")
st.markdown("Enter applicant data below to assess default risk.")

# User Inputs
age = st.selectbox("Age Group", ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74'])
income = st.number_input("Monthly Income", min_value=0, value=5000)
dtir1 = st.slider("Debt-to-Income Ratio", 0.0, 100.0, 35.0)
loan_amount = st.number_input("Loan Amount", min_value=1000, value=150000)
rate_of_interest = st.number_input("Interest Rate (%)", min_value=0.0, value=4.0)
LTV = st.slider("Loan-to-Value Ratio", 0.0, 100.0, 75.0)
term = st.number_input("Loan Term (months)", min_value=1, value=360)

loan_type = st.selectbox("Loan Type", ['type1', 'type2', 'type3'])
loan_purpose = st.selectbox("Loan Purpose", ['p1', 'p2', 'p3', 'p4'])
credit_worthiness = st.selectbox("Credit Worthiness", ['l1', 'l2'])

# Format inputs for model
def preprocess_input():
    input_data = pd.DataFrame({
        'income': [income],
        'dtir1': [dtir1],
        'loan_amount': [loan_amount],
        'rate_of_interest': [rate_of_interest],
        'LTV': [LTV],
        'term': [term],
        'age': [age],
        'loan_type': [loan_type],
        'loan_purpose': [loan_purpose],
        'Credit_Worthiness': [credit_worthiness]
    })

    # One-hot encode nominal fields
    encoded_nom = encoder_ohe.transform(input_data[['loan_type', 'Credit_Worthiness', 'loan_purpose']])
    encoded_nom_df = pd.DataFrame(encoded_nom, columns=encoder_ohe.get_feature_names_out(), index=input_data.index)

    # Ordinal encode age
    input_data['age'] = encoder_age.transform(input_data[['age']])
    
    # Combine
    final_input = pd.concat([input_data.drop(columns=['loan_type', 'Credit_Worthiness', 'loan_purpose']), encoded_nom_df], axis=1)
    return final_input

if st.button("🔍 Predict Default Risk"):
    input_for_model = preprocess_input()
    prediction = model.predict(input_for_model)[0]
    proba = model.predict_proba(input_for_model)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default. Probability: {proba:.2%}")
    else:
        st.success(f"✅ Low Risk of Default. Probability: {proba:.2%}")
