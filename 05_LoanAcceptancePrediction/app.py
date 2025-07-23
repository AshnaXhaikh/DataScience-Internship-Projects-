import streamlit as st
import pandas as pd
import pickle
from utils.custom_transformers import categorize_pdays
# Function for pdays_category (used in your notebook)
def categorize_pdays(value):
    if value == -1:
        return 'never'
    elif value <= 100:
        return 'recent'
    else:
        return 'old'

# Load model and threshold
with open('lgbm_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('optimal_threshold.pkl', 'rb') as f:
    best_threshold = pickle.load(f)


st.title("Bank Term Deposit Prediction")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100)
    balance = st.number_input("Bank Balance", -10000, 100000)
    day = st.number_input("Day of Contact", 1, 31)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50)
    previous = st.number_input("Number of Contacts Before", 0, 50)
    job = st.selectbox("Job", [
        'admin.', 'technician', 'services', 'management', 'retired', 'blue-collar',
        'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed', 'unknown'
    ])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    poutcome = st.selectbox("Previous Outcome", ['success', 'failure', 'other', 'unknown'])
    pdays = st.number_input("Days Since Last Contact (-1 means never contacted)", -1, 999)

    submit = st.form_submit_button("Predict")

if submit:
    # Manually engineer features as done in notebook
    pdays_category = categorize_pdays(pdays)

    input_dict = {
        'age': [age],
        'balance': [balance],
        'day': [day],
        'campaign': [campaign],
        'previous': [previous],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'poutcome': [poutcome],
        'pdays_category': [pdays_category]
    }

    input_df = pd.DataFrame(input_dict)

    # Predict probability
    y_proba = pipeline.predict_proba(input_df)[0][1]

    # Apply threshold
    y_pred = int(y_proba >= best_threshold)

    st.subheader("Prediction Result")
    st.write(f"**Probability of Subscribing:** {y_proba:.2f}")
    st.write(f"**Prediction:** {'Subscribed' if y_pred == 1 else 'Not Subscribed'}")
