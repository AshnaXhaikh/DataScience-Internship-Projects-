import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Required for joblib to unpickle the model correctly
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# ---------------------- Load model and threshold ---------------------- #
pipeline_path = os.path.join(os.path.dirname(__file__), 'lgbm_pipeline.pkl')
threshold_path = os.path.join(os.path.dirname(__file__), 'optimal_threshold.pkl')

try:
    pipeline = joblib.load(pipeline_path)
    best_threshold = joblib.load(threshold_path)
except Exception as e:
    st.error(f"❌ Failed to load model or threshold: {e}")
    st.stop()

# ---------------------- UI Mapping ---------------------- #
job_map = {
    'management': 'Management',
    'technician': 'Technician',
    'entrepreneur': 'Entrepreneur',
    'blue-collar': 'Blue-Collar (manual worker)',
    'unknown': 'Not Specified',
    'retired': 'Retired',
    'admin.': 'Admin',
    'services': 'Services',
    'self-employed': 'Self-Employed',
    'unemployed': 'Unemployed',
    'housemaid': 'Housemaid',
    'student': 'Student'
}

education_map = {
    'tertiary': 'University or Higher Education',
    'secondary': 'High School',
    'primary': 'Primary School',
    'unknown': 'Not Specified'
}

pdays_map = {
    'never': 'Never Contacted Before',
    'old': 'Contacted Long Ago',
    'recent': 'Recently Contacted'
}

# Reverse mapping for input conversion
job_reverse = {v: k for k, v in job_map.items()}
edu_reverse = {v: k for k, v in education_map.items()}
pdays_reverse = {v: k for k, v in pdays_map.items()}

# ---------------------- STREAMLIT UI ---------------------- #
st.set_page_config(page_title="Loan Acceptance Predictor", layout="centered")
st.title("📊 Personal Loan Acceptance Prediction")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100)
    balance = st.number_input("Bank Balance", -10000, 100000)
    day = st.number_input("Day of Contact", 1, 31)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50)
    previous = st.number_input("Number of Contacts Before", 0, 50)

    job = st.selectbox("Job", list(job_map.values()))
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", list(education_map.values()))
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    poutcome = st.selectbox("Outcome of Previous Campaign", ['unknown', 'failure', 'other', 'success'])
    pdays_category = st.selectbox("Previous Contact Timing", list(pdays_map.values()))

    submit = st.form_submit_button("Predict")

# ---------------------- Make Prediction ---------------------- #
if submit:
    input_dict = {
        'age': [age],
        'balance': [balance],
        'day': [day],
        'campaign': [campaign],
        'previous': [previous],
        'job': [job_reverse[job]],
        'marital': [marital],
        'education': [edu_reverse[education]],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'poutcome': [poutcome],
        'pdays_category': [pdays_reverse[pdays_category]]
    }

    input_df = pd.DataFrame(input_dict)

    try:
        proba = pipeline.predict_proba(input_df)[0][1]  # Get probability of positive class
        prediction = int(proba >= best_threshold)

        st.markdown(f"### 💡 Prediction: {'✅ Will Accept' if prediction else '❌ Will Not Accept'}")
        st.markdown(f"**Probability of Acceptance:** {proba:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
