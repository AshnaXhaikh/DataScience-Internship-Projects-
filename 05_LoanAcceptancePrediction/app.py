import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import lightgbm

# Required for joblib to unpickle the model correctly
from utils.custom_transformers import categorize_pdays
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# Load model and threshold using safe paths
pipeline_path = os.path.join(os.path.dirname(__file__), 'lgbm_pipeline.pkl')
threshold_path = os.path.join(os.path.dirname(__file__), 'optimal_threshold.pkl')

pipeline = joblib.load(pipeline_path)
best_threshold = joblib.load(threshold_path)

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
    contact = st.select
