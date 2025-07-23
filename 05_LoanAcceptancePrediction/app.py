import streamlit as st
import pandas as pd
import joblib

# Define the categorize_pdays function (for internal logic or display)
def categorize_pdays(value):
    if value == -1:
        return 'never'
    elif value <= 100:
        return 'recent'
    else:
        return 'old'

st.title("Bank Marketing Campaign Prediction")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox("Marital", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Default", ['yes', 'no'])
housing = st.selectbox("Housing Loan", ['yes', 'no'])
loan = st.selectbox("Personal Loan", ['yes', 'no'])
contact = st.selectbox("Contact Communication", ['cellular', 'telephone', 'unknown'])
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Number of contacts during campaign", min_value=1, value=1)
pdays = st.number_input("Days since last contact (-1 if never)", value=-1)
previous = st.number_input("Number of previous contacts", min_value=0, value=0)
poutcome = st.selectbox("Previous Outcome", ['failure', 'nonexistent', 'success'])

# Optional display: Category based on pdays (not passed to model)
pdays_cat = categorize_pdays(pdays)
st.info(f"Customer contacted: **{pdays_cat}**")

# Create DataFrame with correct columns used in training
input_dict = {
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],  # original column
    'previous': [previous],
    'poutcome': [poutcome]
}

input_df = pd.DataFrame(input_dict)

# Load model and make prediction
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if st.button("Predict"):
    try:
        result = model.predict_proba(input_df)
        st.success(f"Predicted Probability of Subscription: {result[0][1]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
