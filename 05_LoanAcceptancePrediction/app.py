import streamlit as st
import pandas as pd
import pickle

# -------------------- Load model and threshold --------------------
try:
    with open("lgbm_pipeline.pkl", "rb") as file:
        pipeline = pickle.load(file)
    with open("best_threshold.pkl", "rb") as file:
        best_threshold = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model or threshold: {e}")
    st.stop()

# -------------------- Mapping dictionaries --------------------
job_map = {
    'admin.': 'admin.', 'blue-collar': 'blue-collar', 'entrepreneur': 'entrepreneur',
    'housemaid': 'housemaid', 'management': 'management', 'retired': 'retired',
    'self-employed': 'self-employed', 'services': 'services', 'student': 'student',
    'technician': 'technician', 'unemployed': 'unemployed', 'unknown': 'unknown'
}

education_map = {
    'primary': 'primary', 'secondary': 'secondary',
    'tertiary': 'tertiary', 'unknown': 'unknown'
}

month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcome_options = ['failure', 'other', 'success', 'unknown']

# -------------------- Categorize pdays --------------------
def categorize_pdays(value):
    if value == -1:
        return 'never'
    elif value <= 100:
        return 'recent'
    else:
        return 'old'

# -------------------- UI Layout --------------------
st.title("📈 Loan Acceptance Prediction")

with st.form("prediction_form"):
    st.subheader("Enter Customer Details")
    
    age = st.number_input("Age", 18, 100, step=1)
    balance = st.number_input("Bank Balance", -10000, 100000, step=100)
    day = st.number_input("Day of Contact", 1, 31, step=1)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50, step=1)
    previous = st.number_input("Number of Contacts Before", 0, 50, step=1)

    job = st.selectbox("Job", list(job_map.values()))
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education", list(education_map.values()))
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.selectbox("Month of Contact", month_options)
    poutcome = st.selectbox("Outcome of Previous Campaign", poutcome_options)
    pdays = st.number_input("Days Since Last Contact (-1 if never)", -1, 1000, step=1)

    submit = st.form_submit_button("Predict")

# -------------------- Prediction Logic --------------------
if submit:
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

    try:
        proba = pipeline.predict_proba(input_df)[0][1]  # probability of class 1
        prediction = int(proba >= best_threshold)

        st.success(f"🔍 Predicted Probability: {proba:.2f}")
        st.write(f"✅ Prediction: {'Subscribed' if prediction == 1 else 'Not Subscribed'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
