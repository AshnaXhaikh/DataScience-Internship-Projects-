import streamlit as st
import pandas as pd
import joblib

# ─── Load Model and Threshold ───────────────────────────────────────────────
pipeline = joblib.load('lgbm_pipeline.pkl')
best_threshold = joblib.load('optimal_threshold.pkl')

# ─── Categorization Logic ───────────────────────────────────────────────────
def categorize_pdays(value):
    if value == -1:
        return 'never'
    elif value <= 100:
        return 'recent'
    else:
        return 'old'

# ─── Streamlit Page Settings ────────────────────────────────────────────────
st.set_page_config(page_title="Loan Acceptance Prediction", page_icon="🏦", layout="centered")

# ─── Sidebar Info ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 About This App")
    st.markdown("Predict whether the customers is likely to accept a personal loan offer.")
    st.markdown("**Author:** Ashna Imtiaz")
    st.markdown("**Model:** LightGBM Classifier")
    st.markdown("**Deploy:** Hugging Face Spaces")

# ─── Main App Title ─────────────────────────────────────────────────────────
st.title("🏦 Loan Acceptance Prediction App")
st.markdown("Provide customer details to predict their likelihood of **loan subscription**.")

# ─── Input Form ─────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100)
        balance = st.number_input("Account Balance (€)", min_value=0)
        day = st.number_input("Day of Contact", 1, 31)
        campaign = st.number_input("Contacts During Campaign", 1, 50)
        previous = st.number_input("Previous Contacts", 0, 50)
        job = st.selectbox("Job", [
            'admin.', 'technician', 'services', 'management', 'retired', 'blue-collar',
            'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed', 'unknown'
        ])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])

    with col2:
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.selectbox("Credit in Default?", ['no', 'yes'])
        housing = st.selectbox("Housing Loan?", ['yes', 'no'])
        loan = st.selectbox("Personal Loan?", ['no', 'yes'])
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        month = st.selectbox("Last Contact Month", [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
        poutcome = st.selectbox("Previous Campaign Outcome", ['success', 'failure', 'other', 'unknown'])
        pdays = st.number_input("Days Since Last Contact (-1 = never)", -1, 999)

    submit = st.form_submit_button("🔍 Predict")

# ─── On Submit: Run Prediction ──────────────────────────────────────────────
if submit:
    with st.spinner("Making prediction..."):
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
        y_proba = pipeline.predict_proba(input_df)[0][1]
        y_pred = int(y_proba >= best_threshold)

    # ─── Show Results ──────────────────────────────────────────────────────
    st.subheader("📈 Prediction Result")
    st.caption("This represents the model's confidence in the subscription prediction.")
    st.markdown(f"**Confidence Score:** `{y_proba:.2%}`")

    if y_pred == 1:
        st.success("✅ The client is **likely to subscribe** to the term deposit.")
    else:
        st.error("❌ The client is **unlikely to subscribe** to the term deposit.")
