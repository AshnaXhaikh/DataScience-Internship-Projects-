import streamlit as st
import numpy as np
import os
import joblib

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_pipeline.pkl')
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload decision_tree_pipeline.pkl.")
        return None
    return joblib.load(model_path)

pipeline = load_model()

# ─── Define Pages ────────────────────────────────────────────────────────────
pages = ["Client Details", "Management Details", "Prediction"]
page = st.sidebar.selectbox("Choose Page", pages)

# ─── Maintain State ──────────────────────────────────────────────────────────
if "client_inputs" not in st.session_state:
    st.session_state.client_inputs = {}

if "management_inputs" not in st.session_state:
    st.session_state.management_inputs = {}

# ─── Feature Names ───────────────────────────────────────────────────────────
feature_names = [
    'Upfront_charges', 'property_value', 'income', 'dtir1',
    'Gender_Joint', 'Gender_Sex Not Available', 'loan_type_type2',
    'business_or_commercial_nob/c', 'Neg_ammortization_not_neg',
    'lump_sum_payment_not_lpsm', 'credit_type_CRIF', 'credit_type_EQUI',
    'credit_type_EXP', 'co-applicant_credit_type_EXP',
    'submission_of_application_to_inst'
]

# ─── CLIENT DETAILS PAGE ─────────────────────────────────────────────────────
if page == "Client Details":
    st.header("Client Inputs – Demographic Info")

    st.session_state.client_inputs['Upfront_charges'] = st.number_input("Upfront Charges (USD)", min_value=0.0, step=100.0)
    st.session_state.client_inputs['property_value'] = st.number_input("Property Value (USD)", min_value=0.0, step=1000.0)
    st.session_state.client_inputs['income'] = st.number_input("Monthly Income (USD)", min_value=0.0, step=500.0)
    st.session_state.client_inputs['dtir1'] = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, step=1.0)

    gender = st.selectbox("Gender", ["Male", "Female", "Joint Account", "Undisclosed"])
    st.session_state.client_inputs['Gender_Joint'] = 1 if gender == "Joint Account" else 0
    st.session_state.client_inputs['Gender_Sex Not Available'] = 1 if gender == "Undisclosed" else 0

# ─── MANAGEMENT DETAILS PAGE ─────────────────────────────────────────────────
elif page == "Management Details":
    st.header("Internal Inputs – By Loan Manager")

    loan_type = st.selectbox("Loan Type", ["Type 1", "Type 2"])
    st.session_state.management_inputs['loan_type_type2'] = 1 if loan_type == "Type 2" else 0

    business = st.selectbox("Business/Commercial Category", ["Business/Commercial", "Non Business/Commercial"])
    st.session_state.management_inputs['business_or_commercial_nob/c'] = 1 if business == "Non Business/Commercial" else 0

    neg_amort = st.selectbox("Negative Amortization Option", ["Enabled", "Disabled"])
    st.session_state.management_inputs['Neg_ammortization_not_neg'] = 1 if neg_amort == "Disabled" else 0

    lump_sum = st.selectbox("Lump Sum Payment Option", ["Enabled", "Disabled"])
    st.session_state.management_inputs['lump_sum_payment_not_lpsm'] = 1 if lump_sum == "Disabled" else 0

    credit_types = st.multiselect("Reported Credit Bureaus", ["CRIF", "EQUIFAX", "EXPERIAN"])
    for bureau in ["CRIF", "EQUI", "EXP"]:
        label = f"credit_type_{bureau}"
        st.session_state.management_inputs[label] = 1 if bureau in credit_types else 0

    co_credit = st.selectbox("Co-applicant Credit Report From", ["EXPERIAN", "Other"])
    st.session_state.management_inputs['co-applicant_credit_type_EXP'] = 1 if co_credit == "EXPERIAN" else 0

    submission = st.selectbox("Submission Type", ["Submitted to Institution", "Other"])
    st.session_state.management_inputs['submission_of_application_to_inst'] = 1 if submission == "Submitted to Institution" else 0

# ─── PREDICTION PAGE ────────────────────────────────────────────────────────
elif page == "Prediction":
    st.header("📊 Predict Credit Default Risk")

    if st.button("Run Prediction"):
        if pipeline is None:
            st.error("Model is not loaded. Check file path or upload the model.")
        else:
            # Combine inputs
            inputs = {**st.session_state.client_inputs, **st.session_state.management_inputs}
            full_input = [inputs.get(feat, 0) for feat in feature_names]

            prediction = pipeline.predict([full_input])[0]
            probability = pipeline.predict_proba([full_input])[0][1]  # Default class prob

            st.subheader("🔎 Prediction Result")
            if prediction == 1:
                st.error(f"❌ High Risk of Default (Probability: {probability:.2f})")
            else:
                st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")
