import streamlit as st
import numpy as np
import pickle

import os


# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "decision_tree_pipeline.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Define page names
pages = ["Client Details", "Management Details", "Prediction"]

# Define session state for input values
if "client_inputs" not in st.session_state:
    st.session_state.client_inputs = {}

if "management_inputs" not in st.session_state:
    st.session_state.management_inputs = {}

# Navigation
page = st.sidebar.selectbox("Choose Page", pages)

# Feature names (for mapping)
feature_names = [
    'Upfront_charges', 'property_value', 'income', 'dtir1',
    'Gender_Joint', 'Gender_Sex Not Available', 'loan_type_type2',
    'business_or_commercial_nob/c', 'Neg_ammortization_not_neg',
    'lump_sum_payment_not_lpsm', 'credit_type_CRIF', 'credit_type_EQUI',
    'credit_type_EXP', 'co-applicant_credit_type_EXP',
    'submission_of_application_to_inst'
]

# === 1. CLIENT DETAILS PAGE ===
if page == "Client Details":
    st.title("Client Details (Demographic Inputs)")
    st.session_state.client_inputs['Upfront_charges'] = st.number_input("Upfront Charges", min_value=0.0, step=100.0)
    st.session_state.client_inputs['property_value'] = st.number_input("Property Value", min_value=0.0, step=1000.0)
    st.session_state.client_inputs['income'] = st.number_input("Monthly Income", min_value=0.0, step=500.0)
    st.session_state.client_inputs['dtir1'] = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, step=1.0)
    
    gender = st.selectbox("Gender", ["Male", "Female", "Joint", "Not Available"])
    st.session_state.client_inputs['Gender_Joint'] = 1 if gender == "Joint" else 0
    st.session_state.client_inputs['Gender_Sex Not Available'] = 1 if gender == "Not Available" else 0

# === 2. MANAGEMENT DETAILS PAGE ===
elif page == "Management Details":
    st.title("Management Details (Internal Inputs)")
    loan_type = st.selectbox("Loan Type", ["type1", "type2"])
    st.session_state.management_inputs['loan_type_type2'] = 1 if loan_type == "type2" else 0

    business_type = st.selectbox("Business or Commercial", ["b/c", "nob/c"])
    st.session_state.management_inputs['business_or_commercial_nob/c'] = 1 if business_type == "nob/c" else 0

    neg_amort = st.selectbox("Negative Amortization", ["neg", "not_neg"])
    st.session_state.management_inputs['Neg_ammortization_not_neg'] = 1 if neg_amort == "not_neg" else 0

    lump_sum = st.selectbox("Lump Sum Payment", ["lpsm", "not_lpsm"])
    st.session_state.management_inputs['lump_sum_payment_not_lpsm'] = 1 if lump_sum == "not_lpsm" else 0

    credit_type = st.multiselect("Credit Type", ["CRIF", "EQUI", "EXP"])
    for credit in ["CRIF", "EQUI", "EXP"]:
        st.session_state.management_inputs[f'credit_type_{credit}'] = 1 if credit in credit_type else 0

    co_credit = st.selectbox("Co-applicant Credit Type", ["EXP", "Other"])
    st.session_state.management_inputs['co-applicant_credit_type_EXP'] = 1 if co_credit == "EXP" else 0

    submission = st.selectbox("Submission Type", ["to_inst", "other"])
    st.session_state.management_inputs['submission_of_application_to_inst'] = 1 if submission == "to_inst" else 0

# === 3. PREDICTION PAGE ===
elif page == "Prediction":
    st.title("Default Prediction")

    if st.button("Predict"):
        # Combine both inputs
        inputs = {}
        inputs.update(st.session_state.client_inputs)
        inputs.update(st.session_state.management_inputs)

        # Fill any missing values with 0 (for unselected options)
        full_input = [inputs.get(name, 0) for name in feature_names]

        prediction = pipeline.predict([full_input])[0]
        st.success("Prediction: " + ("❌ Default Risk" if prediction == 1 else "✅ No Default Risk"))
