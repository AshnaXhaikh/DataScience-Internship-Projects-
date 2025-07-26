import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature order (same as used during training)
feature_names = [
    'Upfront_charges', 'property_value', 'income', 'dtir1', 'Gender_Joint',
    'Gender_Sex Not Available', 'loan_type_type2',
    'business_or_commercial_nob/c', 'Neg_ammortization_not_neg',
    'lump_sum_payment_not_lpsm', 'credit_type_CRIF', 'credit_type_EQUI',
    'credit_type_EXP', 'co-applicant_credit_type_EXP',
    'submission_of_application_to_inst'
]

# Page selector
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Client Info", "Internal Assessment", "Prediction"])

# Session state to store inputs between pages
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# Page 1 - Client Info
if page == "Client Info":
    st.header("Client Provided Information")

    income = st.number_input("Monthly Income", min_value=0)
    property_value = st.number_input("Property Value", min_value=0)
    upfront_charges = st.number_input("Upfront Charges", min_value=0)
    dtir1 = st.slider("Debt-to-Income Ratio", 0.0, 1.0, step=0.01)
    
    gender = st.selectbox("Gender", ["Individual", "Joint", "Sex Not Available"])
    loan_type = st.selectbox("Loan Type", ["Conventional", "Type 2"])
    business_type = st.selectbox("Business or Commercial", ["Business", "No Business/Commercial"])
    neg_amortization = st.selectbox("Negative Amortization", ["Yes", "No"])
    lump_sum = st.selectbox("Lump Sum Payment", ["Yes", "No"])

    # Store inputs
    st.session_state.inputs.update({
        "income": income,
        "property_value": property_value,
        "Upfront_charges": upfront_charges,
        "dtir1": dtir1,
        "Gender_Joint": 1 if gender == "Joint" else 0,
        "Gender_Sex Not Available": 1 if gender == "Sex Not Available" else 0,
        "loan_type_type2": 1 if loan_type == "Type 2" else 0,
        "business_or_commercial_nob/c": 1 if business_type == "No Business/Commercial" else 0,
        "Neg_ammortization_not_neg": 1 if neg_amortization == "No" else 0,
        "lump_sum_payment_not_lpsm": 1 if lump_sum == "No" else 0
    })

    st.success("Saved! Now go to 'Internal Assessment'")

# Page 2 - Internal Assessment
elif page == "Internal Assessment":
    st.header("Internal Assessment (For Management Use)")

    credit_type = st.multiselect("Credit Type (select one or more)", ["CRIF", "EQUI", "EXP"])
    co_credit_type = st.selectbox("Co-applicant Credit Type", ["None", "EXP"])
    submission_status = st.selectbox("Application Submitted to Institution", ["Yes", "No"])

    # Store inputs
    st.session_state.inputs.update({
        "credit_type_CRIF": int("CRIF" in credit_type),
        "credit_type_EQUI": int("EQUI" in credit_type),
        "credit_type_EXP": int("EXP" in credit_type),
        "co-applicant_credit_type_EXP": 1 if co_credit_type == "EXP" else 0,
        "submission_of_application_to_inst": 1 if submission_status == "Yes" else 0
    })

    st.success("Saved! Now go to 'Prediction'")

# Page 3 - Prediction
elif page == "Prediction":
    st.header("Loan Default Prediction")

    if len(st.session_state.inputs) != len(feature_names):
        st.warning("Please complete all input pages first.")
    else:
        # Create input array in correct order
        X_input = np.array([[st.session_state.inputs[feat] for feat in feature_names]])

        # Scale
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = model.predict(X_scaled)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("⚠️ The client is likely to default.")
        else:
            st.success("✅ The client is **not likely** to default.")
