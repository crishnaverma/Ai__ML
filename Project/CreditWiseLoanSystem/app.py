import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# Load saved files
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")
num_imp = joblib.load("num_imputer.pkl")
cat_imp = joblib.load("cat_imputer.pkl")
ohe = joblib.load("onehot_encoder.pkl")
edu_encoder = joblib.load("education_encoder.pkl")

st.title("🏦 CreditWise Loan Approval System")
st.write("Enter applicant details below.")

col1, col2 = st.columns(2)

with col1:
    
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=20000.0
    )

    employment_status = st.selectbox(
    "Employment Status",
    ["Contract","Salaried","Self-employed","Unemployed"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single","Married"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=20.0
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=100000.0
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=500000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=200000.0
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=60
    )

    loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business","Car","Education","Home","Personal"]
    )

    property_area = st.selectbox(
    "Property Area",
    ["Rural","Semiurban","Urban"]
    )

    education = st.selectbox(
    "Education Level",
    ["Graduate","Not Graduate"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    employer_category = st.selectbox(
    "Employer Category",
    ["Business","Government","MNC","Private","Unemployed"]
    )


if st.button("Predict Loan Approval"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Employment_Status": [employment_status],
        "Age": [age],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Education_Level": [education],
        "Gender": [gender],
        "Employer_Category": [employer_category]
    })

    # Numerical columns
    numerical_cols = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term"
    ]

    # Categorical columns
    cat_imp_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category"
    ]

    ohe_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
    ]

    # Imputation
    input_data[numerical_cols] = num_imp.transform(input_data[numerical_cols])
    input_data[cat_imp_cols] = cat_imp.transform(input_data[cat_imp_cols])
 
    # Label Encode Education_Level
    input_data["Education_Level"] = edu_encoder.transform(
    input_data["Education_Level"]
    )

    # One Hot Encoding
    encoded = ohe.transform(input_data[ohe_cols])

    encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(ohe_cols)
    )

    input_data = pd.concat(
    [
        input_data.drop(columns=ohe_cols).reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
    )


    feature_order = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Education_Level",
        "Employment_Status_Salaried",
        "Employment_Status_Self-employed",
        "Employment_Status_Unemployed",
        "Marital_Status_Single",
        "Loan_Purpose_Car",
        "Loan_Purpose_Education",
        "Loan_Purpose_Home",
        "Loan_Purpose_Personal",
        "Property_Area_Semiurban",
        "Property_Area_Urban",
        "Gender_Male",
        "Employer_Category_Government",
        "Employer_Category_MNC",
        "Employer_Category_Private",
        "Employer_Category_Unemployed"
    ]

    input_data = input_data.reindex(columns=feature_order, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.write(f"Approval Probability: **{probability[1]*100:.2f}%**")
    else:
        st.error("❌ Loan Rejected")
        st.write(f"Rejection Probability: **{probability[0]*100:.2f}%**")