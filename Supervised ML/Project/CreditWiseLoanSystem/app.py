import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #F8FAFC;
}

.hero {
    background: linear-gradient(90deg, #1E3A8A, #2563EB);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.step-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 3px 15px rgba(0,0,0,.08);
    margin-bottom: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: bold;
}

.next-btn > button {
    background: #2563EB;
    color: white;
    border: none;
}

.next-btn > button:hover {
    background: #1E40AF;
}

.back-btn > button {
    background: #E5E7EB;
    color: #111827;
    border: none;
}

.progress-text {
    text-align: center;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 5px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Load saved ML files
# =========================
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")
num_imp = joblib.load("num_imputer.pkl")
cat_imp = joblib.load("cat_imputer.pkl")
ohe = joblib.load("onehot_encoder.pkl")
edu_encoder = joblib.load("education_encoder.pkl")


# =========================
# Session state
# =========================
# Initialize ALL form fields before any step is displayed.
# This prevents AttributeError when the user reaches the Review page
# without Streamlit having created a widget key yet.

defaults = {
    "step": 1,
    "age": 30,
    "gender": "Male",
    "marital_status": "Single",
    "dependents": 0,
    "education": "Graduate",
    "employment_status": "Salaried",
    "employer_category": "Private",

    "applicant_income": 50000.0,
    "coapplicant_income": 20000.0,
    "credit_score": 700,
    "existing_loans": 0,
    "dti_ratio": 20.0,
    "savings": 100000.0,
    "collateral_value": 500000.0,

    "loan_amount": 200000.0,
    "loan_term": 60,
    "loan_purpose": "Personal",
    "property_area": "Urban",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

TOTAL_STEPS = 4


def next_step():
    if st.session_state.step < TOTAL_STEPS:
        st.session_state.step += 1


def previous_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1


def reset_form():
    st.session_state.step = 1
    st.session_state.prediction_done = False


# =========================
# Header
# =========================
st.markdown("""
<div class="hero">
    <h1>🏦 CreditWise Loan Approval System</h1>
    <h4>AI Powered Loan Approval Prediction</h4>
    <p>Built using Machine Learning & Streamlit</p>
</div>
""", unsafe_allow_html=True)


# =========================
# Step progress
# =========================
step_names = [
    "Personal Details",
    "Financial Details",
    "Loan Details",
    "Review & Predict"
]

st.markdown(
    f'<div class="progress-text">Step {st.session_state.step} of {TOTAL_STEPS} — '
    f'{step_names[st.session_state.step - 1]}</div>',
    unsafe_allow_html=True
)

st.progress(st.session_state.step / TOTAL_STEPS)


# ============================================================
# STEP 1 - PERSONAL DETAILS
# ============================================================
if st.session_state.step == 1:

    st.markdown("### 👤 Step 1: Personal Details")
    st.caption("Enter the applicant's basic information.")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=30,
            key="age"
        )

        st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="gender"
        )

        st.selectbox(
            "Marital Status",
            ["Single", "Married"],
            key="marital_status"
        )

        st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0,
            key="dependents"
        )

    with col2:
        st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"],
            key="education"
        )

        st.selectbox(
            "Employment Status",
            ["Contract", "Salaried", "Self-employed", "Unemployed"],
            key="employment_status"
        )

        st.selectbox(
            "Employer Category",
            ["Business", "Government", "MNC", "Private", "Unemployed"],
            key="employer_category"
        )

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="next-btn">', unsafe_allow_html=True)
        st.button("Next ➜", on_click=next_step, key="next_1")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STEP 2 - FINANCIAL DETAILS
# ============================================================
elif st.session_state.step == 2:

    st.markdown("### 💰 Step 2: Financial Details")
    st.caption("Enter the applicant's income, credit and financial information.")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=50000.0,
            key="applicant_income"
        )

        st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=20000.0,
            key="coapplicant_income"
        )

        st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700,
            key="credit_score"
        )

        st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=0,
            key="existing_loans"
        )

    with col2:
        st.number_input(
            "DTI Ratio",
            min_value=0.0,
            value=20.0,
            key="dti_ratio"
        )

        st.number_input(
            "Savings",
            min_value=0.0,
            value=100000.0,
            key="savings"
        )

        st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=500000.0,
            key="collateral_value"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        st.button("← Back", on_click=previous_step, key="back_2")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="next-btn">', unsafe_allow_html=True)
        st.button("Next ➜", on_click=next_step, key="next_2")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STEP 3 - LOAN DETAILS
# ============================================================
elif st.session_state.step == 3:

    st.markdown("### 🏠 Step 3: Loan Details")
    st.caption("Enter the requested loan information.")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=200000.0,
            key="loan_amount"
        )

        st.number_input(
            "Loan Term (Months)",
            min_value=1,
            value=60,
            key="loan_term"
        )

        st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"],
            key="loan_purpose"
        )

    with col2:
        st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
            key="property_area"
        )

        st.info(
            "💡 Make sure the loan amount, term and purpose "
            "match the applicant's actual requirement."
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        st.button("← Back", on_click=previous_step, key="back_3")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="next-btn">', unsafe_allow_html=True)
        st.button("Review Details ➜", on_click=next_step, key="next_3")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STEP 4 - REVIEW & PREDICT
# ============================================================
elif st.session_state.step == 4:

    st.markdown("### 🔍 Step 4: Review & Predict")
    st.caption("Check all entered information before running the ML model.")

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    st.markdown("#### 👤 Personal Details")

    personal_data = {
        "Age": st.session_state.age,
        "Gender": st.session_state.gender,
        "Marital Status": st.session_state.marital_status,
        "Dependents": st.session_state.dependents,
        "Education": st.session_state.education,
        "Employment": st.session_state.employment_status,
        "Employer Category": st.session_state.employer_category
    }

    col1, col2 = st.columns(2)

    personal_items = list(personal_data.items())
    for i, (label, value) in enumerate(personal_items):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{label}:** {value}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    st.markdown("#### 💰 Financial Details")

    financial_data = {
        "Applicant Income": st.session_state.applicant_income,
        "Coapplicant Income": st.session_state.coapplicant_income,
        "Credit Score": st.session_state.credit_score,
        "Existing Loans": st.session_state.existing_loans,
        "DTI Ratio": st.session_state.dti_ratio,
        "Savings": st.session_state.savings,
        "Collateral Value": st.session_state.collateral_value
    }

    col1, col2 = st.columns(2)

    financial_items = list(financial_data.items())
    for i, (label, value) in enumerate(financial_items):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{label}:** {value}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    st.markdown("#### 🏠 Loan Details")

    loan_data = {
        "Loan Amount": st.session_state.loan_amount,
        "Loan Term": f"{st.session_state.loan_term} months",
        "Loan Purpose": st.session_state.loan_purpose,
        "Property Area": st.session_state.property_area
    }

    col1, col2 = st.columns(2)

    loan_items = list(loan_data.items())
    for i, (label, value) in enumerate(loan_items):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{label}:** {value}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Navigation buttons
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        st.button("← Edit Details", on_click=previous_step, key="back_4")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="next-btn">', unsafe_allow_html=True)

        if st.button("🎯 Predict Loan Approval", key="predict"):
            # ============================================
            # Create input dataframe
            # ============================================
            input_data = pd.DataFrame({
                "Applicant_Income": [st.session_state.applicant_income],
                "Coapplicant_Income": [st.session_state.coapplicant_income],
                "Employment_Status": [st.session_state.employment_status],
                "Age": [st.session_state.age],
                "Marital_Status": [st.session_state.marital_status],
                "Dependents": [st.session_state.dependents],
                "Credit_Score": [st.session_state.credit_score],
                "Existing_Loans": [st.session_state.existing_loans],
                "DTI_Ratio": [st.session_state.dti_ratio],
                "Savings": [st.session_state.savings],
                "Collateral_Value": [st.session_state.collateral_value],
                "Loan_Amount": [st.session_state.loan_amount],
                "Loan_Term": [st.session_state.loan_term],
                "Loan_Purpose": [st.session_state.loan_purpose],
                "Property_Area": [st.session_state.property_area],
                "Education_Level": [st.session_state.education],
                "Gender": [st.session_state.gender],
                "Employer_Category": [st.session_state.employer_category]
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
            input_data[numerical_cols] = num_imp.transform(
                input_data[numerical_cols]
            )

            input_data[cat_imp_cols] = cat_imp.transform(
                input_data[cat_imp_cols]
            )

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

            # Feature order
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

            input_data = input_data.reindex(
                columns=feature_order,
                fill_value=0
            )

            # Scale
            input_scaled = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            approval = probability[1] * 100
            rejection = probability[0] * 100

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            st.progress(int(approval))

            if prediction == 1:
                st.balloons()

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.success("🎉 Congratulations! Loan Approved")

                st.metric(
                    label="Approval Probability",
                    value=f"{approval:.2f}%"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.error("❌ Loan Rejected")

                st.metric(
                    label="Rejection Probability",
                    value=f"{rejection:.2f}%"
                )

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔄 Start New Application", key="reset"):
        reset_form()
        st.rerun()
