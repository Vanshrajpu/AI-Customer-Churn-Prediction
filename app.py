import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model.pkl")


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 35%),
        radial-gradient(circle at bottom right, #064e3b 0%, transparent 35%),
        #020617;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    margin-top: 10px;
    background: linear-gradient(90deg, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 35px;
}

.ai-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 15px;
}

.result-card {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.result-text {
    color: #cbd5e1;
    font-size: 17px;
    line-height: 1.7;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #0284c7, #059669);
    color: white;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    transform: scale(1.02);
}

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.8);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(148,163,184,0.15);
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 45px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🤖 ChurnAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Customer Churn Prediction System</div>',
    unsafe_allow_html=True
)


# =====================================================
# AI INTRO
# =====================================================

st.markdown("""
<div class="ai-card">

<div class="section-title">🤖 AI Customer Assistant</div>

<p style="color:#cbd5e1; font-size:16px;">
Welcome! I'm your Customer Churn AI Assistant.
<br><br>
Enter the customer's information below and I'll analyze
their profile using the trained Machine Learning model.
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# CUSTOMER INFORMATION
# =====================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


col1, col2 = st.columns(2)

with col1:

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# =====================================================
# SERVICES
# =====================================================

st.markdown(
    '<div class="section-title">📱 Services & Features</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =====================================================
# BILLING
# =====================================================

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

with col2:

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )


# =====================================================
# PREDICT BUTTON
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🚀 ANALYZE CUSTOMER"
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]
    })


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = model.predict(data)

    result = prediction[0]


    # ---------------------------------------------
    # Probability
    # ---------------------------------------------

    probability = None

    try:

        probability = model.predict_proba(data)[0][1] * 100

    except:

        pass


    # =================================================
    # RESULT
    # =================================================

    if result == 1:

        st.markdown("""
        <div class="result-card">

        <div style="font-size:60px;">🚨</div>

        <div class="result-title">
        CUSTOMER MAY CHURN
        </div>

        <p class="result-text">
        🤖 AI Analysis: This customer has a higher
        possibility of leaving the company.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "💡 Recommendation: Consider offering a personalized "
            "discount, better plan, loyalty benefit, or retention offer."
        )

    else:

        st.markdown("""
        <div class="result-card">

        <div style="font-size:60px;">✅</div>

        <div class="result-title">
        CUSTOMER LIKELY TO STAY
        </div>

        <p class="result-text">
        🤖 AI Analysis: This customer is likely to continue
        using the company's services.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.success(
            "💚 Recommendation: Maintain good customer service "
            "and continue customer engagement."
        )


    # =================================================
    # PROBABILITY
    # =================================================

    if probability is not None:

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                "CHURN" if result == 1 else "STAY"
            )

        with col2:

            st.metric(
                "Churn Probability",
                f"{probability:.2f}%"
            )

        with col3:

            risk = (
                "HIGH"
                if probability >= 70
                else "MEDIUM"
                if probability >= 40
                else "LOW"
            )

            st.metric(
                "Risk Level",
                risk
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">

🤖 <b>ChurnAI</b> — Machine Learning Customer Retention System

<br><br>

Built with Python • Pandas • Scikit-Learn • Streamlit

</div>
""", unsafe_allow_html=True)
