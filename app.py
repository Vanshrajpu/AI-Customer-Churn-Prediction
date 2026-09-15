import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GreenBank PRO MAX",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,255,170,0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(0,150,255,0.08), transparent 25%),
        linear-gradient(135deg, #06110f, #071a18, #03100f);
    color: white;
}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061513, #081e1b);
    border-right: 1px solid rgba(0,255,170,0.12);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white;
}

section[data-testid="stSidebar"] label {
    color: #c9d8d5 !important;
    font-weight: 500;
}


/* ================= HERO ================= */

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px;
    margin-bottom: 25px;

    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(8,40,35,0.95),
            rgba(5,25,23,0.90)
        );

    border: 1px solid rgba(0,255,170,0.18);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        inset 0 0 50px rgba(0,255,170,0.025);
}

.hero:before {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -100px;
    top: -130px;

    background: rgba(0,255,170,0.12);

    border-radius: 50%;

    filter: blur(50px);
}

.hero h1 {
    position: relative;

    font-size: 48px;
    line-height: 1.08;

    margin: 18px 0 12px 0;

    font-weight: 800;

    background: linear-gradient(
        90deg,
        #ffffff,
        #8fffe0
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    position: relative;

    color: #a9c3be;

    font-size: 17px;

    max-width: 720px;

    line-height: 1.7;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 8px 14px;

    border-radius: 30px;

    background: rgba(0,255,170,0.08);

    border: 1px solid rgba(0,255,170,0.20);

    color: #8fffe0;

    font-size: 13px;
    font-weight: 600;
}

.live-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #00ffaa;

    box-shadow: 0 0 15px #00ffaa;

    animation: pulse 1.5s infinite;
}

.hero-orb {
    position: absolute;

    right: 55px;
    bottom: 30px;

    width: 100px;
    height: 100px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    font-size: 45px;

    background:
        radial-gradient(
            circle,
            rgba(0,255,170,0.18),
            rgba(0,255,170,0.03)
        );

    border: 1px solid rgba(0,255,170,0.25);

    box-shadow:
        0 0 40px rgba(0,255,170,0.12);

    animation: float 4s ease-in-out infinite;
}


/* ================= KPI CARDS ================= */

.kpi-card {
    padding: 22px;

    border-radius: 18px;

    background: rgba(8,31,28,0.75);

    border: 1px solid rgba(255,255,255,0.07);

    box-shadow: 0 10px 30px rgba(0,0,0,0.20);

    transition: 0.3s;
}

.kpi-card:hover {
    transform: translateY(-4px);

    border-color: rgba(0,255,170,0.20);
}

.kpi-title {
    color: #8ba6a0;
    font-size: 13px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 27px;
    font-weight: 800;
    color: white;
}

.kpi-sub {
    color: #6d8b84;
    font-size: 12px;
    margin-top: 5px;
}


/* ================= MAIN CARDS ================= */

.main-card {
    padding: 28px;

    margin-top: 20px;

    border-radius: 20px;

    background: rgba(7,27,24,0.75);

    border: 1px solid rgba(255,255,255,0.07);

    box-shadow: 0 15px 40px rgba(0,0,0,0.20);
}

.section-title {
    font-size: 21px;

    font-weight: 700;

    color: white;

    margin-bottom: 5px;
}

.section-subtitle {
    color: #78938d;

    font-size: 13px;

    margin-bottom: 20px;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;

    border: none;

    border-radius: 12px;

    padding: 13px 20px;

    background: linear-gradient(
        90deg,
        #00c98b,
        #00a878
    );

    color: white;

    font-weight: 700;

    font-size: 15px;

    transition: 0.3s;

    box-shadow:
        0 8px 25px rgba(0,200,140,0.18);
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(0,255,170,0.25);
}


/* ================= INPUTS ================= */

div[data-baseweb="select"] > div {
    background-color: #0b2420 !important;

    border-color: rgba(255,255,255,0.10) !important;

    border-radius: 10px !important;
}

input {
    background-color: #0b2420 !important;
    color: white !important;
}


/* ================= RESULT ================= */

.result-card {
    padding: 28px;

    border-radius: 20px;

    margin-top: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(0,150,110,0.15),
            rgba(8,30,27,0.80)
        );

    border: 1px solid rgba(0,255,170,0.20);
}

.result-title {
    color: #89a9a1;

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.result-value {
    font-size: 38px;

    font-weight: 800;

    margin-top: 5px;
}


/* ================= ANIMATIONS ================= */

@keyframes pulse {

    0% {
        transform: scale(1);
        opacity: 1;
    }

    50% {
        transform: scale(1.4);
        opacity: 0.5;
    }

    100% {
        transform: scale(1);
        opacity: 1;
    }

}

@keyframes float {

    0% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-10px);
    }

    100% {
        transform: translateY(0px);
    }

}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #5f7b75;

    font-size: 12px;

    padding: 30px 0 15px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def get_model():

    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    df = pd.read_csv(url)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing rows
    df.dropna(inplace=True)

    # Remove customer ID
    df.drop("customerID", axis=1, inplace=True)

    # Features and target
    X = df.drop("Churn", axis=1)

    y = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Numerical columns
    num = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Categorical columns
    cat = [
        c for c in X.columns
        if c not in num
    ]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                cat
            )
        ]
    )

    # Complete pipeline
    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    # Train model
    pipe.fit(X, y)

    return pipe


model = get_model()


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        <span class="live-dot"></span>
        AI PREDICTION ENGINE • LIVE
    </div>

    <h1>
        Customer Churn<br>
        Intelligence Platform
    </h1>

    <p>
        Predict customer churn using machine learning
        and turn customer data into actionable
        retention decisions.
    </p>

    <div style="
        margin-top:18px;
        display:flex;
        gap:9px;
        flex-wrap:wrap;
    ">

        <span class="hero-badge">
            🧠 Logistic Regression
        </span>

        <span class="hero-badge">
            ⚡ 19 Features
        </span>

        <span class="hero-badge">
            📈 Probability Scoring
        </span>

    </div>

    <div class="hero-orb">
        🤖
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="kpi-card">

        <div class="kpi-title">
            MODEL
        </div>

        <div class="kpi-value">
            Logistic
        </div>

        <div class="kpi-sub">
            Regression
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="kpi-card">

        <div class="kpi-title">
            INPUT FEATURES
        </div>

        <div class="kpi-value">
            19
        </div>

        <div class="kpi-sub">
            Customer attributes
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="kpi-card">

        <div class="kpi-title">
            OUTPUT
        </div>

        <div class="kpi-value">
            Risk %
        </div>

        <div class="kpi-sub">
            Probability scoring
        </div>

    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown("""
    <div class="kpi-card">

        <div class="kpi-title">
            PURPOSE
        </div>

        <div class="kpi-value">
            Retention
        </div>

        <div class="kpi-sub">
            Business intelligence
        </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "# 🏦 Customer Profile"
)

st.sidebar.markdown(
    "Enter customer information below."
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

Partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=1.0
)

TotalCharges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0,
    step=10.0
)


# =========================================================
# SERVICES
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown("### 📡 Services")

PhoneService = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# =========================================================
# CONTRACT INFORMATION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown("### 📄 Contract")

Contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# =========================================================
# MAIN PREDICTION SECTION
# =========================================================

st.markdown("""
<div class="main-card">

    <div class="section-title">
        🔮 Customer Churn Prediction
    </div>

    <div class="section-subtitle">
        Analyze customer behavior and estimate the probability
        that the customer will leave the service.
    </div>

</div>
""", unsafe_allow_html=True)


predict_col, info_col = st.columns(
    [1.5, 1]
)


# =========================================================
# PREDICT BUTTON
# =========================================================

with predict_col:

    st.markdown(
        "### 🚀 Run AI Prediction"
    )

    st.write(
        "Click the button to analyze the customer profile."
    )

    predict = st.button(
        "⚡ PREDICT CUSTOMER CHURN"
    )


# =========================================================
# MODEL INFO
# =========================================================

with info_col:

    st.markdown("""
    <div class="main-card">

        <div class="section-title">
            🧠 Model Information
        </div>

        <div class="section-subtitle">
            Machine learning pipeline
        </div>

        <p>
            <b>Algorithm:</b> Logistic Regression
        </p>

        <p>
            <b>Preprocessing:</b> StandardScaler + OneHotEncoder
        </p>

        <p>
            <b>Output:</b> Churn Probability
        </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # Create input dataframe

    df_input = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [SeniorCitizen],

        "Partner": [Partner],

        "Dependents": [Dependents],

        "tenure": [tenure],

        "PhoneService": [PhoneService],

        "MultipleLines": [MultipleLines],

        "InternetService": [InternetService],

        "OnlineSecurity": [OnlineSecurity],

        "OnlineBackup": [OnlineBackup],

        "DeviceProtection": [DeviceProtection],

        "TechSupport": [TechSupport],

        "StreamingTV": [StreamingTV],

        "StreamingMovies": [StreamingMovies],

        "Contract": [Contract],

        "PaperlessBilling": [PaperlessBilling],

        "PaymentMethod": [PaymentMethod],

        "MonthlyCharges": [MonthlyCharges],

        "TotalCharges": [TotalCharges]

    })


    # Prediction probability

    probability = model.predict_proba(
        df_input
    )[0][1]

    prediction = model.predict(
        df_input
    )[0]


    churn_percent = probability * 100


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        status = "HIGH CHURN RISK"

        message = (
            "This customer is likely to leave the service."
        )

    else:

        status = "LOW CHURN RISK"

        message = (
            "This customer is likely to stay with the service."
        )


    st.markdown(f"""
    <div class="result-card">

        <div class="result-title">
            AI Prediction Result
        </div>

        <div class="result-value">
            {status}
        </div>

        <p style="
            color:#a9c3be;
            margin-top:8px;
        ">
            {message}
        </p>

    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # PROBABILITY + GAUGE
    # =====================================================

    gauge_col, detail_col = st.columns(
        [1, 1]
    )


    with gauge_col:

        st.markdown(
            "### 📊 Churn Probability"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_percent,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 35
                    }
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "thickness": 0.75
                    },

                    "steps": [
                        {
                            "range": [0, 30]
                        },

                        {
                            "range": [30, 70]
                        },

                        {
                            "range": [70, 100]
                        }
                    ],

                    "threshold": {
                        "line": {
                            "width": 4
                        },

                        "thickness": 0.8,

                        "value": churn_percent
                    }
                }
            )
        )

        fig.update_layout(
            height=330,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font={
                "color": "white"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # RISK ANALYSIS
    # =====================================================

    with detail_col:

        st.markdown(
            "### 🎯 Risk Analysis"
        )

        if churn_percent >= 70:

            risk_level = "Critical"

            recommendation = (
                "Immediate retention action is recommended."
            )

        elif churn_percent >= 40:

            risk_level = "Medium"

            recommendation = (
                "Monitor this customer and consider targeted offers."
            )

        else:

            risk_level = "Low"

            recommendation = (
                "Customer appears relatively stable."
            )


        st.markdown(f"""
        <div class="main-card">

            <div class="kpi-title">
                RISK LEVEL
            </div>

            <div class="kpi-value">
                {risk_level}
            </div>

            <br>

            <div class="kpi-title">
                PROBABILITY
            </div>

            <div class="kpi-value">
                {churn_percent:.2f}%
            </div>

            <br>

            <div class="kpi-title">
                RECOMMENDATION
            </div>

            <p style="
                color:#b3c9c4;
                line-height:1.6;
            ">
                {recommendation}
            </p>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.markdown(
        "### 👤 Customer Profile Summary"
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


    with summary_col1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )


    with summary_col2:

        st.metric(
            "Monthly Charges",
            f"${MonthlyCharges:.2f}"
        )


    with summary_col3:

        st.metric(
            "Total Charges",
            f"${TotalCharges:.2f}"
        )


    with summary_col4:

        st.metric(
            "Contract",
            Contract
        )


    # =====================================================
    # BUSINESS INTELLIGENCE
    # =====================================================

    st.markdown("""
    <div class="main-card">

        <div class="section-title">
            💼 Business Intelligence
        </div>

        <div class="section-subtitle">
            Suggested actions based on the prediction
        </div>

    </div>
    """, unsafe_allow_html=True)


    if churn_percent >= 70:

        actions = [
            "🎁 Provide a personalized retention offer",
            "📞 Contact the customer proactively",
            "💳 Review pricing and payment experience",
            "🛠️ Identify possible service/support issues"
        ]

    elif churn_percent >= 40:

        actions = [
            "📊 Monitor customer engagement",
            "🎁 Consider a targeted loyalty offer",
            "📞 Follow up with the customer",
            "🔎 Review service usage patterns"
        ]

    else:

        actions = [
            "⭐ Maintain current customer experience",
            "🎁 Offer loyalty benefits",
            "📈 Encourage additional services",
            "🤝 Continue regular engagement"
        ]


    action_cols = st.columns(2)

    for i, action in enumerate(actions):

        with action_cols[i % 2]:

            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:15px;">

                <div style="
                    color:#d6e8e4;
                    font-size:14px;
                ">
                    {action}
                </div>

            </div>
            """, unsafe_allow_html=True)


    # =====================================================
    # INPUT FEATURE TABLE
    # =====================================================

    st.markdown(
        "### 📋 Prediction Input Data"
    )

    display_df = df_input.T.reset_index()

    display_df.columns = [
        "Feature",
        "Value"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    🏦 GreenBank PRO MAX
    &nbsp; • &nbsp;
    AI Customer Churn Prediction

    <br><br>

    Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True)
