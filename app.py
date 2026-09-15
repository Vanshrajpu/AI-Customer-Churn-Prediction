```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GreenBank PRO",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc,
        #eef2ff,
        #f0fdf4
    );
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.95);
    border-right: 1px solid #e2e8f0;
}

/* HERO */

.hero {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );

    border-radius: 24px;

    padding: 35px;

    color: white;

    margin-bottom: 20px;

    box-shadow:
        0 20px 40px rgba(15,23,42,0.25);

    position: relative;

    overflow: hidden;
}

.hero h1 {
    font-size: 38px;

    line-height: 1.1;

    margin: 12px 0;

    font-weight: 800;
}

.hero p {
    color: #cbd5e1;

    font-size: 14px;

    max-width: 600px;

    line-height: 1.6;
}

.badge {
    display: inline-block;

    background: rgba(34,197,94,0.15);

    color: #4ade80;

    padding: 7px 12px;

    border-radius: 20px;

    font-size: 12px;

    font-weight: 700;
}

/* CARDS */

.card {
    background: white;

    border-radius: 20px;

    padding: 22px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);

    margin-bottom: 18px;
}

/* KPI */

.kpi {
    background: white;

    border-radius: 18px;

    padding: 20px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.05);
}

.kpi-title {
    color: #64748b;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1px;
}

.kpi-value {
    color: #0f172a;

    font-size: 25px;

    font-weight: 800;

    margin-top: 5px;
}

/* BUTTON */

.stButton > button {
    width: 100%;

    height: 52px;

    border: none;

    border-radius: 12px;

    background: linear-gradient(
        135deg,
        #0f172a,
        #334155
    );

    color: white;

    font-weight: 800;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #1e293b,
        #475569
    );
}

/* RESULT */

.high-risk {
    background: #fee2e2;

    border: 1px solid #fecaca;

    border-radius: 15px;

    padding: 18px;

    color: #991b1b;
}

.low-risk {
    background: #dcfce7;

    border: 1px solid #bbf7d0;

    border-radius: 15px;

    padding: 18px;

    color: #166534;
}

.footer {
    text-align: center;

    color: #64748b;

    font-size: 12px;

    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    df = pd.read_csv(url)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    df.drop(
        "customerID",
        axis=1,
        inplace=True
    )

    X = df.drop(
        "Churn",
        axis=1
    )

    y = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_columns = [
        column
        for column in X.columns
        if column not in numerical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                StandardScaler(),
                numerical_columns
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    pipeline.fit(
        X,
        y
    )

    return pipeline


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🏦 GreenBank PRO")

    st.markdown(
        "### 👤 Customer Information"
    )

    st.divider()

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    st.markdown("### 📡 Services")

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    InternetService = st.selectbox(
        "Internet Service",
        [
            "Fiber optic",
            "DSL",
            "No"
        ]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    st.markdown("### 💳 Billing")

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        18
    )

    MonthlyCharges = st.slider(
        "Monthly Charges ($)",
        18,
        120,
        70
    )

    TotalCharges = st.slider(
        "Total Charges ($)",
        0,
        9000,
        1500
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

    <span class="badge">
        🟢 LIVE AI SYSTEM
    </span>

    <h1>
        Customer Retention<br>
        Intelligence Platform
    </h1>

    <p>
        Predict customer churn before it happens.
        Analyze customer behavior using machine learning
        and identify high-risk customers.
    </p>

    <div style="margin-top:18px;">

        <span class="badge">
            🤖 Logistic Regression
        </span>

        &nbsp;

        <span class="badge">
            ⚡ 19 Features
        </span>

        &nbsp;

        <span class="badge">
            📊 Probability Scoring
        </span>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI
# =========================================================

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(f"""
    <div class="kpi">

        <div class="kpi-title">
            TENURE
        </div>

        <div class="kpi-value">
            {tenure} Months
        </div>

    </div>
    """, unsafe_allow_html=True)


with k2:

    st.markdown(f"""
    <div class="kpi">

        <div class="kpi-title">
            MONTHLY CHARGES
        </div>

        <div class="kpi-value">
            ${MonthlyCharges}
        </div>

    </div>
    """, unsafe_allow_html=True)


with k3:

    st.markdown(f"""
    <div class="kpi">

        <div class="kpi-title">
            TOTAL CHARGES
        </div>

        <div class="kpi-value">
            ${TotalCharges}
        </div>

    </div>
    """, unsafe_allow_html=True)


with k4:

    st.markdown(f"""
    <div class="kpi">

        <div class="kpi-title">
            CONTRACT
        </div>

        <div class="kpi-value">
            {Contract}
        </div>

    </div>
    """, unsafe_allow_html=True)


st.write("")


# =========================================================
# MAIN AREA
# =========================================================

left, right = st.columns(
    [1.4, 1]
)


# =========================================================
# PREDICTION
# =========================================================

with left:

    st.markdown("""
    <div class="card">

        <h3>
            🎯 AI Prediction Engine
        </h3>

        <p style="color:#64748b;">
            19 customer features → Machine Learning Model
            → Churn Probability
        </p>

    </div>
    """, unsafe_allow_html=True)

    predict_button = st.button(
        "🚀 RUN AI ANALYSIS"
    )


    if predict_button:

        input_data = pd.DataFrame([{

            "gender": gender,

            "SeniorCitizen":
                1 if SeniorCitizen == "Yes" else 0,

            "Partner": Partner,

            "Dependents": Dependents,

            "tenure": tenure,

            "PhoneService": PhoneService,

            "MultipleLines": MultipleLines,

            "InternetService": InternetService,

            "OnlineSecurity": OnlineSecurity,

            "OnlineBackup": OnlineBackup,

            "DeviceProtection": DeviceProtection,

            "TechSupport": TechSupport,

            "StreamingTV": StreamingTV,

            "StreamingMovies": StreamingMovies,

            "Contract": Contract,

            "PaperlessBilling": PaperlessBilling,

            "PaymentMethod": PaymentMethod,

            "MonthlyCharges": MonthlyCharges,

            "TotalCharges": TotalCharges

        }])


        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = model.predict(
            input_data
        )[0]


        st.session_state["probability"] = probability

        st.session_state["prediction"] = prediction


    # Default value

    probability = st.session_state.get(
        "probability",
        0
    )

    prediction = st.session_state.get(
        "prediction",
        None
    )


    # =====================================================
    # GAUGE
    # =====================================================

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=probability * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 45,
                    "color": "#0f172a"
                }
            },

            title={
                "text": "Churn Probability"
            },

            gauge={

                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "color": "#0f172a"
                },

                "steps": [

                    {
                        "range": [0, 35],
                        "color": "#dcfce7"
                    },

                    {
                        "range": [35, 70],
                        "color": "#fef3c7"
                    },

                    {
                        "range": [70, 100],
                        "color": "#fee2e2"
                    }

                ]

            }

        )
    )


    fig.update_layout(

        height=320,

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),

        paper_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # RESULT
    # =====================================================

    if prediction is not None:

        if prediction == 1:

            st.markdown(f"""
            <div class="high-risk">

                <h3>
                    🔴 HIGH CHURN RISK
                </h3>

                <b>
                    Churn Probability:
                    {probability * 100:.2f}%
                </b>

                <p>
                    Customer has a high probability
                    of leaving the service.
                </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="low-risk">

                <h3>
                    🟢 LOW CHURN RISK
                </h3>

                <b>
                    Churn Probability:
                    {probability * 100:.2f}%
                </b>

                <p>
                    Customer is likely to remain
                    with the service.
                </p>

            </div>
            """, unsafe_allow_html=True)


# =========================================================
# BUSINESS IMPACT
# =========================================================

with right:

    st.markdown("""
    <div class="card">

        <h3>
            💼 Business Impact
        </h3>

        <p style="color:#64748b;">
            Customer financial information
        </p>

    </div>
    """, unsafe_allow_html=True)


    annual_value = MonthlyCharges * 12


    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Annual Revenue",
            f"${annual_value}"
        )

    with c2:

        st.metric(
            "Customer LTV",
            f"${TotalCharges}"
        )


    st.markdown("""
    <div class="card">

        <h3>
            🔍 Risk Factors
        </h3>

    </div>
    """, unsafe_allow_html=True)


    risks = []


    if Contract == "Month-to-month":

        risks.append(
            "Month-to-month contract"
        )


    if tenure < 12:

        risks.append(
            "Low customer tenure"
        )


    if TechSupport == "No":

        risks.append(
            "No technical support"
        )


    if InternetService == "Fiber optic":

        risks.append(
            "Fiber optic customer"
        )


    if PaymentMethod == "Electronic check":

        risks.append(
            "Electronic check payment"
        )


    if len(risks) == 0:

        st.success(
            "No major risk factors detected."
        )

    else:

        for risk in risks:

            st.warning(
                f"⚠️ {risk}"
            )


# =========================================================
# CUSTOMER DATA
# =========================================================

st.markdown("""
<div class="card">

    <h3>
        📋 Customer Feature Summary
    </h3>

</div>
""", unsafe_allow_html=True)


feature_data = pd.DataFrame({

    "Feature": [

        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"

    ],

    "Value": [

        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges

    ]

})


st.dataframe(
    feature_data,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    🏦 GreenBank PRO

    <br>

    AI Customer Churn Prediction System

    <br>

    Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True)
```
