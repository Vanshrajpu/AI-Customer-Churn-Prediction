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
    page_title="ChurnIQ | Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(59,130,246,.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(139,92,246,.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );

    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #cbd5e1 !important;
    font-weight: 600;
}


/* ================= BRAND ================= */

.brand {
    padding: 8px 4px 20px;
}

.brand-icon {
    width: 48px;
    height: 48px;

    border-radius: 15px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        #3b82f6,
        #8b5cf6
    );

    font-size: 25px;

    box-shadow:
        0 12px 30px
        rgba(59,130,246,.25);
}

.brand-title {
    font-size: 23px;
    font-weight: 800;

    color: white;

    margin-top: 12px;
}

.brand-subtitle {
    font-size: 11px;
    color: #94a3b8;

    margin-top: 3px;
}


/* ================= SIDEBAR SECTION ================= */

.side-section {
    color: #94a3b8 !important;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1.4px;

    margin: 18px 0 8px;

    text-transform: uppercase;
}


/* ================= HERO ================= */

.hero {

    position: relative;

    overflow: hidden;

    padding: 34px 38px;

    border-radius: 28px;

    color: white;

    background:

        radial-gradient(
            circle at 85% 20%,
            rgba(96,165,250,.35),
            transparent 25%
        ),

        radial-gradient(
            circle at 65% 90%,
            rgba(139,92,246,.28),
            transparent 28%
        ),

        linear-gradient(
            135deg,
            #0f172a,
            #1e293b 55%,
            #312e81
        );

    box-shadow:
        0 25px 60px
        rgba(15,23,42,.18);

    margin-bottom: 22px;
}


.hero:before {

    content: "";

    position: absolute;

    width: 220px;
    height: 220px;

    right: -70px;
    top: -90px;

    border:
        1px solid
        rgba(255,255,255,.15);

    border-radius: 50%;

    box-shadow:

        0 0 0 35px
        rgba(255,255,255,.025),

        0 0 0 70px
        rgba(255,255,255,.02);
}


.hero-badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 12px;

    border-radius: 999px;

    background:
        rgba(255,255,255,.09);

    border:
        1px solid
        rgba(255,255,255,.12);

    color: #bfdbfe;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: .7px;
}


.live-dot {

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 0 0
        rgba(34,197,94,.6);

    animation:
        pulse 1.8s infinite;
}


@keyframes pulse {

    0% {
        box-shadow:
            0 0 0 0
            rgba(34,197,94,.6);
    }

    70% {
        box-shadow:
            0 0 0 9px
            rgba(34,197,94,0);
    }

    100% {
        box-shadow:
            0 0 0 0
            rgba(34,197,94,0);
    }
}


@keyframes float {

    0%,100% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(-8px);
    }
}


.hero h1 {

    font-size: 39px;

    line-height: 1.08;

    margin: 17px 0 10px;

    letter-spacing: -1.5px;
}


.hero p {

    color: #cbd5e1;

    max-width: 650px;

    font-size: 14px;

    line-height: 1.7;
}


.hero-orb {

    position: absolute;

    right: 60px;

    bottom: 28px;

    font-size: 82px;

    animation:
        float 4s ease-in-out infinite;

    filter:
        drop-shadow(
            0 15px 20px
            rgba(0,0,0,.25)
        );
}


/* ================= MAIN CARD ================= */

.card {

    background:
        rgba(255,255,255,.84);

    border:
        1px solid
        rgba(226,232,240,.9);

    border-radius: 22px;

    padding: 22px;

    box-shadow:
        0 12px 35px
        rgba(15,23,42,.07);

    backdrop-filter: blur(16px);
}


.card-title {

    font-size: 16px;

    font-weight: 800;

    color: #0f172a;

    margin-bottom: 3px;
}


.card-sub {

    font-size: 11px;

    color: #64748b;

    margin-bottom: 18px;
}


/* ================= KPI ================= */

.kpi {

    background:
        rgba(255,255,255,.9);

    border:
        1px solid
        #e2e8f0;

    border-radius: 18px;

    padding: 18px;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,.05);

    transition:
        .25s ease;

    min-height: 105px;
}


.kpi:hover {

    transform:
        translateY(-4px);

    box-shadow:
        0 15px 35px
        rgba(15,23,42,.10);
}


.kpi-label {

    color: #64748b;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: .9px;

    text-transform: uppercase;
}


.kpi-value {

    color: #0f172a;

    font-size: 24px;

    font-weight: 800;

    margin-top: 7px;
}


.kpi-note {

    font-size: 10px;

    margin-top: 4px;

    color: #64748b;
}


/* ================= RESULT ================= */

.result-high {

    padding: 18px;

    border-radius: 17px;

    background:
        linear-gradient(
            135deg,
            #fff1f2,
            #ffe4e6
        );

    border:
        1px solid
        #fecdd3;
}


.result-safe {

    padding: 18px;

    border-radius: 17px;

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #dcfce7
        );

    border:
        1px solid
        #bbf7d0;
}


.result-title {

    font-size: 16px;

    font-weight: 800;
}


.result-text {

    font-size: 12px;

    margin-top: 5px;

    color: #475569;
}


/* ================= RISK ================= */

.risk-item {

    padding: 10px 12px;

    border-radius: 10px;

    background: #fff7ed;

    border:
        1px solid
        #fed7aa;

    color: #9a3412;

    font-size: 11px;

    font-weight: 600;

    margin: 7px 0;
}


/* ================= ACTION ================= */

.action-item {

    padding: 11px 13px;

    border-radius: 11px;

    background: #eff6ff;

    border-left:
        4px solid
        #3b82f6;

    color: #1e40af;

    font-size: 11px;

    margin: 7px 0;
}


/* ================= BUTTON ================= */

.stButton > button {

    height: 52px;

    border: 0;

    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white !important;

    font-weight: 800;

    box-shadow:
        0 10px 25px
        rgba(37,99,235,.25);

    transition:
        .25s ease;
}


.stButton > button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 15px 32px
        rgba(37,99,235,.35);
}


/* ================= METRIC ================= */

div[data-testid="stMetric"] {

    background: white;

    border:
        1px solid
        #e2e8f0;

    padding: 13px;

    border-radius: 14px;
}


/* ================= FOOTER ================= */

.footer {

    text-align: center;

    color: #64748b;

    font-size: 11px;

    padding: 28px 0 10px;
}


@media (max-width: 900px) {

    .hero h1 {
        font-size: 29px;
    }

    .hero-orb {
        display: none;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def get_model():

    url = (
        "https://raw.githubusercontent.com/IBM/"
        "telco-customer-churn-on-icp4d/master/data/"
        "Telco-Customer-Churn.csv"
    )

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

    num = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    cat = [
        c for c in X.columns
        if c not in num
    ]

    preprocessor = ColumnTransformer([
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
    ])

    pipeline = Pipeline([
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
    ])

    pipeline.fit(
        X,
        y
    )

    return pipeline


model = get_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">

        <div class="brand-icon">
            📊
        </div>

        <div class="brand-title">
            ChurnIQ
        </div>

        <div class="brand-subtitle">
            Customer Retention Intelligence
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="side-section">'
        'Customer Profile'
        '</div>',
        unsafe_allow_html=True
    )


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


    st.markdown(
        '<div class="side-section">'
        'Services'
        '</div>',
        unsafe_allow_html=True
    )


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


    st.markdown(
        '<div class="side-section">'
        'Billing & Contract'
        '</div>',
        unsafe_allow_html=True
    )


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


    st.markdown("---")

    st.caption(
        "🤖 ML Model: Logistic Regression"
    )

    st.caption(
        "⚡ Real-time prediction"
    )


# =========================================================
# HERO
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
# KPI CARDS
# =========================================================

k1, k2, k3, k4 = st.columns(4)


k1.markdown(f"""
<div class="kpi">

    <div class="kpi-label">
        Customer Tenure
    </div>

    <div class="kpi-value">
        {tenure}
        <span style="font-size:13px">
            months
        </span>
    </div>

    <div class="kpi-note">
        Customer relationship length
    </div>

</div>
""", unsafe_allow_html=True)


k2.markdown(f"""
<div class="kpi">

    <div class="kpi-label">
        Monthly Revenue
    </div>

    <div class="kpi-value">
        ${MonthlyCharges}
    </div>

    <div class="kpi-note">
        Current monthly charges
    </div>

</div>
""", unsafe_allow_html=True)


k3.markdown(f"""
<div class="kpi">

    <div class="kpi-label">
        Customer Value
    </div>

    <div class="kpi-value">
        ${TotalCharges}
    </div>

    <div class="kpi-note">
        Total recorded charges
    </div>

</div>
""", unsafe_allow_html=True)


if Contract == "Month-to-month":

    contract_short = "Monthly"

elif Contract == "One year":

    contract_short = "1 Year"

else:

    contract_short = "2 Years"


k4.markdown(f"""
<div class="kpi">

    <div class="kpi-label">
        Contract
    </div>

    <div class="kpi-value">
        {contract_short}
    </div>

    <div class="kpi-note">
        {PaymentMethod}
    </div>

</div>
""", unsafe_allow_html=True)


st.write("")


# =========================================================
# MAIN COLUMNS
# =========================================================

left, right = st.columns(
    [1.25, .75],
    gap="large"
)


# =========================================================
# AI PREDICTION
# =========================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="card-title">
        🎯 AI Churn Prediction
    </div>

    <div class="card-sub">
        Analyze the customer profile and calculate
        the probability of churn.
    </div>
    """, unsafe_allow_html=True)


    if st.button(
        "🚀 RUN AI ANALYSIS",
        use_container_width=True
    ):

        df_input = pd.DataFrame([{

            "gender": gender,

            "SeniorCitizen":
                1 if SeniorCitizen == "Yes"
                else 0,

            "Partner": Partner,

            "Dependents": Dependents,

            "tenure": tenure,

            "PhoneService": PhoneService,

            "MultipleLines": MultipleLines,

            "InternetService":
                InternetService,

            "OnlineSecurity":
                OnlineSecurity,

            "OnlineBackup":
                OnlineBackup,

            "DeviceProtection":
                DeviceProtection,

            "TechSupport":
                TechSupport,

            "StreamingTV":
                StreamingTV,

            "StreamingMovies":
                StreamingMovies,

            "Contract": Contract,

            "PaperlessBilling":
                PaperlessBilling,

            "PaymentMethod":
                PaymentMethod,

            "MonthlyCharges":
                MonthlyCharges,

            "TotalCharges":
                TotalCharges

        }])


        probability = (
            model
            .predict_proba(df_input)[0][1]
        )


        prediction = (
            model
            .predict(df_input)[0]
        )


        st.session_state["prob"] = probability

        st.session_state["pred"] = prediction


    prob = st.session_state.get(
        "prob",
        0.0
    )


    pred = st.session_state.get(
        "pred",
        None
    )


    if pred is None:

        st.info(
            "👈 Configure the customer profile "
            "and click **RUN AI ANALYSIS**."
        )


    else:

        # =========================================
        # GAUGE
        # =========================================

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=prob * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 44,
                        "color": "#0f172a"
                    }
                },

                title={
                    "text":
                        "CHURN PROBABILITY",

                    "font": {
                        "size": 13,
                        "color": "#64748b"
                    }
                },

                gauge={

                    "axis": {
                        "range": [0, 100],
                        "tickfont": {
                            "color": "#64748b"
                        }
                    },

                    "bar": {
                        "color": "#4f46e5",
                        "thickness": .25
                    },

                    "bgcolor": "#f1f5f9",

                    "borderwidth": 0,

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

            height=290,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=10
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            font_family="Inter"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =========================================
        # RESULT
        # =========================================

        if pred == 1:

            st.markdown(f"""

            <div class="result-high">

                <div
                    class="result-title"
                    style="color:#be123c;"
                >

                    🔴 HIGH CHURN RISK —
                    {prob*100:.1f}%

                </div>


                <div class="result-text">

                    This customer shows a higher
                    probability of leaving.
                    A retention strategy should
                    be considered.

                </div>

            </div>

            """, unsafe_allow_html=True)


        else:

            safe = (
                1 - prob
            ) * 100


            st.markdown(f"""

            <div class="result-safe">

                <div
                    class="result-title"
                    style="color:#166534;"
                >

                    🟢 LOW CHURN RISK —
                    {safe:.1f}% CUSTOMER STABILITY

                </div>


                <div class="result-text">

                    The customer currently appears
                    relatively stable. Consider
                    loyalty and upselling opportunities.

                </div>

            </div>

            """, unsafe_allow_html=True)


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# BUSINESS INTELLIGENCE
# =========================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="card-title">
        💼 Business Intelligence
    </div>

    <div class="card-sub">
        Customer economics and recommended
        retention actions.
    </div>
    """, unsafe_allow_html=True)


    annual_value = (
        MonthlyCharges * 12
    )


    m1, m2 = st.columns(2)


    m1.metric(
        "Annual Revenue",
        f"${annual_value:,}"
    )


    m2.metric(
        "Total Charges",
        f"${TotalCharges:,}"
    )


    st.markdown("---")


    st.markdown(
        "**🔎 Churn Risk Signals**"
    )


    risks = []


    if Contract == "Month-to-month":

        risks.append(
            "Month-to-month contract"
        )


    if tenure < 12:

        risks.append(
            f"Short tenure: {tenure} months"
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


    if MonthlyCharges > 80:

        risks.append(
            "High monthly charges"
        )


    if risks:

        for risk in risks:

            st.markdown(
                f"""
                <div class="risk-item">
                    ⚠️ {risk}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.success(
            "No major manual risk signals detected."
        )


    st.markdown(
        "**💡 Recommended Actions**"
    )


    if pred == 1:

        actions = [

            "Offer a personalized retention discount.",

            "Contact the customer proactively.",

            "Review service/support issues.",

            "Consider a longer-term contract incentive."

        ]

    else:

        actions = [

            "Offer loyalty rewards.",

            "Explore relevant upsell opportunities.",

            "Maintain service quality.",

            "Monitor future churn probability."

        ]


    for action in actions:

        st.markdown(
            f"""
            <div class="action-item">
                → {action}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FEATURE SUMMARY
# =========================================================

st.write("")


st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)


st.markdown("""
<div class="card-title">
    📋 Customer Feature Summary
</div>

<div class="card-sub">
    Input values currently supplied to the
    machine learning pipeline.
</div>
""", unsafe_allow_html=True)


feature_df = pd.DataFrame({

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

        f"{tenure} months",

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

        f"${MonthlyCharges}",

        f"${TotalCharges}"

    ]

})


st.dataframe(

    feature_df,

    hide_index=True,

    use_container_width=True,

    height=350

)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <b>ChurnIQ</b>
    • AI Customer Churn Prediction Platform

    <br>

    Built with Python • Pandas • Scikit-learn •
    Streamlit • Plotly

</div>
""", unsafe_allow_html=True)
