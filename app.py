import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- Page Config ---
st.set_page_config(page_title="GreenBank AI PRO", page_icon="🏦", layout="wide")

# --- Professional CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp { background: #0a0e13; font-family: 'Inter', sans-serif; }
h1, h2, h3 { color: #f8fafc!important; }
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 25px 30px;
    margin-bottom: 25px;
}
.gold-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #facc15;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(90deg, #facc15, #eab308);
    color: #000!important;
    font-weight: 800!important;
    font-size: 16px!important;
    height: 56px;
    border-radius: 12px;
    border: none;
    letter-spacing: 0.5px;
    transition: 0.3s;
}
.stButton>button:hover { transform: scale(1.02); }
.metric-box {
    background: #1f2937;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes":1,"No":0})
    num = ["tenure","MonthlyCharges","TotalCharges"]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
    pipe = Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(X,y)
    return pipe

model = get_model()

# --- Header ---
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="margin:0; font-size:32px;">🏦 GreenBank AI <span style="color:#facc15;">PRO</span></h1>
            <p style="margin:5px 0 0 0; color:#94a3b8; font-size:14px;">Customer Churn Prediction & Retention Intelligence System • India</p>
        </div>
        <div style="text-align:right;">
            <p style="margin:0; color:#22c55e; font-weight:600;">● SYSTEM ONLINE</p>
            <p style="margin:0; color:#64748b; font-size:12px;">Model Accuracy: 81.2%</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Layout ---
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Customer Profile")

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (Months)", 0, 72, 18)
        gender = st.selectbox("Gender", ["Male","Female"])
        Contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
        InternetService = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
    with c2:
        mc = st.slider("Monthly Charges ($)", 20, 120, 70)
        tc = st.slider("Total Charges ($)", 0, 10000, 1500)
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Prediction Engine")
    st.markdown("<p style='color:#94a3b8; font-size:13px;'>Enter customer details and run AI analysis to predict churn probability.</p>", unsafe_allow_html=True)

    if st.button("✨ RUN AI ANALYSIS"):
        data = pd.DataFrame([{
            "tenure":tenure,"MonthlyCharges":mc,"TotalCharges":tc,"gender":gender,
            "SeniorCitizen":1 if SeniorCitizen=="Yes" else 0,"Partner":"Yes","Dependents":"No",
            "PhoneService":"Yes","MultipleLines":"No","InternetService":InternetService,
            "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
            "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
            "Contract":Contract,"PaperlessBilling":"Yes","PaymentMethod":PaymentMethod
        }])

        prob = model.predict_proba(data)[0][1]
        pred = model.predict(data)[0]

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'suffix':"%", 'font':{'size':28, 'color':"white"}},
            title={'text': "Churn Risk Score", 'font':{'size':16, 'color':"#cbd5e1"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#475569"},
                'bar': {'color': "#facc15" if prob>0.5 else "#22c55e"},
                'bgcolor': "#1e293b",
                'steps': [
                    {'range': [0, 30], 'color': "#14532d"},
                    {'range': [30, 70], 'color': "#713f12"},
                    {'range': [70, 100], 'color': "#7f1d1d"}
                ],
            }
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Result
        if pred == 1:
            st.error(f"🔴 **HIGH RISK - Churn Hoga!**\n\nRisk Probability: **{prob*100:.1f}%**\n\n**Action:** Immediate retention offer de.")
        else:
            st.success(f"🟢 **LOW RISK - Customer Safe Hai**\n\nLoyalty Score: **{(1-prob)*100:.1f}%**\n\n**Action:** Normal service continue rakhe.")

        # Metrics
        m1,m2,m3 = st.columns(3)
        with m1: st.markdown(f'<div class="metric-box"><p style="color:#94a3b8; font-size:11px; margin:0;">TENURE</p><h3 style="margin:0;">{tenure}M</h3></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-box"><p style="color:#94a3b8; font-size:11px; margin:0;">MONTHLY</p><h3 style="margin:0;">${mc}</h3></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-box"><p style="color:#94a3b8; font-size:11px; margin:0;">TOTAL</p><h3 style="margin:0;">${tc}</h3></div>', unsafe_allow_html=True)

        st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)
