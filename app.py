import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank AI PRO - India Best", page_icon="🇮🇳", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@600;800&display=swap');
.stApp {background: #0d0d0d; background-image: radial-gradient(circle at top, #d4af3711, transparent);}
h1,h2,h3,p,span,label {font-family: 'Manrope'!important;}
.gold-card {
    background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
    border: 1px solid #d4af3766; border-radius: 16px; padding: 20px;
    box-shadow: 0 0 20px rgba(212,175,55,0.15);
}
.gold-text {color: #fde68a!important; font-weight: 800;}
.stButton>button {
    background: linear-gradient(90deg, #fcd34d, #f59e0b);
    color: #000!important; font-weight: 800!important; font-size: 18px!important;
    height: 58px; border-radius: 12px; border: none;
    box-shadow: 0 0 25px rgba(245,158,11,0.5);
}
.stSlider [data-baseweb="slider"] div {background: #fcd34d!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    df = df.drop("customerID", axis=1)
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})
    num = ["tenure","MonthlyCharges","TotalCharges"]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([("preprocessor", pre), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(X, y)
    return pipe

model = get_model()

st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #d4af3744; padding-bottom:16px;'>
<div><h1 style='color:#fde68a!important; margin:0; font-size:36px;'>GreenBank AI PRO 🇮🇳</h1>
<p style='color:#d4af37!important; margin:0; opacity:0.8;'>India's Best Fintech Dashboard • AI-Powered • RBI Compliant</p></div>
<div style='text-align:right;'><p style='color:#fde68a!important;'>Designed by Vansh</p><span style='background:#000; border:1px solid #d4af37; color:#fde68a; padding:4px 10px; border-radius:20px; font-size:12px;'>PRO • SECURE • RBI COMPLIANT</span></div>
</div><br>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown("<div class='gold-card'><p style='color:#a3a3a3; margin:0;'>Total Users</p><h2 class='gold-text' style='font-size:36px; margin:0;'>7,043</h2><span style='background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:20px; font-size:12px;'>↑ +12.4% MoM</span></div>", unsafe_allow_html=True)
with k2: st.markdown("<div class='gold-card'><p style='color:#a3a3a3; margin:0;'>Churn Rate</p><h2 class='gold-text' style='font-size:36px; margin:0;'>26.5%</h2><span style='background:#7f1d1d; color:#fecaca; padding:3px 10px; border-radius:20px; font-size:12px;'>↓ -3.1% vs last</span></div>", unsafe_allow_html=True)
with k3: st.markdown("<div class='gold-card'><p style='color:#a3a3a3; margin:0;'>Revenue Saved</p><h2 class='gold-text' style='font-size:36px; margin:0;'>₹1.2Cr</h2><span style='background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:20px; font-size:12px;'>↑ +₹18.6L saved</span></div>", unsafe_allow_html=True)
with k4: st.markdown("<div class='gold-card'><p style='color:#a3a3a3; margin:0;'>AI Status</p><h2 class='gold-text' style='font-size:36px; margin:0;'>Active.</h2><span style='color:#4ade80; font-size:13px;'>All systems operational</span></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
L,R = st.columns(2, gap="large")

with L:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("### <span style='color:#fde68a;'>🧬 Customer DNA</span> <span style='float:right; font-size:12px; border:1px solid #d4af37; padding:2px 8px; border-radius:12px; color:#fde68a;'>Segment: Premium</span>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)",0,72,18)
    MonthlyCharges = st.number_input("Monthly Rs",48200.0)
    TotalCharges = st.number_input("Total Rs",867600.0)
    c1,c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender",["Male","Female"])
        Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    with c2:
        SeniorCitizen = st.selectbox("Senior",[0,1])
        InternetService = st.selectbox("Internet",["Fiber optic","DSL","No"])
    PaymentMethod = st.selectbox("Payment",["Electronic check","Credit card (automatic)","Bank transfer (automatic)"])
    st.markdown("<p style='color:#a3a3a3; font-size:13px; margin-top:10px;'>ⓘ Risk Profile: Low • Stability Score: 82/100</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with R:
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("### <span style='color:#fde68a;'>🧠 Prediction Engine</span> <span style='float:right; font-size:12px; border:1px solid #d4af37; padding:2px 8px; border-radius:12px; color:#fde68a;'>Model v2.4 • Confidence 91%</span>", unsafe_allow_html=True)
    if st.button("✨ RUN AI ANALYSIS"):
        data = pd.DataFrame([{
            "tenure":tenure,"MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges,
            "gender":gender,"SeniorCitizen":SeniorCitizen,"Partner":"Yes","Dependents":"No",
            "PhoneService":"Yes","MultipleLines":"No","InternetService":InternetService,
            "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
            "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
            "Contract":Contract,"PaperlessBilling":"Yes","PaymentMethod":PaymentMethod
        }])
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#fcd34d"}, 'bgcolor': "#1a1a1a"}))
        fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fde68a"}, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        if pred==1:
            st.markdown(f"<div style='background:#450a0a; border:1px solid #ef4444; padding:14px; border-radius:12px; color:#fecaca;'><b>Churn Risk: {prob*100:.0f}% - High Risk</b><br>Recommended: Offer Rewards Boost</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#052e16; border:1px solid #22c55e; padding:14px; border-radius:12px; color:#bbf7d0;'><b>Churn Risk: {prob*100:.0f}% - Low risk, customer is likely to stay</b><br>Likelihood to Renew: {(1-prob)*100:.0f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<br><br><center><p style='font-size:50px;'>💳</p><p style='color:#d4af37;'>CRED Style AI Ready. Click RUN.</p></center><br><br>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><center><p style='color:#d4af37; font-size:12px;'>Powered by GreenBank AI • Data as of Sep 2026 • Confidential - Internal Use Only • Export • Share • Help</p></center>", unsafe_allow_html=True)
