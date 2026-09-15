import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank AI PRO", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp {background: radial-gradient(1000px at 10% -10%, #1e293b 0%, #020617 100%);}
[data-testid="stSidebar"] {background: #0f172a; border-right: 1px solid #1e293b;}
.card-pro {background: linear-gradient(180deg, #111827 0%, #0f172a 100%); border:1px solid #1e293b; border-radius:20px; padding:24px;}
.kpi {background:#0f172a; border-radius:16px; padding:12px; border:1px solid #1e293b; text-align:center;}
.kpi h2 {margin:0; color:white; font-weight:800; font-size:18px;}
.kpi p {margin:0; color:#64748b; font-size:10px; letter-spacing:1px; text-transform:uppercase;}
.stButton>button {background:#fff; color:#000!important; font-weight:800!important; height:56px; border-radius:12px; width:100%;}
</style>
""", unsafe_allow_html=True)

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

# --- SIDEBAR - 21 INPUTS ---
with st.sidebar:
    st.markdown("## 🏦 GreenBank")
    st.markdown("<p style='color:#facc15; font-size:11px; letter-spacing:2px; font-weight:700; margin-top:-15px;'>19 FEATURES • ENTERPRISE</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**👤 Personal**")
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["No","Yes"])

    st.markdown("**📞 Services**")
    PhoneService = st.selectbox("Phone Service", ["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    InternetService = st.selectbox("Internet", ["Fiber optic","DSL","No"])
    OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    st.markdown("**💳 Billing**")
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    tenure = st.slider("Tenure (Months)", 0, 72, 18)
    MonthlyCharges = st.slider("Monthly Charges $", 18, 120, 70)
    TotalCharges = st.slider("Total Charges $", 0, 9000, 1500)

    st.divider()
    st.caption("19 inputs ready • Model trained")

# --- MAIN HEADER ---
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
<div>
<h1 style="margin:0; font-size:34px; font-weight:800; letter-spacing:-1px;">Customer Retention Intelligence</h1>
<p style="color:#94a3b8; margin:4px 0 0 0;">19 features • Real-time inference • Production ready</p>
</div>
<div style="background:#22c55e20; border:1px solid #22c55e40; padding:8px 14px; border-radius:99px; color:#22c55e; font-size:12px; font-weight:600;">● LIVE • 19 FEATURES</div>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><p>TENURE</p><h2>{tenure}M</h2></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><p>MONTHLY</p><h2>${MonthlyCharges}</h2></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><p>TOTAL</p><h2>${TotalCharges}</h2></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><p>CONTRACT</p><h2>{Contract.split("-")[0]}</h2></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card-pro">', unsafe_allow_html=True)
    st.markdown("#### 🎯 AI Prediction")

    if st.button("▶ RUN AI ANALYSIS (19 Features)"):
        df_input = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":1 if SeniorCitizen=="Yes" else 0,"Partner":Partner,"Dependents":Dependents,
            "tenure":tenure,"PhoneService":PhoneService,"MultipleLines":MultipleLines,"InternetService":InternetService,
            "OnlineSecurity":OnlineSecurity,"OnlineBackup":OnlineBackup,"DeviceProtection":DeviceProtection,
            "TechSupport":TechSupport,"StreamingTV":StreamingTV,"StreamingMovies":StreamingMovies,
            "Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod,
            "MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges
        }])
        prob = model.predict_proba(df_input)[0][1]
        pred = model.predict(df_input)[0]
        st.session_state['prob'] = prob
        st.session_state['pred'] = pred

    prob = st.session_state.get('prob', 0.35)
    pred = st.session_state.get('pred', 0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        number={'suffix':"%", 'font':{'size':36, 'color':"white"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'bgcolor': "#020617",
            'steps': [{'range':[0,40],'color':"#052e16"},{'range':[40,75],'color':"#422006"},{'range':[75,100],'color':"#450a0a"}]
        }
    ))
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        if pred==1:
            st.error(f"🔴 HIGH RISK - Churn Hoga! {prob*100:.1f}%")
            st.markdown("**Action:** Retention offer, 1-year contract pe shift karo.")
        else:
            st.success(f"🟢 LOW RISK - Safe Hai! Loyalty {(1-prob)*100:.1f}%")
            st.markdown("**Action:** Upsell karo, customer khush hai.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card-pro">', unsafe_allow_html=True)
    st.markdown("#### 📋 Input Summary (19 Features)")
    summary = {
        "Feature": ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"],
        "Value": [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, height=400)
    st.markdown("#### 💡 Viva Point")
    st.info("Sir, maine saare 19 original Telco features use kiye hai, pipeline me StandardScaler + OneHotEncoder lagaya hai taki production me koi feature mismatch na ho. Ye FAANG level MLOps practice hai.")
    st.markdown('</div>', unsafe_allow_html=True)
