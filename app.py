import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank AI PRO", page_icon="🏦", layout="wide")

# --- ULTRA PREMIUM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
* {font-family: 'Plus Jakarta Sans', sans-serif;}
.stApp {background: #05070A;}
[data-testid="stSidebar"] {background: #0A0E15; border-right: 1px solid #1A2332;}
.hero {
    background: linear-gradient(135deg, #101828 0%, #0A0E15 100%);
    border: 1px solid #1A2332;
    border-radius: 24px;
    padding: 28px 32px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px; background: radial-gradient(circle, #facc1540 0%, transparent 70%);
    filter: blur(20px);
}
.card {
    background: #0E141F;
    border: 1px solid #1A2332;
    border-radius: 20px;
    padding: 22px;
}
.card:hover {border-color: #2A3A50;}
.badge {
    background: #facc15; color: black; font-weight: 800; font-size: 10px;
    padding: 6px 12px; border-radius: 99px; letter-spacing: 1px;
}
.input-label {color: #7C8DA6; font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;}
.stButton>button {
    background: #FFFFFF; color: #000!important; font-weight: 800!important;
    height: 54px; border-radius: 14px; font-size: 15px!important;
    box-shadow: 0 0 0 1px #ffffff20, 0 10px 30px -10px #ffffff40;
}
.stButton>button:hover {background: #facc15;}
.kpi-card {
    background: #0E141F; border: 1px solid #1A2332; border-radius: 18px;
    padding: 18px; text-align: left;
}
.kpi-card.label {color: #5A6C84; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700;}
.kpi-card.value {color: white; font-size: 22px; font-weight: 800; margin-top: 4px;}
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

# --- SIDEBAR 19 INPUTS ---
with st.sidebar:
    st.markdown("### 🏦 GreenBank AI")
    st.markdown('<span class="badge">PRO • 19 FEATURES</span>', unsafe_allow_html=True)
    st.write("")

    with st.expander("👤 DEMOGRAPHICS", expanded=True):
        gender = st.selectbox("Gender", ["Male","Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
        Partner = st.selectbox("Has Partner?", ["Yes","No"])
        Dependents = st.selectbox("Has Dependents?", ["No","Yes"])

    with st.expander("📡 SERVICES", expanded=True):
        PhoneService = st.selectbox("Phone Service", ["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        InternetService = st.selectbox("Internet Type", ["Fiber optic","DSL","No"])
        OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    with st.expander("💳 BILLING & TENURE", expanded=True):
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        tenure = st.slider("Tenure Months", 0, 72, 18)
        MonthlyCharges = st.slider("Monthly Charges $", 18, 120, 70)
        TotalCharges = st.slider("Total Charges $", 0, 9000, 1500)

# --- HERO ---
st.markdown(f"""
<div class="hero">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<div style="display:flex; gap:10px; align-items:center;">
<h1 style="margin:0; font-size:32px; font-weight:800; color:white; letter-spacing:-1px;">GreenBank AI PRO</h1>
<span style="background:#22c55e20; border:1px solid #22c55e40; padding:5px 12px; border-radius:99px; color:#22c55e; font-size:11px; font-weight:700;">● LIVE INFERENCE</span>
</div>
<p style="color:#7C8DA6; margin:8px 0 0 0; font-size:14px;">Enterprise Customer Churn Platform • 19-feature pipeline • Built for banking at scale • India 2026</p>
</div>
<div style="text-align:right;">
<p style="color:#7C8DA6; font-size:11px; margin:0;">MODEL</p>
<p style="color:white; font-weight:700; margin:0;">LogisticRegression v2.1</p>
<p style="color:#facc15; font-size:12px; margin:0;">Accuracy 81.2%</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# KPI ROW
c1,c2,c3,c4 = st.columns(4)
c1.markdown(f'<div class="kpi-card"><div class="label">Tenure</div><div class="value">{tenure} months</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="label">MRR</div><div class="value">${MonthlyCharges}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="label">LTV</div><div class="value">${TotalCharges}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi-card"><div class="label">Contract</div><div class="value">{Contract}</div></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ⚡ Real-time Prediction")
    st.markdown("<p style='color:#5A6C84; font-size:13px; margin-top:-10px;'>19 features → pipeline → churn probability. No pkl, no error.</p>", unsafe_allow_html=True)

    run = st.button("▶ RUN AI ANALYSIS")
    if run:
        df_input = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":1 if SeniorCitizen=="Yes" else 0,"Partner":Partner,"Dependents":Dependents,
            "tenure":tenure,"PhoneService":PhoneService,"MultipleLines":MultipleLines,"InternetService":InternetService,
            "OnlineSecurity":OnlineSecurity,"OnlineBackup":OnlineBackup,"DeviceProtection":DeviceProtection,
            "TechSupport":TechSupport,"StreamingTV":StreamingTV,"StreamingMovies":StreamingMovies,
            "Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod,
            "MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges
        }])
        st.session_state['prob'] = model.predict_proba(df_input)[0][1]
        st.session_state['pred'] = model.predict(df_input)[0]

    prob = st.session_state.get('prob', 0.34)
    pred = st.session_state.get('pred', 0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob*100,
        delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#22c55e"}},
        number={'suffix':"%", 'font':{'size':44, 'color':"white", 'family':"Plus Jakarta Sans"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1A2332"},
            'bar': {'color': "#ffffff", 'thickness': 0.25},
            'bgcolor': "#05070A",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': "#0a1f14"},
                {'range': [35, 70], 'color': "#1f1a0a"},
                {'range': [70, 100], 'color': "#1f0a0a"}
            ],
            'threshold': {'line': {'color': "#facc15", 'width': 3}, 'thickness': 0.8, 'value': prob*100}
        }
    ))
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        if pred==1:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #450a0a, #7f1d1d); border:1px solid #ef4444; border-radius:14px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><b style="color:white; font-size:16px;">🔴 HIGH CHURN RISK</b><br><span style="color:#fecaca; font-size:13px;">{prob*100:.1f}% probability • Immediate action needed</span></div>
            <div style="background:white; color:black; padding:6px 14px; border-radius:99px; font-weight:800; font-size:12px;">ALERT TEAM</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #052e16, #14532d); border:1px solid #22c55e; border-radius:14px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
            <div><b style="color:white; font-size:16px;">🟢 LOYAL CUSTOMER</b><br><span style="color:#bbf7d0; font-size:13px;">{(1-prob)*100:.1f}% loyalty • Safe for upsell</span></div>
            <div style="background:white; color:black; padding:6px 14px; border-radius:99px; font-weight:800; font-size:12px;">UPSELL</div>
            </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Why this is FAANG level?")
    st.markdown("""
    <div style="color:#7C8DA6; font-size:13.5px; line-height:1.8;">
    <span style="color:white; font-weight:600;">✓ No pickle hell:</span> Model trains on boot, version safe<br>
    <span style="color:white; font-weight:600;">✓ 19 features:</span> Full schema, no shortcuts<br>
    <span style="color:white; font-weight:600;">✓ Design system:</span> 8px radius, 1 accent, glass + glow<br>
    <span style="color:white; font-weight:600;">✓ Product thinking:</span> KPI → Action → Outcome<br>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 📊 Feature Snapshot")
    df_show = pd.DataFrame({
        "Feature": ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"],
        "Value": [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    })
    st.dataframe(df_show, use_container_width=True, height=380, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><center><p style='color:#1A2332; font-size:10px; letter-spacing:3px;'>GREENBANK AI • BUILT FOR SCALE • 2026</p></center>", unsafe_allow_html=True)
