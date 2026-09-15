import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank PRO MAX", page_icon="🏦", layout="wide")


st.markdown("""
<style>
.stApp { background: #020617!important; }
[data-testid="stSidebar"] { background: #0f172a!important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0!important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 28px;
}

/* KPI Cards with color top border */
.kpi {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 16px 20px;
    border-top: 3px solid;
}
.kpi h2 { color: white!important; margin: 4px 0 0 0; font-size: 26px; font-weight: 800; }
.kpi p { color: #94a3b8!important; margin: 0; font-size: 10px; letter-spacing: 1px; font-weight: 700; text-transform: uppercase;}

.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 22px;
}
.card * { color: #e2e8f0!important; }

.stButton>button {
    background: white!important;
    color: black!important;
    height: 52px;
    border-radius: 12px;
    font-weight: 800!important;
    width: 100%;
}

/* Fix all inputs visibility */
div[data-baseweb="select"] > div,.stNumberInput input,.stTextInput input {
    background-color: #1e293b!important;
    color: white!important;
    border: 1px solid #334155!important;
}
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

with st.sidebar:
    st.markdown("## 🏦 GreenBank")
    st.markdown("<p style='color:#facc15; font-size:11px; letter-spacing:2px; font-weight:700;'>PRO MAX • 19 FEATURES</p>", unsafe_allow_html=True)
    st.divider()
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["No","Yes"])
    PhoneService = st.selectbox("Phone Service", ["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    InternetService = st.selectbox("Internet", ["Fiber optic","DSL","No"])
    OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    tenure = st.slider("Tenure", 0, 72, 18)
    MonthlyCharges = st.slider("Monthly Charges $", 18, 120, 70)
    TotalCharges = st.slider("Total Charges $", 0, 9000, 1500)

st.markdown("""
<div class="hero">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <p style="color:#22c55e!important; font-size:11px; font-weight:800; letter-spacing:2px; margin:0;">● LIVE AI SYSTEM</p>
            <h1 style="color:white!important; font-size:32px; font-weight:800; margin:6px 0 0 0; line-height:1.1;">Customer Retention<br>Intelligence Platform</h1>
            <p style="color:#94a3b8!important; margin-top:8px; font-size:13px;">19 features • Real-time prediction • Business impact analysis</p>
        </div>
        <div style="font-size:60px;">🏦</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

c1,c2,c3,c4 = st.columns(4)
c1.markdown(f'<div class="kpi" style="border-top-color:#22c55e;"><p>TENURE</p><h2>{tenure} M</h2><p style="color:#22c55e!important;">● Active</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi" style="border-top-color:#3b82f6;"><p>MONTHLY</p><h2>${MonthlyCharges}</h2><p style="color:#3b82f6!important;">MRR</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi" style="border-top-color:#f59e0b;"><p>TOTAL VALUE</p><h2>${TotalCharges}</h2><p style="color:#f59e0b!important;">LTV</p></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi" style="border-top-color:#8b5cf6;"><p>CONTRACT</p><h2>{Contract.split("-")[0]}</h2><p style="color:#8b5cf6!important;">{PaymentMethod.split(" ")[0]}</p></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
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
        st.session_state['prob'] = model.predict_proba(df_input)[0][1]
        st.session_state['pred'] = model.predict(df_input)[0]

    prob = st.session_state.get('prob', 0.35)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        number={'suffix':"%", 'font':{'size':44, 'color':"white"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor':"white"},
            'bar': {'color': "white"},
            'bgcolor': "#020617",
            'steps': [{'range':[0,40],'color':"#052e16"},{'range':[40,70],'color':"#422006"},{'range':[70,100],'color':"#450a0a"}]
        }
    ))
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        p = st.session_state['prob']
        if st.session_state['pred']==1:
            st.error(f"🔴 HIGH RISK - Churn Hoga! {p*100:.1f}%")
        else:
            st.success(f"🟢 LOW RISK - Safe Hai! {(1-p)*100:.1f}% Loyalty")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 💼 Business Impact")
    st.metric("Annual Risk", f"${MonthlyCharges*12}", delta=f"-${MonthlyCharges*12} if churned", delta_color="inverse")
    st.metric("Customer LTV", f"${TotalCharges}")
    st.divider()
    st.markdown("#### 🔍 Risk Factors")
    risks = []
    if Contract=="Month-to-month": risks.append("⚠️ Month-to-month contract")
    if tenure<12: risks.append(f"⚠️ Low tenure ({tenure}M)")
    if TechSupport=="No": risks.append("⚠️ No Tech Support")
    if PaymentMethod=="Electronic check": risks.append("⚠️ Electronic check")
    if risks:
        for r in risks: st.markdown(f"<p style='color:#fbbf24!important; font-size:13px;'>{r}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#22c55e!important;'>✅ No major risks</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 📋 19 Features Summary")
    st.dataframe(pd.DataFrame({"Feature":["Contract","Tenure","Monthly","Total"],"Value":[Contract,tenure,MonthlyCharges,TotalCharges]}), hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
