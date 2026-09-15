import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank - Retention Intelligence", layout="wide")

# --- CLEAN ENTERPRISE CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {background: #F8FAFC;}
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
}
.stButton>button {
    background: #0F172A; color: #fff!important; border-radius: 10px;
    height: 50px; font-weight: 600!important; border: 0; width: 100%;
}
.stButton>button:hover {background: #1E293B;}
.badge {background: #F1F5F9; color: #0F172A; padding: 4px 10px; border-radius: 99px; font-size: 11px; font-weight: 600; border: 1px solid #E2E8F0;}
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🏦 GreenBank")
    st.caption("Retention Intelligence Platform")
    st.divider()

    tab1, tab2 = st.tabs(["Personal", "Services & Billing"])
    with tab1:
        gender = st.selectbox("Gender", ["Male","Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["No","Yes"])
        tenure = st.slider("Tenure (Months)", 0, 72, 18)
        MonthlyCharges = st.slider("Monthly Charges", 18, 120, 70)
        TotalCharges = st.slider("Total Charges", 0, 9000, 1500)
    with tab2:
        PhoneService = st.selectbox("Phone Service", ["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        InternetService = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

# --- HEADER ---
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
<div>
<h1 style="font-size: 28px; margin: 0; color: #0F172A;">Customer Churn Prediction</h1>
<p style="color: #64748B; margin: 4px 0 0 0;">19 features • Production pipeline • Explainable AI</p>
</div>
<div style="display:flex; gap:8px;">
<span class="badge">19 FEATURES</span>
<span class="badge">LOGISTIC REGRESSION</span>
</div>
</div>
""", unsafe_allow_html=True)

# KPIs
k1,k2,k3,k4 = st.columns(4)
k1.markdown(f'<div class="kpi"><p style="margin:0; color:#64748B; font-size:11px; font-weight:600;">TENURE</p><h3 style="margin:4px 0 0 0;">{tenure} months</h3></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><p style="margin:0; color:#64748B; font-size:11px; font-weight:600;">MRR</p><h3 style="margin:4px 0 0 0;">${MonthlyCharges}</h3></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><p style="margin:0; color:#64748B; font-size:11px; font-weight:600;">LTV</p><h3 style="margin:4px 0 0 0;">${TotalCharges}</h3></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><p style="margin:0; color:#64748B; font-size:11px; font-weight:600;">CONTRACT</p><h3 style="margin:4px 0 0 0;">{Contract}</h3></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Prediction Result")

    if st.button("Run Prediction"):
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

    prob = st.session_state.get('prob', 0.32)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        number={'suffix':"%", 'font':{'size':36, 'color':"#0F172A"}},
        title={'text':"Churn Probability", 'font':{'size':14, 'color':"#64748B"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0F172A"},
            'bgcolor': "#F1F5F9",
            'steps': [
                {'range': [0, 40], 'color': "#DCFCE7"},
                {'range': [40, 70], 'color': "#FEF9C3"},
                {'range': [70, 100], 'color': "#FEE2E2"}
            ]
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="white", margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        p = st.session_state['prob']
        if st.session_state['pred']==1:
            st.error(f"High churn risk: {p:.1%} probability. Action: Initiate retention workflow.")
        else:
            st.success(f"Low churn risk: {(1-p):.1%} retention probability. Action: Eligible for upsell.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Risk Analysis")

    prob_val = st.session_state.get('prob', 0.32)
    annual_loss = MonthlyCharges * 12

    st.metric("Potential Revenue at Risk (Annual)", f"${annual_loss:.0f}")
    st.metric("Current Customer Value", f"${TotalCharges}")

    st.divider()
    st.markdown("**Key Risk Drivers**")
    risks = []
    if Contract == "Month-to-month": risks.append("Month-to-month contract")
    if tenure < 12: risks.append(f"Low tenure ({tenure} months)")
    if TechSupport == "No": risks.append("No tech support")
    if OnlineSecurity == "No": risks.append("No online security")
    if PaymentMethod == "Electronic check": risks.append("Electronic check payment")

    if risks:
        for r in risks:
            st.markdown(f"- {r}")
    else:
        st.markdown("No critical risk factors identified.")

    st.divider()
    st.markdown("**All Features (19)**")
    st.caption(f"Gender: {gender}, Senior: {SeniorCitizen}, Partner: {Partner}, Dependents: {Dependents}, Contract: {Contract}")
    st.caption(f"Internet: {InternetService}, Phone: {PhoneService}, Tenure: {tenure}M, Charges: ${MonthlyCharges}/${TotalCharges}")

    st.markdown('</div>', unsafe_allow_html=True)
