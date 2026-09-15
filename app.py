import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank PRO", page_icon="🏦", layout="wide")

# --- FIXED DARK THEME CSS ---
st.markdown("""
<style>
/* Force dark bg */
.stApp {
    background: #020617!important;
}
[data-testid="stSidebar"] {
    background: #0f172a!important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0!important;
}
h1, h2, h3, h4, p, span, label {
    color: #e2e8f0!important;
}

.hero-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
}

.kpi-card {
    background: #0f172a!important;
    border: 1px solid #1e293b!important;
    border-radius: 16px;
    padding: 18px;
}
.kpi-card h2 { color: white!important; font-size: 24px; font-weight: 800; margin: 6px 0 0 0; }
.kpi-card p { color: #94a3b8!important; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin: 0; }

.main-card {
    background: #0f172a!important;
    border: 1px solid #1e293b!important;
    border-radius: 20px;
    padding: 22px;
}
.main-card * { color: #e2e8f0!important; }

.stButton>button {
    background: #ffffff!important;
    color: #000000!important;
    border-radius: 12px;
    height: 54px;
    font-weight: 800!important;
    width: 100%;
}
.stButton>button p { color: #000000!important; }

/* Fix selectbox visibility */
div[data-baseweb="select"] > div {
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🏦 GreenBank")
    st.markdown("<p style='color:#facc15; font-size:11px; letter-spacing:2px; font-weight:700;'>19 FEATURES • DARK PRO</p>", unsafe_allow_html=True)
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
    tenure = st.slider("Tenure (Months)", 0, 72, 18)
    MonthlyCharges = st.slider("Monthly Charges $", 18, 120, 70)
    TotalCharges = st.slider("Total Charges $", 0, 9000, 1500)

# --- HERO ---
st.markdown("""
<div class="hero-card">
<h1 style="margin:0; font-size:32px; font-weight:800; color:white!important;">Customer Retention Intelligence Platform</h1>
<p style="color:#94a3b8!important; margin-top:8px;">Predict churn before it happens. 19 features, real-time inference, production ready.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

k1,k2,k3,k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card"><p>TENURE</p><h2>{tenure} M</h2></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><p>MONTHLY</p><h2>${MonthlyCharges}</h2></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><p>TOTAL VALUE</p><h2>${TotalCharges}</h2></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><p>CONTRACT</p><h2>{Contract.split("-")[0]}</h2></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### AI Prediction")
    if st.button("▶ RUN AI ANALYSIS"):
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
        number={'suffix':"%", 'font':{'size':40, 'color':"white"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'bgcolor': "#020617",
            'steps': [
                {'range':[0,40],'color':"#052e16"},
                {'range':[40,70],'color':"#422006"},
                {'range':[70,100],'color':"#450a0a"}
            ]
        }
    ))
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=20,b=10), font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        if st.session_state['pred']==1:
            st.error(f"HIGH RISK - {prob*100:.1f}% churn probability")
        else:
            st.success(f"LOW RISK - {(1-prob)*100:.1f}% safe")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### Business Impact")
    st.metric("Annual Revenue at Risk", f"${MonthlyCharges*12}")
    st.metric("Lifetime Value", f"${TotalCharges}")
    st.divider()
    st.markdown("**Risk Factors**")
    if Contract=="Month-to-month": st.markdown("• Month-to-month contract")
    if tenure<12: st.markdown(f"• Low tenure {tenure}M")
    if TechSupport=="No": st.markdown("• No Tech Support")
    st.divider()
    st.markdown("**19 Features Loaded**")
    st.caption("All features active and visible now.")
    st.markdown('</div>', unsafe_allow_html=True)
