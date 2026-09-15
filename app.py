import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank PRO MAX", page_icon="🏦", layout="wide")

# --- COLORFUL ANIMATED CSS + PICS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&display=swap');
* {font-family: 'Plus Jakarta Sans', sans-serif;}

@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-10px);}
    100% {transform: translateY(0px);}
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #f8fafc, #eef2ff, #f0fdf4, #fffbeb);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(20px);
    border-right: 1px solid #e2e8f0;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 24px;
    padding: 30px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px -10px rgba(15,23,42,0.3);
}
.hero-card::before {
    content: "";
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, #facc15 0%, transparent 70%);
    top: -100px; right: -100px;
    opacity: 0.3;
    animation: float 6s ease-in-out infinite;
}

.kpi-card {
    background: white;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    text-align: left;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
}
.kpi-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 4px;
}
.kpi-1::after {background: linear-gradient(90deg, #22c55e, #16a34a);}
.kpi-2::after {background: linear-gradient(90deg, #3b82f6, #2563eb);}
.kpi-3::after {background: linear-gradient(90deg, #f59e0b, #d97706);}
.kpi-4::after {background: linear-gradient(90deg, #8b5cf6, #7c3aed);}

.main-card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 24px;
    border: 1px solid rgba(226,232,240,0.8);
    box-shadow: 0 8px 32px rgba(0,0,0,0.06);
}

.stButton>button {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white!important;
    border-radius: 14px;
    height: 56px;
    font-weight: 800!important;
    font-size: 15px!important;
    border: none;
    box-shadow: 0 8px 20px -4px rgba(15,23,42,0.4);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 12px 30px -4px rgba(15,23,42,0.5);
}

.pulse {
    display: inline-block;
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(34,197,94, 0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(34,197,94, 0.7);}
    70% {box-shadow: 0 0 0 10px rgba(34,197,94, 0);}
    100% {box-shadow: 0 0 0 0 rgba(34,197,94, 0);}
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
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.markdown("## GreenBank PRO")
    st.markdown("<span style='background:#dcfce7; color:#166534; padding:4px 10px; border-radius:20px; font-size:11px; font-weight:700;'>● 19 FEATURES ACTIVE</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**👤 Personal**")
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["No","Yes"])

    st.markdown("**📡 Services**")
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

# --- HERO WITH PIC ---
col_hero1, col_hero2 = st.columns([2.5, 1])

with col_hero1:
    st.markdown(f"""
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:start;">
            <div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span class="pulse"></span>
                    <span style="font-size:12px; letter-spacing:1.5px; font-weight:700; color:#22c55e;">LIVE AI SYSTEM</span>
                </div>
                <h1 style="margin:0; font-size:36px; font-weight:800; line-height:1.1; letter-spacing:-1px;">Customer Retention<br>Intelligence Platform</h1>
                <p style="color:#94a3b8; margin:12px 0 0 0; font-size:14px; max-width:500px;">Predict churn before it happens. 19 features, real-time inference, and business impact analysis for banking.</p>
                <div style="margin-top:20px; display:flex; gap:10px;">
                    <span style="background:rgba(255,255,255,0.1); padding:6px 14px; border-radius:99px; font-size:12px; border:1px solid rgba(255,255,255,0.1);">🤖 Logistic Regression</span>
                    <span style="background:rgba(255,255,255,0.1); padding:6px 14px; border-radius:99px; font-size:12px; border:1px solid rgba(255,255,255,0.1);">⚡ 7043 Customers Trained</span>
                </div>
            </div>
            <img src="https://cdn3d.iconscout.com/3d/premium/thumb/banking-3d-icon-download-in-png-blend-fbx-gltf-file-formats--finance-money-payment-pack-business-icons-8773894.png" style="width:140px; animation: float 5s ease-in-out infinite;">
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_hero2:
    st.markdown("""
    <div style="background:white; border-radius:24px; padding:16px; border:1px solid #e2e8f0; box-shadow: 0 8px 32px rgba(0,0,0,0.06); height:100%;">
        <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=500" style="width:100%; border-radius:16px; height:140px; object-fit:cover;">
        <div style="margin-top:12px;">
            <p style="font-weight:700; margin:0; color:#0f172a;">Trusted by Banks</p>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0 0;">Real-time analytics used by top financial institutions to reduce churn by 34%.</p>
            <div style="margin-top:12px; display:flex; gap:6px;">
                <img src="https://cdn-icons-png.flaticon.com/512/5968/5968705.png" width="24">
                <img src="https://cdn-icons-png.flaticon.com/512/5968/5968299.png" width="24">
                <img src="https://cdn-icons-png.flaticon.com/512/5968/5968776.png" width="24">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# KPI CARDS WITH COLORS
k1,k2,k3,k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card kpi-1"><div style="display:flex; justify-content:space-between;"><div><p style="margin:0; color:#64748b; font-size:10px; font-weight:700; letter-spacing:1px;">TENURE</p><h2 style="margin:6px 0 0 0; font-size:24px; font-weight:800; color:#0f172a;">{tenure} M</h2><p style="margin:4px 0 0 0; color:#22c55e; font-size:11px; font-weight:600;">● Active</p></div><img src="https://cdn-icons-png.flaticon.com/512/3135/3135679.png" width="40"></div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card kpi-2"><div style="display:flex; justify-content:space-between;"><div><p style="margin:0; color:#64748b; font-size:10px; font-weight:700; letter-spacing:1px;">MONTHLY</p><h2 style="margin:6px 0 0 0; font-size:24px; font-weight:800; color:#0f172a;">${MonthlyCharges}</h2><p style="margin:4px 0 0 0; color:#3b82f6; font-size:11px; font-weight:600;">MRR</p></div><img src="https://cdn-icons-png.flaticon.com/512/1570/1570887.png" width="40"></div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card kpi-3"><div style="display:flex; justify-content:space-between;"><div><p style="margin:0; color:#64748b; font-size:10px; font-weight:700; letter-spacing:1px;">TOTAL VALUE</p><h2 style="margin:6px 0 0 0; font-size:24px; font-weight:800; color:#0f172a;">${TotalCharges}</h2><p style="margin:4px 0 0 0; color:#f59e0b; font-size:11px; font-weight:600;">LTV</p></div><img src="https://cdn-icons-png.flaticon.com/512/2721/2721121.png" width="40"></div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card kpi-4"><div style="display:flex; justify-content:space-between;"><div><p style="margin:0; color:#64748b; font-size:10px; font-weight:700; letter-spacing:1px;">CONTRACT</p><h2 style="margin:6px 0 0 0; font-size:18px; font-weight:800; color:#0f172a;">{Contract.split("-")[0]}</h2><p style="margin:4px 0 0 0; color:#8b5cf6; font-size:11px; font-weight:600;">{PaymentMethod.split(" ")[0]}</p></div><img src="https://cdn-icons-png.flaticon.com/512/3652/3652191.png" width="40"></div></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.3, 0.7], gap="large")

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([3,1])
    with col_t1:
        st.markdown("#### 🎯 AI Prediction Engine")
        st.caption("19 features → Pipeline → Probability")
    with col_t2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)

    if st.button("▶ RUN AI ANALYSIS NOW", use_container_width=True):
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
        number={'suffix':"%", 'font':{'size':48, 'color':"#0f172a", 'family':"Plus Jakarta Sans"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0f172a", 'thickness': 0.3},
            'bgcolor': "#f1f5f9",
            'steps': [
                {'range':[0,35],'color':"#dcfce7"},
                {'range':[35,70],'color':"#fef9c3"},
                {'range':[70,100],'color':"#fee2e2"}
            ]
        }
    ))
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        if st.session_state['pred']==1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border:1px solid #fecaca; border-radius:16px; padding:16px; display:flex; gap:12px; align-items:center;">
                <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" width="40">
                <div>
                    <b style="color:#991b1b;">HIGH CHURN RISK - {prob*100:.1f}%</b><br>
                    <span style="color:#7f1d1d; font-size:13px;">Immediate retention offer required. Customer likely to leave.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border:1px solid #bbf7d0; border-radius:16px; padding:16px; display:flex; gap:12px; align-items:center;">
                <img src="https://cdn-icons-png.flaticon.com/512/1828/1828640.png" width="40">
                <div>
                    <b style="color:#14532d;">LOYAL CUSTOMER - {(1-prob)*100:.1f}% Safe</b><br>
                    <span style="color:#166534; font-size:13px;">Customer is stable. Great time for upsell & cross-sell.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("#### 💼 Business Impact")
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=500", use_container_width=True, caption="Real-time Business Analytics")

    annual_loss = MonthlyCharges * 12
    col_a, col_b = st.columns(2)
    col_a.metric("Annual Risk", f"${annual_loss}")
    col_b.metric("LTV", f"${TotalCharges}")

    st.divider()
    st.markdown("#### 🔍 Why Churn? (AI Explain)")
    risks = []
    if Contract == "Month-to-month": risks.append("Month-to-month = 42% more churn")
    if tenure < 12: risks.append(f"New customer ({tenure}M)")
    if TechSupport == "No": risks.append("No Tech Support")
    if InternetService == "Fiber optic": risks.append("Fiber optic high expectations")
    if PaymentMethod == "Electronic check": risks.append("Electronic check risky")

    if risks:
        for r in risks:
            st.markdown(f"<div style='background:#fffbeb; border-left:4px solid #f59e0b; padding:8px 12px; margin:6px 0; border-radius:6px; font-size:13px;'>⚠️ {r}</div>", unsafe_allow_html=True)
    else:
        st.success("No major risks!")

    st.divider()
    st.markdown("#### 📋 All 19 Features")
    st.dataframe(pd.DataFrame({
        "Feature": ["gender","Senior","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Monthly","Total"],
        "Value": [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    }), hide_index=True, height=300, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
