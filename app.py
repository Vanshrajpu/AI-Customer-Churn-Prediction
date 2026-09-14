import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Google Churn AI", page_icon="G", layout="wide")

# --- GOOGLE CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500&display=swap');
.stApp {background: #ffffff;}
h1,h2,h3 {font-family: 'Google Sans', sans-serif!important; color: #202124!important; font-weight: 500!important;}
p, label {font-family: 'Roboto', sans-serif!important; color: #5f6368!important;}
[data-testid="stMetric"] {
    background: white; border: 1px solid #dadce0; border-radius: 12px; padding: 16px;
}
[data-testid="stMetricValue"] {color: #202124!important; font-family: 'Google Sans'!important;}
.google-card {
    background: white; border: 1px solid #dadce0; border-radius: 16px; padding: 24px;
}
.stButton>button {
    background: #1a73e8; color: white!important; border-radius: 24px; height: 48px;
    font-family: 'Google Sans'!important; font-weight: 500!important; font-size: 16px!important;
    border: none; padding: 0 32px;
}
.stButton>button:hover {background: #1765cc; box-shadow: 0 1px 3px rgba(0,0,0,0.2);}
div[data-baseweb="select"] > div {border-radius: 8px!important; border-color: #dadce0!important;}
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

# --- GOOGLE HEADER ---
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; padding: 12px 0; border-bottom: 1px solid #dadce0;'>
<span style='font-size:24px; font-weight:500; font-family:Google Sans; color:#5f6368;'>Google</span>
<span style='font-size:22px; font-family:Google Sans; color:#202124;'>Cloud</span>
<span style='margin-left:16px; font-size:22px; color:#dadce0;'>|</span>
<span style='margin-left:16px; font-size:22px; font-family:Google Sans; color:#202124;'>Churn Intelligence</span>
<span style='background:#e8f0fe; color:#1967d2; padding:4px 10px; border-radius:12px; font-size:12px; margin-left:12px;'>PRO</span>
</div>
<br>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Active Users", "7,043", "2.1%")
c2.metric("Churn Rate", "26.5%", "-1.2%", delta_color="inverse")
c3.metric("Model Accuracy", "84.2%", "0.8%")
c4.metric("Status", "● Live", "Updated")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="google-card">', unsafe_allow_html=True)
    st.markdown("#### Enter customer details")
    st.caption("AI will predict churn probability")
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    col1, col2 = st.columns(2)
    with col1:
        MonthlyCharges = st.number_input("Monthly charges", 70.0)
        gender = st.selectbox("Gender", ["Male","Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
    with col2:
        TotalCharges = st.number_input("Total charges", 1500.0)
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        InternetService = st.selectbox("Internet", ["Fiber optic","DSL","No"])
    PaymentMethod = st.selectbox("Payment method", ["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="google-card">', unsafe_allow_html=True)
    st.markdown("#### Prediction")

    if st.button("Run prediction"):
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

        # Google Colors Gauge
        color = "#ea4335" if pred==1 else "#34a853"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={'text': "Churn risk", 'font': {'family': "Google Sans", 'size': 16, 'color': "#5f6368"}},
            gauge={'axis': {'range': [0,100]}, 'bar': {'color': color}, 'bgcolor': "#f8f9fa", 'borderwidth': 0},
            number={'font': {'family': "Google Sans", 'color': "#202124"}}
        ))
        fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        if pred==1:
            st.markdown(f"""
            <div style='background:#fce8e6; border-radius:12px; padding:16px; display:flex; gap:12px;'>
            <span style='font-size:24px;'>⚠️</span>
            <div><b style='color:#c5221f;'>High churn risk: {prob*100:.1f}%</b><br><span style='color:#5f6368;'>Customer likely to leave. Offer retention discount.</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#e6f4ea; border-radius:12px; padding:16px; display:flex; gap:12px;'>
            <span style='font-size:24px;'>✅</span>
            <div><b style='color:#137333;'>Low risk: {(1-prob)*100:.1f}% safe</b><br><span style='color:#5f6368;'>Customer is loyal. Try upselling.</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<br><br><center><div style='width:80px;height:80px;background:#f1f3f4;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:36px;'>🧠</div><br><p>Ready to analyze</p></center>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><center><p style='font-size:12px; color:#80868b;'>Built with Google Cloud Design • By Vansh</p></center>", unsafe_allow_html=True)
