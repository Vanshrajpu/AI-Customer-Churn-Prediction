import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank AI PRO", layout="wide")

st.markdown("""
<style>
.stApp {background:#0d0d0d;}
.gold-card {background:#1a1a1a; border:1px solid #d4af3766; border-radius:16px; padding:20px;}
.stButton>button {background:linear-gradient(90deg,#fcd34d,#f59e0b); color:black!important; font-weight:800; height:60px; width:100%; border-radius:12px;}
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
st.success("✅ Model Ready - No pkl needed")

st.title("GreenBank AI PRO 🇮🇳")

tenure = st.slider("Tenure",0,72,18)
mc = st.slider("Monthly Charges $",20,120,70)
tc = st.slider("Total Charges $",0,10000,1500)
Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
InternetService = st.selectbox("Internet",["Fiber optic","DSL","No"])

if st.button("✨ RUN AI ANALYSIS"):
    data = pd.DataFrame([{
        "tenure":tenure,"MonthlyCharges":mc,"TotalCharges":tc,"gender":"Male","SeniorCitizen":0,
        "Partner":"Yes","Dependents":"No","PhoneService":"Yes","MultipleLines":"No",
        "InternetService":InternetService,"OnlineSecurity":"No","OnlineBackup":"No",
        "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
        "Contract":Contract,"PaperlessBilling":"Yes","PaymentMethod":"Electronic check"
    }])
    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text':"Churn Risk %"}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color':"#fde68a"})
    st.plotly_chart(fig, use_container_width=True)
    if pred==1: st.error(f"🔴 Churn Hoga! {prob*100:.1f}%")
    else: st.success(f"🟢 Safe Hai! {(1-prob)*100:.1f}%")
