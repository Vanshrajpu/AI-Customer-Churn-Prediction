import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Telco Churn Predictor - Final Working")
st.caption("No pkl - Model trains inside app | No AttributeError")

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})

    num_cols = ["tenure","MonthlyCharges","TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(X, y)
    return pipe

model = get_model()
st.success("Model Ready ✅")

c1,c2,c3 = st.columns(3)
with c1:
    tenure = st.slider("tenure",0,72,24)
    MonthlyCharges = st.number_input("MonthlyCharges",70.0)
    TotalCharges = st.number_input("TotalCharges",1500.0)
with c2:
    gender = st.selectbox("gender",["Male","Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen",[0,1])
    Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
with c3:
    InternetService = st.selectbox("InternetService",["DSL","Fiber optic","No"])
    PaymentMethod = st.selectbox("PaymentMethod",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    PaperlessBilling = st.selectbox("PaperlessBilling",["Yes","No"])

if st.button("PREDICT CHURN", use_container_width=True):
    data = pd.DataFrame([{
        "tenure":tenure,"MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges,
        "gender":gender,"SeniorCitizen":SeniorCitizen,"Partner":"Yes","Dependents":"No",
        "PhoneService":"Yes","MultipleLines":"No","InternetService":InternetService,
        "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
        "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
        "Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod
    }])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred==1:
        st.error(f"⚠️ CHURN HOGA - {prob*100:.1f}%")
    else:
        st.success(f"✅ SAFE HAI - {prob*100:.1f}%")

    st.progress(float(prob))
