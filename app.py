import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    elif os.path.exists("churn.pkl"):
        return joblib.load("churn.pkl")
    else:
        return None

model = load_model()

if model is None:
    st.error("model.pkl GitHub pe nahi mila! File ko main folder me upload kar.")
    st.stop()

st.title("📉 Telco Customer Churn Predictor")
st.success(f"Model Loaded: {type(model.named_steps['classifier']).__name__} | Accuracy ~75%")
st.divider()

# --- UI ---
c1,c2,c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (months)",0,72,24)
    MonthlyCharges = st.number_input("MonthlyCharges",20.0,200.0,70.7)
    TotalCharges = st.number_input("TotalCharges",0.0,10000.0,1500.0)
    SeniorCitizen = st.selectbox("SeniorCitizen",[0,1])
with c2:
    gender = st.selectbox("gender",["Male","Female"])
    Partner = st.selectbox("Partner",["Yes","No"])
    Dependents = st.selectbox("Dependents",["Yes","No"])
    PhoneService = st.selectbox("PhoneService",["Yes","No"])
with c3:
    InternetService = st.selectbox("InternetService",["DSL","Fiber optic","No"])
    Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    PaymentMethod = st.selectbox("PaymentMethod",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    PaperlessBilling = st.selectbox("PaperlessBilling",["Yes","No"])

if st.button("🔮 PREDICT CHURN", use_container_width=True):
    # Tere model ke liye saare columns chahiye
    input_df = pd.DataFrame([{
        "tenure":tenure,"MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges,
        "SeniorCitizen":SeniorCitizen,"gender":gender,"Partner":Partner,"Dependents":Dependents,
        "PhoneService":PhoneService,"MultipleLines":"No","InternetService":InternetService,
        "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
        "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
        "Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod
    }])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        churn_prob = prob[1] if len(prob)>1 else prob[0]

        # Tere mapping ke hisab se: Yes=1 / No=2 (ya No=0)
        if str(pred) == "1" or pred == 1:
            st.error(f"⚠️ CHURN HOGA - Customer Chhod dega ({churn_prob*100:.1f}%)")
        else:
            st.success(f"✅ SAFE HAI - Churn nahi hoga ({(1-churn_prob)*100:.1f}% Safe)")

        st.progress(float(churn_prob))

    except Exception as e:
        st.error(f"Error: {e}")
