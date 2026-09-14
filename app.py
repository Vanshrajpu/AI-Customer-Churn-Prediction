import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Churn Predictor", page_icon="📉", layout="wide")

# --- Custom CSS for Sahi UI ---
st.markdown("""
<style>
.big-font {font-size:22px!important; font-weight:600;}
.stButton>button {width:100%; background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%); color:white; border-radius:10px; height:50px; font-size:18px;}
</style>
""", unsafe_allow_html=True)

# --- Load Model (Error-Proof) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn.pkl")
        return model
    except Exception as e:
        st.error(f"Model load me error: {e}. requirements.txt check kar.")
        return None

model = load_model()

st.title("📉 AI Customer Churn Prediction")
st.markdown("**Customer bhaagega ya rahega? AI se pata karo**")
st.divider()

# --- Inputs in 3 Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("📞 Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    st.subheader("💰 Billing & Add-ons")
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# --- Predict ---
if st.button("🔮 PREDICT CHURN"):
    if model is None:
        st.stop()

    input_data = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges,
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod
    }])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    churn_prob = proba[1] if model.classes_[0]==0 or model.classes_[0]=='No' else proba[0]
    # Handle if classes are ['No','Yes']
    if hasattr(model, 'classes_') and 'Yes' in model.classes_:
        churn_prob = proba[list(model.classes_).index('Yes')]

    st.markdown("### Result")
    c1, c2 = st.columns(2)

    with c1:
        if pred == 1 or pred == "Yes":
            st.error(f"⚠️ **CHURN HOGA** - Customer bhaag jayega!")
        else:
            st.success(f"✅ **SAFE HAI** - Customer nahi bhaagega")

    with c2:
        st.metric("Churn Probability", f"{churn_prob*100:.2f}%")
        st.progress(float(churn_prob))

    if churn_prob > 0.6:
        st.warning("💡 **Action:** Is customer ko retention offer do - Discount / Free Upgrade / Loyalty Bonus")
