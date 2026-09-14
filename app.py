import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Churn Predictor", page_icon="📉", layout="wide")

st.markdown("""
<style>
.stButton>button {width:100%; background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%); color:white; border-radius:10px; height:50px; font-size:18px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["customerID", "Churn"]]
    X = df[num_cols + cat_cols]
    y = df["Churn"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X, y)
    return pipe

st.title("📉 AI Customer Churn Prediction")
st.caption("Customer bhaagega ya rahega? AI se pata karo - by Vansh Rajput")
st.divider()

model = get_model()

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
    PaymentMethod = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    st.subheader("💰 Billing")
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()
if st.button("🔮 PREDICT CHURN"):
    input_data = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges,
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": TechSupport, "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": Contract, "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod
    }])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    churn_p = prob[list(model.classes_).index('Yes')]

    if pred == "Yes":
        st.error(f"⚠️ CHURN HOGA - {churn_p*100:.1f}% chance")
        st.warning("Action: Discount / Offer deke roko")
    else:
        st.success(f"✅ SAFE HAI - {100-churn_p*100:.1f}% safe")
    st.progress(float(churn_p))
    st.metric("Churn Probability", f"{churn_p*100:.2f}%")
