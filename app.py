import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GreenBank - Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
   .stApp { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    h1, h2, h3, h4 { color: #0f172a; }
   .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Load data
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Define preprocessing
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Create pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X, y)
    return model

model = load_model()

# --- SIDEBAR - ALL 19 FEATURES ---
with st.sidebar:
    st.title("🏦 GreenBank")
    st.markdown("**Customer Churn Prediction**")
    st.divider()

    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.subheader("Services")
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.subheader("Billing")
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

    st.divider()
    st.caption("Model trained on 7043 customers")

# --- MAIN CONTENT ---
st.title("Customer Churn Prediction System")
st.markdown("Predict whether a customer will churn using 19 customer attributes and a production-ready ML pipeline.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tenure", f"{tenure} months")
col2.metric("Monthly Charges", f"${MonthlyCharges}")
col3.metric("Total Charges", f"${TotalCharges}")
col4.metric("Contract", Contract)

st.write("")

left, right = st.columns([2, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Prediction")

        if st.button("Run Prediction", type="primary", use_container_width=True):
            input_data = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }])

            prob = model.predict_proba(input_data)[0][1]
            pred = model.predict(input_data)[0]

            st.session_state["prob"] = prob
            st.session_state["pred"] = pred

        prob = st.session_state.get("prob", 0.35)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f172a"},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#fee2e2"}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if "prob" in st.session_state:
            if st.session_state["pred"] == 1:
                st.error(f"Result: High risk of churn ({st.session_state['prob']:.1%}) - Retention action recommended.")
            else:
                st.success(f"Result: Low risk of churn ({1-st.session_state['prob']:.1%} retention) - Customer is stable.")

with right:
    with st.container(border=True):
        st.subheader("Business Analysis")

        annual_value = MonthlyCharges * 12
        st.metric("Annual Revenue at Risk", f"${annual_value:.0f}")

        st.divider()
        st.markdown("**Key Risk Factors**")
        risks = []
        if Contract == "Month-to-month":
            risks.append("Month-to-month contract")
        if tenure < 12:
            risks.append(f"Low tenure ({tenure} months)")
        if TechSupport == "No" and InternetService!= "No":
            risks.append("No tech support")
        if PaymentMethod == "Electronic check":
            risks.append("Electronic check payment")

        if risks:
            for r in risks:
                st.write(f"• {r}")
        else:
            st.write("No major risk factors found.")

        st.divider()
        st.markdown("**Input Summary (19 features)**")
        summary_df = pd.DataFrame({
            "Feature": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
            "Value": [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True, height=350)
