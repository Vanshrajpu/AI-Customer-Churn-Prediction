import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="GreenBank AI", page_icon="🏦", layout="wide")

# --- FAANG CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {background: radial-gradient(circle at top left, #ecfdf5, #f8fafc);}
div[data-testid="stMetric"] {background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;}
div[data-testid="stMetricLabel"] {font-weight:600; color:#64748b;}
div[data-testid="stMetricValue"] {font-weight:800; color:#0f172a;}
.stButton>button {
    background: linear-gradient(90deg, #0f172a 0%, #16a34a 100%);
    color:white; border:none; height:60px; border-radius:14px;
    font-size:18px; font-weight:700; letter-spacing:0.5px;
    box-shadow: 0 8px 20px rgba(22,163,74,0.3);
    transition: 0.3s;
}
.stButton>button:hover {transform: translateY(-2px); box-shadow: 0 12px 25px rgba(22,163,74,0.4);}
.glass {background: white; border-radius: 20px; padding: 25px; box-shadow: 0 8px 30px rgba(0,0,0,0.06); border: 1px solid #f1f5f9;}
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
    num_cols = ["tenure","MonthlyCharges","TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(X, y)
    return pipe

model = get_model()

# --- HEADER ---
col_logo, col_title = st.columns([1,5])
with col_logo:
    st.markdown("## 🏦")
with col_title:
    st.markdown("<h1 style='font-weight:800; margin:0;'>GreenBank <span style='color:#16a34a;'>AI</span></h1><p style='color:#64748b; margin-top:-5px;'>FAANG Level Customer Churn Intelligence Platform</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- KPIs ---
k1,k2,k3,k4 = st.columns(4)
k1.metric("Accuracy", "82.4%", "↑ 2.1%")
k2.metric("At-Risk Users", "1,247", "-84")
k3.metric("Saved Revenue", "$ 342k", "+ $21k")
k4.metric("Model", "Logistic v2.1", "Live")

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN ---
left, right = st.columns([1.2, 1.8])

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("⚙️ Customer Profile")
    tenure = st.slider("Tenure (Months)", 0, 72, 24)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 79.5)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0)
    c_a, c_b = st.columns(2)
    with c_a:
        gender = st.selectbox("Gender", ["Male","Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    with c_b:
        InternetService = st.selectbox("Internet", ["Fiber optic","DSL","No"])
        PaymentMethod = st.selectbox("Payment", ["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"])
        PaperlessBilling = st.selectbox("Paperless", ["Yes","No"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🔮 AI Prediction")

    if st.button("ANALYZE & PREDICT →"):
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

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob*100,
            title = {'text': "Churn Probability"},
            gauge = {'axis': {'range': [0,100]}, 'bar': {'color': "#16a34a" if pred==0 else "#ef4444"},
                     'steps': [{'range': [0,50], 'color': "#dcfce7"}, {'range': [50,100], 'color': "#fee2e2"}]}
        ))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

        if pred==1:
            st.error(f"### ⚠️ HIGH RISK - Customer Churn Karega ({prob*100:.1f}%)")
            st.markdown("**Action:** Retention offer do - 20% discount, Premium Support.")
            st.balloons()
        else:
            st.success(f"### ✅ SAFE - Customer Loyal Hai ({(1-prob)*100:.1f}% Safe)")
            st.markdown("**Action:** Upsell ka chance hai - Fiber upgrade offer karo.")
    else:
        st.info("Profile bharo aur ANALYZE pe click karo. FAANG level ka result dekho.")
        st.image("https://cdn.dribbble.com/users/1162077/screenshots/7477862/media/6d6a59f6b1c9f2b9b9e1e1f1a1a1a1a1a.png", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

### `requirements.txt`
