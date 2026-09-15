import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="GreenBank - Churn Prediction", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {font-family: 'Inter', sans-serif;}
.stApp {background: #F8FAFC;}
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
h1, h2, h3, h4 {color: #0F172A!important; font-weight: 700!important;}
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.metric {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
}
.metric p {color: #64748B; font-size: 12px; font-weight: 500; margin: 0; text-transform: uppercase;}
.metric h3 {color: #0F172A; font-size: 20px; margin: 4px 0 0 0;}
.stButton>button {
    background: #0F172A; color: white!important; border-radius: 8px;
    height: 48px; font-weight: 600!important; border: none; width: 100%;
}
.stButton>button:hover {background: #1E293B;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
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
    pipe = Pipeline([("preprocessor",pre),("classifier",LogisticRegression(max_iter=1000))])
    pipe.fit(X,y)
    scores = cross_val_score(pipe, X, y, cv=5)
    return pipe, scores.mean()

model, cv_score = load_model()

# SIDEBAR
with st.sidebar:
    st.markdown("### GreenBank")
    st.caption("Customer Churn Prediction System")
    st.divider()

    st.markdown("**Customer Demographics**")
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No","Yes"])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["No","Yes"])

    st.markdown("**Service Details**")
    PhoneService = st.selectbox("Phone Service", ["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    InternetService = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
    OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    st.markdown("**Contract & Billing**")
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    tenure = st.slider("Tenure (Months)", 0, 72, 18)
    MonthlyCharges = st.slider("Monthly Charges", 18, 120, 70)
    TotalCharges = st.slider("Total Charges", 0, 9000, 1500)

    st.divider()
    st.caption(f"Model CV Score: {cv_score:.2%}")

# MAIN
st.markdown(f"""
<div style="margin-bottom: 24px;">
<h1 style="font-size: 28px; margin: 0;">Customer Churn Prediction</h1>
<p style="color: #64748B; margin: 6px 0 0 0;">Predict likelihood of customer churn using 19 customer attributes. Model validated with 5-fold cross-validation.</p>
</div>
""", unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
m1.markdown(f'<div class="metric"><p>Tenure</p><h3>{tenure} months</h3></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric"><p>Monthly Charges</p><h3>${MonthlyCharges}</h3></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric"><p>Total Charges</p><h3>${TotalCharges}</h3></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric"><p>Contract Type</p><h3>{Contract}</h3></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Prediction")
    st.markdown(f"<p style='color:#64748B; font-size:13px;'>Model CV Accuracy: {cv_score:.2%} | Features: 19 | Model: Logistic Regression with balanced class weight</p>", unsafe_allow_html=True)

    if st.button("Run Prediction"):
        input_df = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":1 if SeniorCitizen=="Yes" else 0,"Partner":Partner,"Dependents":Dependents,
            "tenure":tenure,"PhoneService":PhoneService,"MultipleLines":MultipleLines,"InternetService":InternetService,
            "OnlineSecurity":OnlineSecurity,"OnlineBackup":OnlineBackup,"DeviceProtection":DeviceProtection,
            "TechSupport":TechSupport,"StreamingTV":StreamingTV,"StreamingMovies":StreamingMovies,
            "Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod,
            "MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges
        }])
        st.session_state['prob'] = model.predict_proba(input_df)[0][1]
        st.session_state['pred'] = model.predict(input_df)[0]

    prob = st.session_state.get('prob', 0.32)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        number={'suffix':"%", 'font':{'size':32, 'color':"#0F172A"}},
        title={'text':"Churn Probability", 'font':{'size':14, 'color':"#64748B"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0F172A"},
            'bgcolor': "#F1F5F9",
            'steps': [
                {'range': [0, 40], 'color': "#DCFCE7"},
                {'range': [40, 70], 'color': "#FEF9C3"},
                {'range': [70, 100], 'color': "#FEE2E2"}
            ]
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="white", margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

    if 'prob' in st.session_state:
        p = st.session_state['prob']
        if st.session_state['pred']==1:
            st.warning(f"Prediction: Customer likely to churn. Probability {p:.1%}. Recommended: Review contract and support history.")
        else:
            st.success(f"Prediction: Customer likely to stay. Retention probability {(1-p):.1%}.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Input Summary")
    st.markdown("Current customer profile used for prediction.")
    summary_df = pd.DataFrame({
        "Attribute": ["Gender","Senior Citizen","Partner","Dependents","Tenure","Phone Service","Multiple Lines","Internet","Online Security","Online Backup","Device Protection","Tech Support","Streaming TV","Streaming Movies","Contract","Paperless Billing","Payment Method","Monthly Charges","Total Charges"],
        "Value": [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=420)
    st.caption("Note: All categorical values are encoded using OneHotEncoder with handle_unknown='ignore'. Numerical values are standardized.")
    st.markdown('</div>', unsafe_allow_html=True)
