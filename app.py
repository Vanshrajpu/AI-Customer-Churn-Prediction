import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

st.set_page_config(page_title="GreenBank AI PRO", page_icon="🇮🇳", layout="wide")

st.markdown("""
<style>
.stApp {background: #0d0d0d; color: #fde68a;}
.gold-card {background: #1a1a1a; border: 1px solid #d4af3766; border-radius: 16px; padding: 20px; margin-bottom:20px;}
.stButton>button {background: linear-gradient(90deg, #fcd34d, #f59e0b); color: black!important; font-weight:800!important; height:58px; border-radius:12px; width:100%;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()

if model:
    st.success("✅ model.pkl Loaded - Prediction Ready")
else:
    st.error("❌ model.pkl nahi mila. Check kar file GitHub pe hai ya nahi.")

st.markdown("<h2 style='color:#fde68a;'>GreenBank AI PRO 🇮🇳 • model.pkl Version</h2>", unsafe_allow_html=True)

L,R = st.columns(2)
with L:
    st.markdown('<div class="gold-card"><h4>Customer Details</h4>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)",0,72,18)
    mc = st.slider("Monthly Charges $",20,120,70)
    tc = st.slider("Total Charges $",0,10000,1500)
    Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    InternetService = st.selectbox("Internet Service",["Fiber optic","DSL","No"])
    gender = st.selectbox("Gender",["Male","Female"])
    PaymentMethod = st.selectbox("Payment Method",["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"])
    st.markdown('</div>', unsafe_allow_html=True)

with R:
    st.markdown('<div class="gold-card"><h4>AI Prediction</h4>', unsafe_allow_html=True)
    if st.button("✨ RUN AI ANALYSIS"):
        if model is None:
            st.error("Model load nahi hua!")
        else:
            data = pd.DataFrame([{
                "tenure":tenure,"MonthlyCharges":mc,"TotalCharges":tc,
                "gender":gender,"SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
                "PhoneService":"Yes","MultipleLines":"No","InternetService":InternetService,
                "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
                "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
                "Contract":Contract,"PaperlessBilling":"Yes","PaymentMethod":PaymentMethod
            }])
            try:
                prob = model.predict_proba(data)[0][1]
                pred = model.predict(data)[0]
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': "Churn Risk %"}, gauge={'bar': {'color': "#fcd34d"}, 'axis': {'range': [0, 100]}}))
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#fde68a"})
                st.plotly_chart(fig, use_container_width=True)
                if pred==1:
                    st.error(f"🔴 Churn Hoga! Risk: {prob*100:.1f}%")
                else:
                    st.success(f"🟢 Customer Safe Hai! Loyalty: {(1-prob)*100:.1f}%")
                st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Tera model.pkl purane version ka hai. Laptop pe wala train.py wala script se naya bana ke daal.")
    st.markdown('</div>', unsafe_allow_html=True)
