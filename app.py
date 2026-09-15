```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")
st.write(
    "AI-powered customer churn prediction using Logistic Regression."
)

st.divider()


# =========================================================
# TRAIN MODEL INSIDE APP.PY
# =========================================================

@st.cache_resource
def train_model():

    # Load dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    df = pd.read_csv(url)

    # Convert TotalCharges into numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df.dropna(inplace=True)

    # Remove customer ID
    df.drop("customerID", axis=1, inplace=True)

    # Features and target
    X = df.drop("Churn", axis=1)

    y = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Numerical columns
    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Categorical columns
    categorical_columns = [
        column for column in X.columns
        if column not in numerical_columns
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ]
    )

    # Complete ML pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    # Train model
    model.fit(X, y)

    return model, df


# Train model
try:
    model, dataset = train_model()

except Exception as e:
    st.error("Model training failed.")
    st.exception(e)
    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset Records",
        len(dataset)
    )

with col2:
    st.metric(
        "Features",
        19
    )

with col3:
    st.metric(
        "Algorithm",
        "Logistic Regression"
    )

with col4:
    st.metric(
        "Target",
        "Customer Churn"
    )


st.divider()


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.title("Customer Details")
st.sidebar.write("Enter customer information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0,
    step=10.0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.subheader("🔮 Customer Churn Prediction")

st.write(
    "Click the button below to predict whether the customer "
    "is likely to churn."
)

predict_button = st.button(
    "🚀 Predict Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Convert Senior Citizen Yes/No into 0/1
    senior_citizen_value = 1 if senior_citizen == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen_value],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    churn_percentage = probability * 100
    stay_percentage = 100 - churn_percentage


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    if prediction == 1:

        st.error("⚠️ HIGH CHURN RISK")

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

        st.progress(
            min(probability, 1.0)
        )

        st.warning(
            "This customer is likely to leave the company. "
            "Consider offering retention benefits, discounts, "
            "or personalized support."
        )

    else:

        st.success("✅ LOW CHURN RISK")

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

        st.progress(
            min(probability, 1.0)
        )

        st.info(
            "This customer is currently less likely to churn. "
            "Continue providing good service and engagement."
        )


    # =====================================================
    # PROBABILITY CHART
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Prediction Probability")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Stay", "Churn"],
                    y=[
                        stay_percentage,
                        churn_percentage
                    ],
                    text=[
                        f"{stay_percentage:.2f}%",
                        f"{churn_percentage:.2f}%"
                    ],
                    textposition="auto"
                )
            ]
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Prediction",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    with col2:

        st.subheader("👤 Customer Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Gender",
                "Senior Citizen",
                "Partner",
                "Dependents",
                "Tenure",
                "Internet Service",
                "Contract",
                "Monthly Charges",
                "Total Charges"
            ],
            "Value": [
                gender,
                senior_citizen,
                partner,
                dependents,
                f"{tenure} months",
                internet_service,
                contract,
                f"${monthly_charges:.2f}",
                f"${total_charges:.2f}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# MODEL DETAILS
# =========================================================

st.divider()

with st.expander("🤖 About This Machine Learning Model"):

    st.write(
        "This application uses Logistic Regression to predict "
        "customer churn."
    )

    st.write(
        "The model is trained automatically when the Streamlit "
        "application starts."
    )

    st.write(
        "Numerical features are standardized using StandardScaler "
        "and categorical features are converted using OneHotEncoder."
    )

    st.write(
        "The complete preprocessing and prediction process is "
        "handled through a Scikit-learn Pipeline."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project | "
    "Built with Python, Pandas, Scikit-learn & Streamlit"
)
```
