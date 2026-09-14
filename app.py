import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def get_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv") # ye csv bhi GitHub pe daal de
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)
    
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["customerID", "Churn"]]
    
    X = df[num_cols + cat_cols]
    y = df["Churn"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X, y)
    return pipeline

model = get_model()
# ... baaki wahi predict wala code jo maine pehle diya tha
