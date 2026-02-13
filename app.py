import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("BITS Pilani - ML Assignment 2")
st.markdown("**Student:** Fouzan Ashraf | **ID:** [Your ID]")

# --- 1. Dataset Handling  ---
st.sidebar.header("1. Data Configuration")
data_option = st.sidebar.radio("Select Data Source", ["Use Repository Data", "Upload CSV"])

df = None

if data_option == "Use Repository Data":
    try:
        df = pd.read_csv('data.csv')
        st.success("Repository dataset loaded.")
    except:
        st.error("data.csv not found in repository.")
elif data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded.")

if df is not None:
    # Preprocessing
    if 'id' in df.columns: df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])
    
    target_col = 'diagnosis' # Default assumption
    if target_col not in df.columns:
        target_col = df.columns[-1] # Fallback to last column
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 2. Model Selection  ---
    st.sidebar.header("2. Model Configuration")
    model_name = st.sidebar.selectbox("Choose Model", 
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

    if model_name == "Logistic Regression": model = LogisticRegression()
    elif model_name == "Decision Tree": model = DecisionTreeClassifier()
    elif model_name == "KNN": model = KNeighborsClassifier()
    elif model_name == "Naive Bayes": model = GaussianNB()
    elif model_name == "Random Forest": model = RandomForestClassifier()
    elif model_name == "XGBoost": model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    if st.sidebar.button("Train & Evaluate"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, "predict_proba") else y_pred

        # --- 3. Metrics  ---
        st.subheader(f"Results: {model_name}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

        st.divider()

        # --- 4. Confusion Matrix  ---
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        with c2:
            st.write("### Classification Report")
            report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report)
