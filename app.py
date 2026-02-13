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
import time

# --- Page Configuration ---
st.set_page_config(page_title="Machine Learning Assignment-2 - Fouzan Ashraf", layout="wide")

# --- Student Information Section ---
st.sidebar.title("Student Information")
st.sidebar.info(
    """
    **BITS ID:** 2025AB05236
    
    **Name:** FOUZAN ASHRAF
    
    **Email:** 2025ab05236@wilp.bits-pilani.ac.in
    
    **Date:** 14-02-2026
    
    **Program:** M.Tech AI & ML (BITS Pilani)
    
    **Semester:** 1st
    """
)

# --- Main Title ---
st.title("BITS Pilani - Machine Learning Assignment 2")
st.markdown("### Classification Model Deployment & Evaluation")
st.markdown("---")

# --- 1. Dataset Loading ---
st.header("1. Data Loading")
data_source = st.radio("Select Data Source", ["Upload Your Own CSV", "Use GitHub Repository Dataset (data.csv)"])

df = None

# Logic to load data
if data_source == "Use GitHub Repository Dataset (data.csv)":
    try:
        df = pd.read_csv('model/data.csv')
        st.success("Repository dataset 'data.csv' loaded successfully!")
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found in the repository. Please upload a file.")
elif data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully!")

# --- 2. Data Preprocessing & Setup ---
if df is not None:
    # Basic Cleanup (Based upon what is implemented in Notebook)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])

    # Show Raw Data Toggle
    if st.checkbox("Show Raw Data Preview"):
        st.write(df.head())

    # Target Column Selection
    # Auto-detect 'diagnosis' or default to last column
    target_options = df.columns.tolist()
    default_target = 'diagnosis' if 'diagnosis' in target_options else target_options[-1]
    
    st.subheader("Data Configuration")
    target_col = st.selectbox("Select Target Column", target_options, index=target_options.index(default_target))
    
    # Feature & Target Split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encoding Target (if categorical like M/B)
    le = LabelEncoder()
    try:
        y = le.fit_transform(y)
        st.write(f"**Target Class Mapping:** {dict(zip(le.classes_, le.transform(le.classes_)))}")
    except:
        st.write("Target is already numeric.")

    # Train-Test Split (Standard 80-20 as per notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling (Crucial for KNN/Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write(f"**Training Features Shape:** {X_train.shape}")
    st.write(f"**Testing Features Shape:** {X_test.shape}")
    st.markdown("---")

    # --- 3. Model Selection & Training ---
    st.header("2. Model Selection & Evaluation")
    
    model_name = st.selectbox("Choose a Classification Model", 
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)", 
         "Naive Bayes", "Random Forest", "XGBoost"])

    # Initialize Model based on selection
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Train Button
    if st.button(f"Train {model_name}"):
        with st.spinner(f"Training {model_name}..."):
            # Measure Training Time
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Get Probabilities (for AUC)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = y_pred  # Fallback
            
            # Calculate Metrics
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)

        # --- 4. Display Results ---
        st.success(f"Model Trained Successfully in {training_time:.4f} seconds!")
        
        st.subheader("Key Performance Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # --- 5. Visualizations ---
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        with col_viz2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))

else:
    st.info("Awaiting dataset. Please upload a CSV or ensure 'data.csv' is in the repository.")
