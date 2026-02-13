import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

# --- Page Configuration ---
st.set_page_config(page_title="Machine Learning Assignment-2 : Fouzan Ashraf", layout="wide")

# --- Student Information (Top Left Header) ---
st.sidebar.markdown("### FOUZAN ASHRAF (2025AB05236)")
st.sidebar.caption("Release Date: 13-02-2026")
st.sidebar.markdown("---")

# --- Main Title ---
st.title("Machine Learning Assignment-2 : Fouzan Ashraf")
st.markdown("### Classification Model Deployment & Evaluation")
st.markdown("This application implements 6 classification models to predict breast cancer diagnosis (Malignant vs Benign).")
st.markdown("---")

# --- 1. Dataset Loading ---
st.header("1. Data Configuration")
data_source = st.radio("Select Data Source", ["Use Repository Dataset (data.csv)", "Upload Your Own CSV"], horizontal=True)

df = None

# Logic to load data
if data_source == "Use Repository Dataset (data.csv)":
    try:
        # Check both potential locations (root or model folder)
        try:
            df = pd.read_csv('model/data.csv')
        except FileNotFoundError:
            df = pd.read_csv('data.csv')
        st.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found in 'model/' or root directory.")
elif data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully!")

# --- 2. Main Application Logic ---
if df is not None:
    # --- Preprocessing (Common for all modes) ---
    # Cleanup
    if 'id' in df.columns: df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])

    # Target Selection
    target_options = df.columns.tolist()
    default_target = 'diagnosis' if 'diagnosis' in target_options else target_options[-1]
    
    with st.expander("Data Settings"):
        target_col = st.selectbox("Select Target Column", target_options, index=target_options.index(default_target))
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encoding
    le = LabelEncoder()
    try:
        y = le.fit_transform(y)
    except:
        pass # Already numeric

    # Train-Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- TABS Interface ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "⚙️ Train Individual Model", "🏆 Compare All Models"])

    # ==========================================
    # TAB 1: DATA ANALYSIS
    # ==========================================
    with tab1:
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Raw Data Preview:")
            st.dataframe(df.head())
            st.write(f"**Shape:** {df.shape}")
        
        with col2:
            st.write("Target Distribution:")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.countplot(x=target_col, data=df, palette='viridis', ax=ax)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', xytext = (0, 9), 
                            textcoords = 'offset points', fontweight='bold')
            st.pyplot(fig)

    # ==========================================
    # TAB 2: INDIVIDUAL MODEL TRAINING
    # ==========================================
    with tab2:
        st.header("Train a Single Model")
        
        # Sidebar Model Selection (Moved here for better flow)
        model_name = st.selectbox("Choose Classification Model", 
            ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)", 
             "Naive Bayes", "Random Forest", "XGBoost"])

        # Initialize Model
        if model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "Decision Tree": model = DecisionTreeClassifier(random_state=42)
        elif model_name == "K-Nearest Neighbors (KNN)": model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naive Bayes": model = GaussianNB()
        elif model_name == "Random Forest": model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "XGBoost": model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        if st.button(f"Train {model_name}", type="primary"):
            with st.spinner("Training model..."):
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test_scaled)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_prob = y_pred

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                
                st.success(f"Training completed in {train_time:.4f} seconds")
                
                # Metrics Display
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Accuracy", f"{acc:.4f}")
                col_m2.metric("F1 Score", f"{f1:.4f}")
                col_m3.metric("AUC Score", f"{auc:.4f}")
                
                # Visualizations for Individual Model
                st.subheader("Model Visualization")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    st.pyplot(fig_cm)
                with c2:
                    st.write("**Classification Report**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))

    # ==========================================
    # TAB 3: DYNAMIC COMPARISON
    # ==========================================
    with tab3:
        st.header("Compare All Models")
        st.write("Click the button below to train all models dynamically and compare their performance.")
        
        if st.button("🚀 Run Full Model Comparison", type="primary"):
            with st.spinner("Training all 6 models... this may take a moment"):
                
                # Define all models
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "Naive Bayes": GaussianNB(),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                }
                
                results_list = []
                
                # Loop through and train
                for name, clf in models.items():
                    # Train
                    clf.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred_c = clf.predict(X_test_scaled)
                    if hasattr(clf, "predict_proba"):
                        y_prob_c = clf.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_prob_c = y_pred_c
                        
                    # Calculate Metrics
                    res = {
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, y_pred_c),
                        "AUC": roc_auc_score(y_test, y_prob_c) if len(np.unique(y_test)) > 1 else 0.5,
                        "Precision": precision_score(y_test, y_pred_c, average='weighted'),
                        "Recall": recall_score(y_test, y_pred_c, average='weighted'),
                        "F1 Score": f1_score(y_test, y_pred_c, average='weighted'),
                        "MCC": matthews_corrcoef(y_test, y_pred_c)
                    }
                    results_list.append(res)
                
                # Create DataFrame
                results_df = pd.DataFrame(results_list)
                
                # Display Leaderboard
                st.subheader("🏆 Model Leaderboard")
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen', subset=["Accuracy", "AUC", "F1 Score"]))
                
                # Comparison Visualization
                st.subheader("Performance Comparison (Accuracy)")
                
                # Create Bar Chart
                fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
                sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis", ax=ax_comp)
                plt.xlim(0.8, 1.0) # Zoom in to see differences
                plt.title("Model Accuracy Comparison")
                st.pyplot(fig_comp)
                
                st.success("Comparison Complete!")

else:
    st.info("Please upload a CSV or ensure 'data.csv' is present to proceed.")
