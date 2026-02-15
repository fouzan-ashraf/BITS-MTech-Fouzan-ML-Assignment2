import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
from sklearn.model_selection import train_test_split, learning_curve
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

# Ensure the 'model' directory exists
os.makedirs('model', exist_ok=True)

# --- CUSTOM CSS FOR FULL-WIDTH TABS & HEADER ---
st.markdown("""
    <style>
    button[data-baseweb="tab"] {
        flex: 1;
        font-size: 20px !important;
        font-weight: bold !important;
        height: 70px !important;
        background-color: #f8f9fc !important;
        border-radius: 5px 5px 0px 0px !important;
        margin: 0px 2px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4e73df !important;
        color: white !important;
        border-bottom: 4px solid #2e59d9 !important;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: 800 !important; 
    }
    .header-box {
        background-color: #f1f3f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
        margin-bottom: 20px;
    }
    .header-text {
        font-size: 16px !important;
        margin: 0;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. HORIZONTAL HEADER ---
st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between;">
            <p class="header-text"><b>Name:</b> FOUZAN ASHRAF</p>
            <p class="header-text"><b>BITS ID:</b> 2025AB05236</p>
            <p class="header-text"><b>Release Date:</b> 13-02-2026</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.title("Machine Learning Assignment-2 : Fouzan Ashraf")
st.markdown("### Classification Model Deployment & Evaluation Dashboard")

# --- 2. DATA LOADING & TARGET SELECTION ---
st.header("Data Upload/Load")
data_source = st.radio("Select Data Source", ["Upload Your Own CSV", "Use Preloaded GitHub Repository Dataset (data.csv)"], horizontal=True)

df = None
if data_source == "Use Preloaded GitHub Repository Dataset (data.csv)":
    try:
        try: df = pd.read_csv('model/data.csv')
        except: df = pd.read_csv('data.csv')
        st.success("GitHub Repository dataset loaded successfully!")
    except: st.error("Dataset 'data.csv' not found.")
elif data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

target_col = None
if df is not None:
    if 'id' in df.columns: df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])
    
    col_target1, col_target2 = st.columns([1, 1])
    with col_target1:
        target_options = df.columns.tolist()
        default_idx = target_options.index('diagnosis') if 'diagnosis' in target_options else len(target_options)-1
        target_col = st.selectbox("🎯 Select Target Column (Label)", target_options, index=default_idx)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    
    is_categorical = y.dtype == 'object'
    if is_categorical:
        y_encoded = le.fit_transform(y)
        joblib.dump(le, 'model/label_encoder.pkl') # Save to model/
    else:
        y_encoded = y

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'model/scaler.pkl') # Save to model/

    test_export_df = X_test.copy()
    if is_categorical:
        test_export_df[target_col] = le.inverse_transform(y_test)
    else:
        test_export_df[target_col] = y_test

    st.markdown("---")

    # --- 3. TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "⚙️ Select Model to Train", "🏆 Compare All Models", "🚀 Inference (.pkl)"])

    # TAB 1: DATA ANALYSIS
    with tab1:
        st.subheader("Dataset Health & Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Rows", df.shape[0])
        s2.metric("Total Features", df.shape[1] - 1)
        s3.metric("Missing Values", df.isnull().sum().sum())
        s4.metric("Duplicate Rows", df.duplicated().sum())
        
        st.markdown(f"**Target Class Distribution ({target_col}):**")
        class_counts = df[target_col].value_counts()
        count_cols = st.columns(len(class_counts))
        for i, (cls_name, count) in enumerate(class_counts.items()):
            label = str(cls_name)
            if label.upper() == 'M': label = "Malignant (M)"
            if label.upper() == 'B': label = "Benign (B)"
            pct = (count / df.shape[0]) * 100
            count_cols[i].metric(f"Class: {label}", f"{count} ({pct:.1f}%)")
        
        c_da1, c_da2 = st.columns([1, 1])
        with c_da1:
            st.dataframe(df.head(10), use_container_width=True)
        with c_da2:
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, palette='viridis', ax=ax_dist)
            st.pyplot(fig_dist)

    # TAB 2: SELECT MODEL TO TRAIN
    with tab2:
        model_name = st.selectbox("Choose Classification Model", 
                                  ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])
        
        if model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "Decision Tree": model = DecisionTreeClassifier(random_state=42)
        elif model_name == "KNN": model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naive Bayes": model = GaussianNB()
        elif model_name == "Random Forest": model = RandomForestClassifier(random_state=42)
        else: model = XGBClassifier(eval_metric='logloss', random_state=42)

        if st.button(f"🚀 Execute {model_name} Training", type="primary"):
            with st.spinner("Processing..."):
                start = time.time()
                model.fit(X_train_scaled, y_train)
                dur = time.time() - start
                
                # Save specifically named model to model/
                safe_name = model_name.replace(" ", "_")
                joblib.dump(model, f'model/{safe_name}_model.pkl')
                
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

                st.success(f"Model trained and saved as 'model/{safe_name}_model.pkl' in {dur:.4f}s")
                
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                m2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
                m3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                m4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                m5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                m6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    # TAB 3: COMPARE ALL MODELS
    with tab3:
        if st.button("🔥 Run All-Model Dynamic Comparison", type="primary"):
            with st.spinner("Training and saving all 6 models to model/ directory..."):
                models_list = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "Naive Bayes": GaussianNB(),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
                }
                
                comp_results = []
                for name, m in models_list.items():
                    m.fit(X_train_scaled, y_train)
                    
                    # Save each model to model/
                    safe_name = name.replace(" ", "_")
                    joblib.dump(m, f'model/{safe_name}_model.pkl')
                    
                    p = m.predict(X_test_scaled)
                    prob = m.predict_proba(X_test_scaled)[:, 1] if hasattr(m, "predict_proba") else p
                    
                    comp_results.append({
                        "Model": name, "Accuracy": accuracy_score(y_test, p), "AUC": roc_auc_score(y_test, prob), 
                        "Precision": precision_score(y_test, p, average='weighted', zero_division=0),
                        "Recall": recall_score(y_test, p, average='weighted', zero_division=0),
                        "F1 Score": f1_score(y_test, p, average='weighted'), "MCC": matthews_corrcoef(y_test, p)
                    })
                
                res_df = pd.DataFrame(comp_results)
                st.subheader("🏆 Comparative Leaderboard")
                st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen').format(
                    {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}", 
                     "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"}), use_container_width=True)
                st.success("All models successfully saved to the 'model/' directory as .pkl files!")

    # --- TAB 4: INFERENCE (STRICTLY FROM .PKL) ---
    with tab4:
        st.subheader("1. Download Holdout Test Data")
        csv_test = test_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Download test-data.csv", data=csv_test, file_name="test-data.csv", mime="text/csv")
        
        st.markdown("---")
        st.subheader("2. Upload Test Data for Inference (Using Saved .pkl Models)")
        test_file = st.file_uploader("Upload your test-data.csv", type=["csv"], key="test_upload")
        
        if test_file is not None:
            new_test_df = pd.read_csv(test_file)
            st.success(f"Test data loaded! ({new_test_df.shape[0]} rows)")
            
            if target_col not in new_test_df.columns:
                st.error(f"Dataset must contain target column: '{target_col}'")
            else:
                try:
                    # LOAD SCALER FROM GitHub Repo model/ folder
                    loaded_scaler = joblib.load('model/scaler.pkl')
                    
                    X_new = new_test_df.drop(columns=[target_col])
                    y_new_raw = new_test_df[target_col]
                    
                    if is_categorical:
                        loaded_le = joblib.load('model/label_encoder.pkl')
                        y_new = loaded_le.transform(y_new_raw)
                    else:
                        y_new = y_new_raw
                        
                    X_new_scaled = loaded_scaler.transform(X_new)
                    
                    models_to_test = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
                    inference_mode = st.radio("Select Inference Mode:", ["Evaluate Single Model (.pkl)", "Compare All Models (.pkl)"], horizontal=True)
                    
                    if inference_mode == "Evaluate Single Model (.pkl)":
                        inf_model_name = st.selectbox("Select Model for Inference", models_to_test)
                        if st.button("🧠 Run Inference from .pkl", type="primary"):
                            safe_name = inf_model_name.replace(" ", "_")
                            try:
                                loaded_model = joblib.load(f'model/{safe_name}_model.pkl')
                                with st.spinner("Predicting using saved model..."):
                                    p = loaded_model.predict(X_new_scaled)
                                    prob = loaded_model.predict_proba(X_new_scaled)[:, 1] if hasattr(loaded_model, "predict_proba") else p
                                    
                                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                                    m1.metric("Accuracy", f"{accuracy_score(y_new, p):.4f}")
                                    m2.metric("AUC", f"{roc_auc_score(y_new, prob):.4f}")
                                    m3.metric("Precision", f"{precision_score(y_new, p, average='weighted', zero_division=0):.4f}")
                                    m4.metric("Recall", f"{recall_score(y_new, p, average='weighted', zero_division=0):.4f}")
                                    m5.metric("F1 Score", f"{f1_score(y_new, p, average='weighted'):.4f}")
                                    m6.metric("MCC Score", f"{matthews_corrcoef(y_new, p):.4f}")
                            except FileNotFoundError:
                                st.error(f"⚠️ 'model/{safe_name}_model.pkl' not found. Ensure it is committed to GitHub or train it in Tab 2 first.")

                    else:
                        if st.button("🔥 Run All-Model .pkl Inference", type="primary"):
                            inf_results = []
                            with st.spinner("Loading .pkl files from model/ directory..."):
                                for name in models_to_test:
                                    safe_name = name.replace(" ", "_")
                                    try:
                                        loaded_m = joblib.load(f'model/{safe_name}_model.pkl')
                                        p = loaded_m.predict(X_new_scaled)
                                        prob = loaded_m.predict_proba(X_new_scaled)[:, 1] if hasattr(loaded_m, "predict_proba") else p
                                        
                                        inf_results.append({
                                            "Model": name, "Accuracy": accuracy_score(y_new, p), "AUC": roc_auc_score(y_new, prob), 
                                            "Precision": precision_score(y_new, p, average='weighted', zero_division=0),
                                            "Recall": recall_score(y_new, p, average='weighted', zero_division=0),
                                            "F1 Score": f1_score(y_new, p, average='weighted'), "MCC": matthews_corrcoef(y_new, p)
                                        })
                                    except FileNotFoundError:
                                        st.warning(f"⚠️ {name} .pkl not found in model/. Skipping.")
                                
                                if inf_results:
                                    res_df_new = pd.DataFrame(inf_results)
                                    st.subheader("🏆 Inference Leaderboard (From Saved .pkl Models)")
                                    st.dataframe(res_df_new.style.highlight_max(axis=0, color='lightgreen').format(
                                        {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}", 
                                         "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"}), use_container_width=True)

                except FileNotFoundError:
                    st.error("⚠️ Preprocessor files ('model/scaler.pkl' or 'model/label_encoder.pkl') not found. Ensure they are in your GitHub repo.")
else:
    st.info("Please select a data source or upload a file to begin the analysis.")
