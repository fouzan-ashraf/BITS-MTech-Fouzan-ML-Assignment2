import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

# --- Page Configuration ---
st.set_page_config(page_title="Machine Learning Assignment-2 : Fouzan Ashraf", layout="wide")

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

# --- HORIZONTAL HEADER ---
st.markdown(f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between;">
            <p class="header-text"><b>Name:</b> FOUZAN ASHRAF</p>
            <p class="header-text"><b>BITS ID:</b> 2025AB05236</p>
            <p class="header-text"><b>Release Date:</b> 13-02-2026</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.title("ML Classification Model Comparision - Breast Cancer Diagnostic")
st.markdown("Machine Learning Assignment-2 : Fouzan Ashraf")

# --- DATA LOADING (For generating Test Data Download & Analysis) ---
try:
    try: df = pd.read_csv('model/data.csv')
    except: df = pd.read_csv('data.csv')

    original_shape = df.shape
    
    # Cleanup & Split for deterministic test data generation
    if 'id' in df.columns: df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])
    
    target_col = 'diagnosis'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Deterministic Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Recombine test data for the download button
    test_export_df = X_test.copy()
    test_export_df[target_col] = y_test

except FileNotFoundError:
    st.error("⚠️ Base dataset 'data.csv' not found. Please ensure it is in the repository.")
    st.stop()

# ==========================================
# ONLY 2 TABS AS REQUESTED
# ==========================================
tab1, tab2 = st.tabs(["📊 Training Data Analysis", "🚀 Model Inference and Evaluation"])

# ------------------------------------------
# TAB 1: TRAINING DATA ANALYSIS
# ------------------------------------------
with tab1:
    st.subheader("Dataset Used: Breast Cancer Wisconsin (Diagnostic) Dataset")
    st.markdown("""
    **Dataset Details:**
    This dataset consists of 31 total features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features mathematically describe the characteristics of the cell nuclei present in the image, capturing 10 distinct traits (such as radius, texture, and area) across their mean, standard error, and 'worst' (largest) values, alongside the final diagnosis.
    * **Target Variable:** `diagnosis`
    * **Target Classes:** `M` = Malignant (Cancerous), `B` = Benign (Non-Cancerous)
    * **Key Features:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Symmetry, and Fractal Dimension.
    """)
    
    st.subheader("Dataset Health & Statistics")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Original Data Shape", f"{original_shape[0]} × {original_shape[1]}")
    s2.metric("Total Rows", df.shape[0])
    s3.metric("Total Features", df.shape[1] - 1, help="This count (30) excludes the target variable ('diagnosis') and non-predictive columns like 'id'.")
    s4.metric("Missing Values", df.isnull().sum().sum())
    s5.metric("Duplicate Rows", df.duplicated().sum())
    
    st.markdown(f"**Target Class Distribution ({target_col}):**")
    class_counts = df[target_col].value_counts()
    count_cols = st.columns(len(class_counts))
    for i, (cls_name, count) in enumerate(class_counts.items()):
        label = "Malignant (M)" if cls_name == 'M' else "Benign (B)"
        pct = (count / df.shape[0]) * 100
        count_cols[i].metric(f"Class: {label}", f"{count} ({pct:.1f}%)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_da1, c_da2 = st.columns([1, 1])
    with c_da1:
        st.write("**Dataset Preview**")
        st.dataframe(df.head(10), use_container_width=True)
    with c_da2:
        st.write("**Target Distribution**")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.countplot(x=target_col, data=df, palette='viridis', ax=ax_dist)
        st.pyplot(fig_dist)

# ------------------------------------------
# TAB 2: MODEL INFERENCE AND EVALUATION
# ------------------------------------------
with tab2:
    st.subheader("1. Download Holdout Test Data")
    st.write("Download the unseen 20% test split to evaluate the pre-trained models.")
    
    csv_test = test_export_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download test-data.csv", data=csv_test, file_name="test-data.csv", mime="text/csv")
    
    st.markdown("---")
    st.subheader("2. Upload Test Data for Inference")
    test_file = st.file_uploader("Upload your test-data.csv", type=["csv"], key="test_upload")
    
    if test_file is not None:
        new_test_df = pd.read_csv(test_file)
        st.success(f"Test data loaded! ({new_test_df.shape[0]} rows)")
        
        if target_col not in new_test_df.columns:
            st.error(f"Dataset must contain target column: '{target_col}'")
        else:
            try:
                # LOAD PREPROCESSORS
                loaded_scaler = joblib.load('model/scaler.pkl')
                loaded_le = joblib.load('model/label_encoder.pkl')
                
                # PREPARE DATA
                X_new = new_test_df.drop(columns=[target_col])
                y_new_raw = new_test_df[target_col]
                y_new = loaded_le.transform(y_new_raw)
                X_new_scaled = loaded_scaler.transform(X_new)
                
                models_to_test = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
                
                st.markdown("### 3. Run Evaluation")
                inference_mode = st.radio("Select Inference Mode:", ["Evaluate Single Model", "Compare All Models"], horizontal=True)
                
                # --- SINGLE MODEL EVALUATION ---
                if inference_mode == "Evaluate Single Model":
                    inf_model_name = st.selectbox("Select Model for Inference", models_to_test)
                    if st.button("🧠 Run Inference", type="primary"):
                        safe_name = inf_model_name.replace(" ", "_")
                        try:
                            loaded_model = joblib.load(f'model/{safe_name}_model.pkl')
                            with st.spinner("Predicting..."):
                                p = loaded_model.predict(X_new_scaled)
                                prob = loaded_model.predict_proba(X_new_scaled)[:, 1] if hasattr(loaded_model, "predict_proba") else p
                                
                                st.markdown(f"#### Results for {inf_model_name}")
                                m1, m2, m3, m4, m5, m6 = st.columns(6)
                                m1.metric("Accuracy", f"{accuracy_score(y_new, p):.4f}")
                                m2.metric("AUC", f"{roc_auc_score(y_new, prob):.4f}")
                                m3.metric("Precision", f"{precision_score(y_new, p, average='weighted', zero_division=0):.4f}")
                                m4.metric("Recall", f"{recall_score(y_new, p, average='weighted', zero_division=0):.4f}")
                                m5.metric("F1 Score", f"{f1_score(y_new, p, average='weighted'):.4f}")
                                m6.metric("MCC Score", f"{matthews_corrcoef(y_new, p):.4f}")
                                
                                v1, v2 = st.columns(2)
                                with v1:
                                    st.write("**Confusion Matrix**")
                                    fig_cm_new, ax_cm_new = plt.subplots(figsize=(4, 3))
                                    sns.heatmap(confusion_matrix(y_new, p), annot=True, fmt='d', cmap='Blues', ax=ax_cm_new)
                                    st.pyplot(fig_cm_new)
                                with v2:
                                    st.write("**Classification Report**")
                                    st.dataframe(pd.DataFrame(classification_report(y_new, p, target_names=loaded_le.classes_, output_dict=True)).T.style.format("{:.4f}"))
                                    
                        except FileNotFoundError:
                            st.error(f"⚠️ Pre-trained model 'model/{safe_name}_model.pkl' not found.")

                # --- ALL MODELS COMPARISON ---
                else:
                    if st.button("🔥 Run All-Model Comparison", type="primary"):
                        inf_results = []
                        with st.spinner("Loading .pkl files and evaluating..."):
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
                                    st.warning(f"⚠️ {name} model not found in model/. Skipping.")
                            
                            if inf_results:
                                res_df_new = pd.DataFrame(inf_results)
                                st.subheader("🏆 Inference Leaderboard")
                                st.dataframe(res_df_new.style.highlight_max(axis=0, color='lightgreen').format(
                                    {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}", 
                                     "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"}), use_container_width=True)
                                
                                fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
                                sns.barplot(x="Accuracy", y="Model", data=res_df_new, palette="viridis")
                                plt.title("Model Accuracy on Unseen Test Data")
                                st.pyplot(fig_bar)

            except FileNotFoundError:
                st.error("⚠️ Preprocessor files ('scaler.pkl' or 'label_encoder.pkl') not found in 'model/' directory.")
