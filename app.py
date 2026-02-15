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
    /* 1. REMOVE TOP BLANK SPACE */
    .block-container {
        padding-top: 3rem !important;
    }
    
    /* 2. TAB STYLING */
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
    
    /* 3. HORIZONTAL HEADER STYLING */
    .header-box {
        background-color: #f1f3f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
        margin-bottom: 10px;
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
            <p class="header-text"><b>Release Date:</b> 15-02-2026</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SHRINK TITLE TO FIT ONE LINE ---
st.markdown("<div style='font-size: 32px; font-weight: bold; text-align: center; margin-bottom: 0px; padding-top: 0px; width: 100%;'>Machine Learning Classification Model Comparison - Breast Cancer Diagnosis</div>", unsafe_allow_html=True)

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
# 2 Side by Side Tab
# ==========================================
tab1, tab2 = st.tabs(["📊 Training Data Analysis", "🚀 Model Inference and Evaluation"])

# ------------------------------------------
# TAB 1: TRAINING DATA ANALYSIS
# ------------------------------------------
with tab1:
    # --- NAVIGATION INSTRUCTION BANNER ---
    st.info("👉 **Ready to test the models?** Click on the **'🚀 Model Inference and Evaluation'** tab to download/upload test data and perform evaluations or predictions.")

    st.markdown("<h3><b>Training Data Used:</b> <span style='font-weight: normal;'>Breast Cancer Wisconsin (Diagnostic) Dataset</span></h3>", unsafe_allow_html=True)
    st.markdown("""
    **Dataset Details:**
    This dataset consists of 30 total features (excluding target variable "diagnosis" and non-predictive column like "id") computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features mathematically describe the characteristics of the cell nuclei present in the image, capturing 10 distinct traits (such as radius, texture, and area) across their mean, standard error, and 'worst' (largest) values.
    * **Dataset Source:** Kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
    * **Target Variable:** `diagnosis`
    * **Target Classes:** `M` = Malignant (Cancerous), `B` = Benign (Non-Cancerous)
    * **Key Features:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Symmetry, and Fractal Dimension.
    """)
    
    st.subheader("Dataset Health & Statistics")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Original Data Shape", "569 × 32")
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
    # --- VISUAL INSTRUCTION BANNER ---
    st.info("""
    💡 **GUIDE: Evaluation Mode vs. Blind Prediction Mode**
    * **Evaluation Mode:** Upload test data *with* the `diagnosis` column. This allows the app to calculate Accuracy, AUC & other metrics and compare all 6 models against the true answers.
    * **Blind Prediction Mode:** Upload unseen data *without* the `diagnosis` column. Use the **Predict Using Individual Models from Dropdown** option to make the AI generate actual predictions for the unknown data. 
    *(Note: "Compare All Models" is automatically disabled for blind data, as metrics cannot be calculated without the ground truth).*
    """)

    # --- NEW FEATURE: BASELINE METRICS FROM TRAINING PHASE ---
    st.subheader("1. Pre-Trained Models Baseline Performance")
    with st.expander("🏆 View Evaluation Metrics Comparison (Calculated during Model Training)"):
        st.write("These metrics were generated during the initial model training phase using the 20% holdout test split. They serve as the baseline performance expectations for the deployed models.")
        
        # Hardcoded data from your README to avoid needing external JSON files
        baseline_data = {
            "Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"],
            "Accuracy": [0.973684, 0.947368, 0.947368, 0.964912, 0.964912, 0.956140],
            "AUC": [0.997380, 0.943990, 0.981985, 0.997380, 0.995251, 0.990829],
            "Precision": [0.973719, 0.947368, 0.947368, 0.965205, 0.965205, 0.956088],
            "Recall": [0.973684, 0.947368, 0.947368, 0.964912, 0.964912, 0.956140],
            "F1 Score": [0.973621, 0.947368, 0.947368, 0.964738, 0.964738, 0.956036],
            "MCC": [0.943898, 0.887979, 0.887979, 0.925285, 0.925285, 0.906379]
        }
        df_baseline = pd.DataFrame(baseline_data)
        
        st.dataframe(df_baseline.style.highlight_max(axis=0, color='lightgreen', subset=["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]).format(
            {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
             "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"}), use_container_width=True)

    st.markdown("---")

    st.subheader("2. Test Data Selection")
    data_source_test = st.radio(
        "Choose how to load test data:", 
        ["Upload Custom Test Data", "Load Test Data from GitHub"],
        horizontal=True
    )

    new_test_df = None

    # --- OPTION A: UPLOAD CUSTOM DATA ---
    if data_source_test == "Upload Custom Test Data":
        st.write("Download the unseen 20% test split to evaluate the pre-trained models, or download the blind dataset to test real-world predictions.")
        
        # Two side-by-side download buttons
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            csv_test = test_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download test-data.csv (With Target)", data=csv_test, file_name="test-data.csv", mime="text/csv")
        with col_down2:
            # X_test already has the target column dropped during the train_test_split
            csv_blind = X_test.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download blind-data.csv (No Target)", data=csv_blind, file_name="blind-data.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("3. Upload Test Data for Inference")
        test_file = st.file_uploader("Upload your test-data.csv or blind-data.csv", type=["csv"], key="test_upload")

        if test_file is not None:
            new_test_df = pd.read_csv(test_file)
            st.success(f"Data loaded successfully! ({new_test_df.shape[0]} rows)")
            
            # Display what mode the app is going into
            if target_col in new_test_df.columns:
                st.success("🎯 Target column detected: Entering **Evaluation Mode**")
            else:
                st.warning("❓ No target column detected: Entering **Blind Prediction Mode**")

    # --- OPTION B: LOAD FROM GITHUB ---
    else:
        st.markdown("---")
        st.subheader("3. GitHub Test Data Details")
        try:
            # Attempts to load a physically saved test-data.csv if it exists in the repo
            new_test_df = pd.read_csv('test-data.csv')
            st.success(f"GitHub test data loaded successfully! Entering **Evaluation Mode**.")
        except FileNotFoundError:
            # Falls back to the dynamically generated test split if the file isn't physically committed yet
            new_test_df = test_export_df.copy()
            st.success("Default holdout test data generated from base dataset loaded! Entering **Evaluation Mode**.")
        
        # Display details about the loaded GitHub data
        st.write(f"**Total Rows:** {new_test_df.shape[0]}")
        st.write(f"**Total Columns:** {new_test_df.shape[1]}")
        
        st.write("**Test Data Preview:**")
        st.dataframe(new_test_df.head(5), use_container_width=True)

    st.markdown("---")

    # --- 4. RUN EVALUATION & PREDICTION (Always Visible) ---
    st.subheader("4. Run Evaluation / Predictions")

    # Determine dynamic label based on presence of target column
    if new_test_df is not None and target_col not in new_test_df.columns:
        single_model_label = "Predict Using Individual Models from Dropdown"
    else:
        single_model_label = "Evaluate Individual Models from Dropdown"

    if new_test_df is None:
        st.info("ℹ️ Please upload a dataset or select 'Load Test Data from GitHub' above to enable evaluations/predictions.")

    models_to_test = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    
    # Use the dynamic label in the radio button
    inference_mode = st.radio("Select Inference Mode:", [single_model_label, "Compare All Models"], horizontal=True)

    # --- SINGLE MODEL EVALUATION / PREDICTION ---
    # Check against the dynamic label instead of a hardcoded string
    if inference_mode == single_model_label:
        inf_model_name = st.selectbox("Select Model for Inference", models_to_test)
        if st.button("🧠 Run Inference", type="primary"):
            if new_test_df is None:
                st.error("⚠️ Cannot run inference. Please provide data first.")
                st.toast("Missing data!", icon="⚠️")
            else:
                try:
                    loaded_scaler = joblib.load('model/scaler.pkl')
                    loaded_le = joblib.load('model/label_encoder.pkl')
                    safe_name = inf_model_name.replace(" ", "_")
                    loaded_model = joblib.load(f'model/{safe_name}_model.pkl')

                    is_evaluation = target_col in new_test_df.columns

                    if is_evaluation:
                        # --- EVALUATION MODE ---
                        X_new = new_test_df.drop(columns=[target_col])
                        y_new_raw = new_test_df[target_col]
                        y_new = loaded_le.transform(y_new_raw)
                        X_new_scaled = loaded_scaler.transform(X_new)

                        with st.spinner("Evaluating model..."):
                            p = loaded_model.predict(X_new_scaled)
                            prob = loaded_model.predict_proba(X_new_scaled)[:, 1] if hasattr(loaded_model, "predict_proba") else p

                            st.toast("Evaluation Complete! Scrolling to results...", icon="✅")
                            
                            st.markdown(f"#### Evaluation Results for {inf_model_name}")
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
                            
                            st.write("**Receiver Operating Characteristic (ROC) Curve**")
                            from sklearn.metrics import RocCurveDisplay
                            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                            RocCurveDisplay.from_predictions(y_new, prob, ax=ax_roc, name=inf_model_name)
                            st.pyplot(fig_roc)
                            
                    else:
                        # --- BLIND PREDICTION MODE ---
                        X_new_scaled = loaded_scaler.transform(new_test_df)
                        
                        with st.spinner("Generating Predictions for Unseen Data..."):
                            predictions = loaded_model.predict(X_new_scaled)
                            decoded_predictions = loaded_le.inverse_transform(predictions)
                            
                            results_df = new_test_df.copy()
                            results_df.insert(0, 'Predicted_Diagnosis', decoded_predictions)
                            
                            st.toast("Predictions Generated! Scrolling to results...", icon="✅")
                            st.success(f"Predictions generated successfully using {inf_model_name}!")
                            st.write("**Prediction Results (First 10 Rows):**")
                            st.dataframe(results_df.head(10), use_container_width=True)
                            
                            csv_preds = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="⬇️ Download Full Predictions as CSV", data=csv_preds, file_name=f"{safe_name}_predictions.csv", mime="text/csv")
                            
                except FileNotFoundError as e:
                    st.error(f"⚠️ Required pre-trained file not found. Ensure models and preprocessors exist in 'model/'. Error: {e}")
                    st.toast("File missing error.", icon="❌")

    # --- ALL MODELS COMPARISON ---
    else:
        if st.button("🔥 Run All-Model Comparison", type="primary"):
            if new_test_df is None:
                st.error("⚠️ Cannot run evaluation. Please provide test data first.")
                st.toast("Missing data!", icon="⚠️")
            elif target_col not in new_test_df.columns:
                st.error("⚠️ Blind prediction mode (missing target column) is only supported in 'Predict Using Individual Models from Dropdown'. To compare accuracy across models, the dataset must contain the 'diagnosis' column so we can check the answers.")
                st.toast("Incompatible Mode!", icon="❌")
            else:
                inf_results = []
                try:
                    loaded_scaler = joblib.load('model/scaler.pkl')
                    loaded_le = joblib.load('model/label_encoder.pkl')

                    X_new = new_test_df.drop(columns=[target_col])
                    y_new_raw = new_test_df[target_col]
                    y_new = loaded_le.transform(y_new_raw)
                    X_new_scaled = loaded_scaler.transform(X_new)

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
                            st.toast("Comparison Complete! Scrolling down...", icon="🏆")
                            res_df_new = pd.DataFrame(inf_results)
                            st.subheader("🏆 Inference Leaderboard")
                            st.dataframe(res_df_new.style.highlight_max(axis=0, color='lightgreen').format(
                                {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
                                 "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"}), use_container_width=True)

                            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
                            sns.barplot(x="Accuracy", y="Model", data=res_df_new, palette="viridis")
                            plt.title("Model Accuracy on Unseen Test Data")
                            st.pyplot(fig_bar)

                            best_acc = res_df_new.loc[res_df_new['Accuracy'].idxmax(), 'Model']
                            best_rec = res_df_new.loc[res_df_new['Recall'].idxmax(), 'Model']
                            best_f1 = res_df_new.loc[res_df_new['F1 Score'].idxmax(), 'Model']
                            
                            obs_text = f"""
                            **💡 Key Observations on Holdout Test Data:**
                            * **Highest Overall Accuracy:** The **{best_acc}** model is the top performer in raw accuracy on this unseen data.
                            * **Best for Safety (Recall):** In medical diagnostics, minimizing false negatives (missing a cancer diagnosis) is critical. **{best_rec}** achieved the highest recall, making it the safest model for initial screening.
                            * **Best Balance (F1 Score):** **{best_f1}** leads in the F1 Score, indicating it handles the trade-off between false alarms (Precision) and missed diagnoses (Recall) the best.
                            """
                            st.info(obs_text)
                            
                except FileNotFoundError:
                    st.error("⚠️ Preprocessor files ('scaler.pkl' or 'label_encoder.pkl') not found in 'model/' directory.")
