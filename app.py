import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

# --- CUSTOM UI STYLING (CSS) ---
st.markdown("""
    <style>
    /* Make Tabs bigger and more appealing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        font-size: 18px;
        color: #31333F;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e73df !important;
        color: white !important;
    }
    /* Compact Header Styling */
    .header-text {
        font-size: 14px !important;
        font-weight: 500;
        color: #5a5c69;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. HORIZONTAL HEADER ---
h_col1, h_col2, h_col3 = st.columns([2, 2, 1])
with h_col1:
    st.markdown('<p class="header-text">👤 <b>Name:</b> FOUZAN ASHRAF</p>', unsafe_allow_html=True)
with h_col2:
    st.markdown('<p class="header-text">🆔 <b>BITS ID:</b> 2025AB05236</p>', unsafe_allow_html=True)
with h_col3:
    st.markdown('<p class="header-text">📅 <b>Date:</b> 13-02-2026</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Main Title ---
st.title("Machine Learning Assignment-2 : Fouzan Ashraf")

# --- 2. DATA LOADING ---
data_source = st.radio("Select Data Source", ["Upload Your Own CSV", "Use Preloaded GitHub Repository Dataset (data.csv)"], horizontal=True)
df = None

if data_source == "Use Preloaded GitHub Repository Dataset (data.csv)":
    try:
        try: df = pd.read_csv('model/data.csv')
        except: df = pd.read_csv('data.csv')
        st.success("Dataset loaded successfully!")
    except: st.error("Dataset 'data.csv' not found.")
elif data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# --- 3. APP LOGIC ---
if df is not None:
    # Preprocessing
    if 'id' in df.columns: df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])
    
    target_options = df.columns.tolist()
    default_target = 'diagnosis' if 'diagnosis' in target_options else target_options[-1]
    target_col = st.sidebar.selectbox("🎯 Set Target Column", target_options, index=target_options.index(default_target))
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    try: y = le.fit_transform(y)
    except: pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. TABS INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "⚙️ Select Model to Train", "🏆 Compare all Model"])

    # TAB 1: DATA ANALYSIS
    with tab1:
        st.subheader("Detailed Dataset Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())
        c4.metric("Duplicate Rows", df.duplicated().sum())
        
        st.markdown("### Data Preview & Correlation")
        col_da1, col_da2 = st.columns([1, 1])
        with col_da1:
            st.write("**Dataset Head**")
            st.dataframe(df.head(10))
        with col_da2:
            st.write("**Class Distribution**")
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, palette='viridis', ax=ax_dist)
            st.pyplot(fig_dist)

    # TAB 2: SELECT MODEL TO TRAIN
    with tab2:
        model_name = st.selectbox("Select ML Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])
        
        if model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "Decision Tree": model = DecisionTreeClassifier(random_state=42)
        elif model_name == "KNN": model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naive Bayes": model = GaussianNB()
        elif model_name == "Random Forest": model = RandomForestClassifier(random_state=42)
        else: model = XGBClassifier(eval_metric='logloss', random_state=42)

        if st.button(f"🚀 Execute {model_name} Training"):
            start = time.time()
            model.fit(X_train_scaled, y_train)
            dur = time.time() - start
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

            st.success(f"Trained in {dur:.4f}s")
            
            # 6 Evaluation Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            m2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
            m3.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            m4.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            m5.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
            m6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

            # Visualizations
            st.markdown("---")
            v1, v2 = st.columns(2)
            with v1:
                st.write("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                st.pyplot(fig_cm)
            with v2:
                st.write("**Classification Report**")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.style.format("{:.3f}"))
            
            # --- CONVERGENCE GRAPH (Learning Curve) ---
            st.write("**Convergence Graph (Learning Curve)**")
            
            train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train, cv=5)
            fig_lc, ax_lc = plt.subplots(figsize=(10, 4))
            ax_lc.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
            ax_lc.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation Score")
            ax_lc.set_title(f"Convergence Graph: {model_name}")
            ax_lc.set_xlabel("Training Examples"), ax_lc.set_ylabel("Score"), ax_lc.legend()
            st.pyplot(fig_lc)

    # TAB 3: COMPARE ALL MODELS
    with tab3:
        if st.button("🔥 Run All-Model Comparison"):
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss')
            }
            results = []
            for name, m in models.items():
                m.fit(X_train_scaled, y_train)
                p = m.predict(X_test_scaled)
                results.append({
                    "Model": name, "Accuracy": accuracy_score(y_test, p),
                    "AUC": roc_auc_score(y_test, m.predict_proba(X_test_scaled)[:, 1]) if hasattr(m, "predict_proba") else 0,
                    "F1 Score": f1_score(y_test, p), "MCC": matthews_corrcoef(y_test, p)
                })
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen'))
            
            fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Accuracy", y="Model", data=res_df, palette="viridis")
            st.pyplot(fig_comp)
