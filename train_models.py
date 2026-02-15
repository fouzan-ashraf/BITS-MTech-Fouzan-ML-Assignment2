import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)

print("🚀 Starting Model Training Pipeline...")

# 1. Create model directory
os.makedirs('model', exist_ok=True)

# 2. Load Data
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("❌ Error: 'data.csv' not found. Please ensure the dataset is in the same directory as this script.")
    exit()

if 'id' in df.columns: df = df.drop(columns=['id'])
if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 3. Preprocessing
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'model/label_encoder.pkl')

# Deterministic Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test set so we can evaluate the models!
X_test_scaled = scaler.transform(X_test) 
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Train, Evaluate, and Save Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# Dictionary to hold our calculated metrics
evaluation_metrics = {}

for name, m in models.items():
    print(f"Training and Evaluating {name}...")
    
    # Train
    m.fit(X_train_scaled, y_train)
    
    # Predict on the holdout test set
    p = m.predict(X_test_scaled)
    prob = m.predict_proba(X_test_scaled)[:, 1] if hasattr(m, "predict_proba") else p
    
    # Calculate Metrics and store them in the dictionary (converted to float for JSON compatibility)
    evaluation_metrics[name] = {
        "accuracy": float(accuracy_score(y_test, p)),
        "auc": float(roc_auc_score(y_test, prob)),
        "precision": float(precision_score(y_test, p, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_test, p, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_test, p, average='weighted')),
        "mcc": float(matthews_corrcoef(y_test, p))
    }
    
    # Save the model
    safe_name = name.replace(" ", "_")
    joblib.dump(m, f'model/{safe_name}_model.pkl')

# 5. Export Metrics to JSON for the Streamlit App
with open('model/evaluation_metrics.json', 'w') as f:
    json.dump(evaluation_metrics, f, indent=4)

print("✅ All models trained, evaluated, and saved successfully in 'model/' directory!")
print("✅ Baseline metrics successfully exported to 'model/evaluation_metrics.json'!")
