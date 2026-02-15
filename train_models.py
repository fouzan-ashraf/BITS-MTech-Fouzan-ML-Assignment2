import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("🚀 Starting Model Training Pipeline...")

# 1. Create model directory
os.makedirs('model', exist_ok=True)

# 2. Load Data
df = pd.read_csv('data.csv')
if 'id' in df.columns: df = df.drop(columns=['id'])
if 'Unnamed: 32' in df.columns: df = df.drop(columns=['Unnamed: 32'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 3. Preprocessing
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'model/label_encoder.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'model/scaler.pkl')

# 4. Train and Save Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, m in models.items():
    print(f"Training {name}...")
    m.fit(X_train_scaled, y_train)
    safe_name = name.replace(" ", "_")
    joblib.dump(m, f'model/{safe_name}_model.pkl')

print("✅ All models trained and saved successfully in 'model/' directory!")
