# Machine Learning Classification Model Implementation, Demonstration and Comparison

## a. Problem Statement
The objective of this assignment is to implement an end-to-end Machine Learning classification workflow. This involves:
1.  Selecting a real-world classification dataset.
2.  Implementing and training 6 different classification models (Logistic Regression, Decision Tree Classifier (DNN), K-Nearest Neighbor Classifier (KNN), Naive Bayes Classifier, Ensemble Model - Random Forest and Ensemble Model - XGBoost).
3.  Evaluating models using standard metrics (Accuracy, Precision, Recall, F1 Score, AUC, MCC).
4.  Developing an interactive web application using **Streamlit** to demonstrate the models.
5.  Deploying the application to the **Streamlit Community Cloud** for public access.

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Description:** The dataset predicts whether a breast mass is **benign (B)** or **malignant (M)** based on 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
* **Target Variable:** `Diagnosis`
* **Instances:** 569 samples (Class distribution: 357 Benign, 212 Malignant)
* **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.) + 1 Target variable
* **Original Data Shape:** 569*32

## c. Models used
### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.973684 | 0.997380 | 0.973719 | 0.973684 | 0.973621 | 0.943898 |
| **Decision Tree** | 0.947368 | 0.943990 | 0.947368 | 0.947368 | 0.947368 | 0.887979 |
| **KNN** | 0.947368 | 0.981985 | 0.947368 | 0.947368 | 0.947368 | 0.887979 |
| **Naive Bayes** | 0.964912 | 0.997380 | 0.965205 | 0.964912 | 0.964738 | 0.925285 |
| **Random Forest (Ensemble)** | 0.964912 | 0.995251 | 0.965205 | 0.964912 | 0.964738 | 0.925285 |
| **XGBoost (Ensemble)** | 0.956140 | 0.990829 | 0.956088 | 0.956140 | 0.956036 | 0.906379 |

### Metric Definitions
* **Accuracy:** The ratio of correctly predicted observations to the total observations.
* **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between classes. A higher AUC means the model is better at predicting 0 classes as 0 and 1 classes as 1.
* **Precision:** The ratio of correctly predicted positive observations to the total predicted positive observations. Highlights how many selected items are relevant.
* **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all observations in the actual class. Highlights how many relevant items are selected.
* **F1 Score:** The weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.
* **MCC (Matthews Correlation Coefficient):** A more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives).

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest performance** across all metrics (Accuracy: ~0.9737, F1: ~0.9736). The exceptionally high AUC (~0.9974) and MCC (~0.9439) confirm that the dataset is highly linearly separable, making this simpler model highly effective. |
| **Decision Tree** | Showed the lowest AUC (~0.9440) among all models. While its accuracy (~0.9474) is respectable, the lower MCC (~0.8880) compared to ensembles suggests it suffers from higher variance and overfitting on the training data. |
| **KNN** | Produced identical accuracy, precision, and F1 scores to the Decision Tree (~0.9474), but achieved a significantly better AUC (~0.9820). This indicates it is better at ranking probabilities even if the final hard classification labels matched the tree. |
| **Naive Bayes** | Performed exceptionally well, tying exactly with Random Forest across almost all metrics (Accuracy: ~0.9649). Its AUC (~0.9974) matched Logistic Regression as the highest in the group, suggesting the features largely follow a Gaussian distribution. |
| **Random Forest (Ensemble)** | Significantly outperformed the single Decision Tree (Accuracy: ~0.9649 vs ~0.9474). The ensemble bagging method successfully reduced variance, matching Naive Bayes in accuracy but with a slightly lower AUC (~0.9953). |
| **XGBoost (Ensemble)** | Delivered strong results (Accuracy: ~0.9561) but slightly trailed the simpler linear and probabilistic models. On this smaller dataset (569 rows), the complex boosting algorithm may have slightly overfitted compared to Logistic Regression. |

## e. Project Structure

```text
├── app.py                  
├── train_models.py         
├── requirements.txt        
├── data.csv                
├── README.md          
└── model/                  
    ├── 2025ab05236_ml_assignment2.ipynb
    ├── scaler.pkl          
    ├── label_encoder.pkl   
    ├── Logistic_Regression_model.pkl
    ├── Decision_Tree_model.pkl
    ├── KNN_model.pkl
    ├── Naive_Bayes_model.pkl
    ├── Random_Forest_model.pkl
    └── XGBoost_model.pkl
```

## f. Building and Deployment

### Pre-Requisites:
* Python 3.8 or higher
* `pip` package manager

### Steps to Build Locally:
1. **Clone the GitHub repository**
   ```bash
   git clone https://github.com/fouzan-ashraf/2025AB05236-FOUZAN-ASHRAF-MACHINE-LEARNING-ASSIGNMENT-2.git](https://github.com/fouzan-ashraf/2025AB05236-FOUZAN-ASHRAF-MACHINE-LEARNING-ASSIGNMENT-2.git
   cd 2025AB05236-FOUZAN-ASHRAF-MACHINE-LEARNING-ASSIGNMENT-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**
   Download the dataset manually from Kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and save it as `data.csv` in the project root (Already downloaded and saved in GitHub project root).

4. **Train the models**
   ```bash
   python train_models.py
   ```

5. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

### Steps to Deploy on Streamlit Community Cloud:
0. Code push to GitHub Repo (already pushed).
1. Go to https://streamlit.io/cloud
2. Sign in using your GitHub account.
3. Click “New App”.
4. Select your repository (`fouzan-ashraf/2025AB05236-FOUZAN-ASHRAF-MACHINE-LEARNING-ASSIGNMENT-2` in this case).
5. Choose branch (`master` in this case).
6. Select `app.py` as the main file path.
7. Click Deploy.
8. Wait for the app to come online within 3-4 minutes.

## g. How to Use the Deployed Application
The deployed Streamlit application features two main sections:
* **Training Data Analysis:** A visual overview of the dataset used to train the models, including feature counts, target distribution, and a raw data preview.
* **Model Inference and Evaluation:** The core engine of the app. 
    * **Evaluation Mode:** Download the unseen test data provided on the screen (or use the GitHub default), upload it back to the app, and evaluate individual models or compare all 6 side-by-side to view detailed performance metrics.
    * **Blind Prediction Mode:** Upload a "blind" dataset (data *without* the `Diagnosis` column) and use the "Predict using One of the 6 models from dropdown" feature to have the AI generate and download a CSV of actual medical predictions.

## Technology Used
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment UI:** Streamlit
* **Model Serialization:** Joblib

---
**Author:** Fouzan Ashraf  
**Information:** *This project is for educational purposes only and should not be relied upon for actual medical diagnosis or scenario testing.*
