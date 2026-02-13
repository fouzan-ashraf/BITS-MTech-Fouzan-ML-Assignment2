## a. Problem Statement
The objective of this assignment is to implement an end-to-end Machine Learning classification workflow. This involves:
1.  Selecting a real-world classification dataset.
2.  Implementing and training 6 different classification models (Linear, Tree-based, Instance-based, Probabilistic, and Ensembles).
3.  Evaluating models using standard metrics (Accuracy, Precision, Recall, F1 Score, AUC, MCC).
4.  Developing an interactive web application using **Streamlit** to demonstrate the models.
5.  Deploying the application to the cloud for public access.

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Description:** The dataset predicts whether a breast mass is **benign (B)** or **malignant (M)** based on 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
* **Instances:** 569 samples (Class distribution: 357 Benign, 212 Malignant)
* **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.) + 1 Target variable.

## c. Models used
### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9737 | 0.9974 | 0.9737 | 0.9737 | 0.9736 | 0.9439 |
| **Decision Tree** | 0.9474 | 0.9440 | 0.9474 | 0.9474 | 0.9474 | 0.8880 |
| **KNN** | 0.9474 | 0.9820 | 0.9474 | 0.9474 | 0.9474 | 0.8880 |
| **Naive Bayes** | 0.9649 | 0.9974 | 0.9652 | 0.9649 | 0.9647 | 0.9253 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9953 | 0.9652 | 0.9649 | 0.9647 | 0.9253 |
| **XGBoost (Ensemble)** | 0.9561 | 0.9908 | 0.9561 | 0.9561 | 0.9560 | 0.9064 |

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest performance** across all metrics (Accuracy: 0.9737, F1: 0.9736). The high AUC (0.9974) and MCC (0.9439) confirm that the dataset is linearly separable to a high degree. |
| **Decision Tree** | Showed the lowest AUC (0.9440) among all models. While its accuracy (0.9474) is respectable, the lower MCC (0.8880) compared to ensembles suggests it suffers from higher variance and overfitting on the training data. |
| **KNN** | Produced identical accuracy, precision, and F1 scores to the Decision Tree (0.9474), but achieved a significantly better AUC (0.9820). This indicates it is better at ranking probabilities even if the final hard classification labels matched the tree. |
| **Naive Bayes** | Performed exceptionally well, tying with Random Forest for second place (Accuracy: 0.9649). Its AUC (0.9974) matched Logistic Regression, suggesting the features largely follow a Gaussian distribution. |
| **Random Forest (Ensemble)** | Significantly outperformed the single Decision Tree (Accuracy: 0.9649 vs 0.9474). The ensemble bagging method successfully reduced variance, matching Naive Bayes in accuracy but with slightly lower AUC (0.9953). |
| **XGBoost (Ensemble)** | Delivered strong results (Accuracy: 0.9561) but slightly trailed the simpler linear models. On this smaller dataset (569 rows), the complex boosting algorithm may have slightly overfitted compared to Logistic Regression. |
