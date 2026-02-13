# ML Assignment 2 - Classification Analysis

## Problem Statement
To implement 6 classification models, compare their performance on a chosen dataset, and deploy a Streamlit web application.

## [cite_start]Dataset Description [cite: 68]
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
* **Source:** Kaggle / UCI
* **Features:** 30 features
* **Instances:** 569 samples

## [cite_start]Models Used & Comparison [cite: 69, 70]
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9975 | 0.9737 | 0.9737 | 0.9737 | 0.9416 |
| Decision Tree | 0.9474 | 0.9431 | 0.9474 | 0.9474 | 0.9474 | 0.8847 |
| KNN | 0.9474 | 0.9804 | 0.9482 | 0.9474 | 0.9472 | 0.8845 |
| Naive Bayes | 0.9649 | 0.9961 | 0.9652 | 0.9649 | 0.9648 | 0.9228 |
| Random Forest | 0.9649 | 0.9953 | 0.9652 | 0.9649 | 0.9649 | 0.9228 |
| XGBoost | 0.9561 | 0.9940 | 0.9565 | 0.9561 | 0.9560 | 0.9038 |

## [cite_start]Observations [cite: 79]
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed best with highest F1 score and AUC due to linear separability of data. |
| Decision Tree | Showed slightly lower performance due to potential overfitting on the training split. |
| KNN | Good performance but sensitive to scaling; slightly lower AUC than ensemble methods. |
| Naive Bayes | Surprisingly high recall, indicating features largely follow Gaussian distribution. |
| Random Forest | Robust performance, matching Naive Bayes but with higher training time. |
| XGBoost | High accuracy, though slightly more complex to tune than Logistic Regression for this dataset. |
