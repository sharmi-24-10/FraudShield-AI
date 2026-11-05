# FraudShield AI – Financial Fraud Detection Using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare supervised machine learning models that identify potentially fraudulent credit card transactions. This is a binary classification problem using the target variable `Class`:

- **0:** Legitimate transaction.
- **1:** Fraudulent transaction.

The project compares five machine learning classification algorithms using multiple evaluation metrics. Since the dataset contains a highly imbalanced target distribution, where fraudulent transactions represent a very small proportion of the total transactions, evaluating models using accuracy alone may not provide a complete understanding of their performance.

Therefore, Accuracy, ROC-AUC, Precision, Recall, F1-score, PR-AUC and Matthews Correlation Coefficient (MCC) are used to evaluate and compare the classifiers.

The trained models are deployed through a Streamlit web application, allowing users to upload test data, select a trained model and evaluate its performance without retraining.

## b. Dataset Description

**Dataset:** Credit Card Fraud Detection Dataset  
**Source:** Kaggle  
**Dataset URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transactions made by European cardholders over a two-day period in September 2013. It includes anonymized transaction features obtained through Principal Component Analysis (PCA), along with transaction time, amount and the fraud classification label.

The dataset contains:

- Total records: 284,807
- Input features: 30
- Target variable: Class
- Classification type: Binary classification
- Target classes: 0 (Legitimate) and 1 (Fraudulent)
- Fraudulent transactions: 492
- Legitimate transactions: 284,315

The input features include `Time`, `Amount` and 28 anonymized PCA-transformed features (`V1` to `V28`).

The target variable `Class` identifies whether a transaction is legitimate or fraudulent.

### Preprocessing

The following preprocessing steps were applied:

- The dataset was checked for missing values and duplicate records.
- The input features were separated from the target variable.
- Stratified sampling was used to split the dataset into 80% training data and 20% testing data while preserving the original class distribution.
- The training dataset was undersampled to reduce class imbalance, retaining all fraudulent transactions and 10,000 legitimate transactions.
- StandardScaler was applied to numerical features in models that require feature scaling.
- The testing dataset was kept separate from the undersampling process to preserve the original class distribution.
- The same preprocessing pipeline was applied during model training and evaluation.

## c. Project Links

**GitHub Repository:**  
https://github.com/sharmi-24-10/FraudShield-AI

**Live Streamlit Application:**  
https://fraudshield-ai-sharmi.streamlit.app/

The GitHub repository contains the source code, Jupyter Notebook, requirements file, test data, saved trained models and Streamlit application.

The project is deployed using Streamlit Community Cloud.

## d. Models Used

The five classification models implemented in this project are:

1. **Logistic Regression** – A probabilistic linear classification algorithm used to estimate the probability of a credit card transaction being fraudulent.

2. **Decision Tree Classifier** – A tree-based classification algorithm that learns decision rules from transaction features to classify fraudulent and legitimate transactions.

3. **K-Nearest Neighbors (KNN)** – A classification algorithm that predicts the class of a transaction based on the nearest training samples.

4. **Gaussian Naive Bayes** – A probabilistic classifier based on Bayes' theorem and the assumption of conditional independence between features.

5. **Support Vector Machine (SVM)** – A classification algorithm that learns a decision boundary to distinguish fraudulent transactions from legitimate transactions.

### Model Comparison Table

| ML Model Name | Accuracy | ROC-AUC | Precision | Recall | F1-score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| Decision Tree | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| KNN | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| Gaussian Naive Bayes | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |
| Support Vector Machine | To be updated | To be updated | To be updated | To be updated | To be updated | To be updated |

*Note: Replace the placeholders with the actual evaluation results from `model_comparison.csv`.*

### Evaluation Metrics

- **Accuracy:** Proportion of all predictions that are correctly classified.
- **ROC-AUC:** Measures the model's ability to distinguish between fraudulent and legitimate transactions across different classification thresholds.
- **Precision:** Proportion of transactions predicted as fraudulent that are actually fraudulent.
- **Recall:** Proportion of actual fraudulent transactions correctly identified by the model.
- **F1-score:** Harmonic mean of Precision and Recall.
- **PR-AUC:** Area under the Precision-Recall curve, particularly useful for evaluating models on highly imbalanced datasets.
- **MCC:** Matthews Correlation Coefficient, measuring the overall agreement between predicted and actual classes using all four confusion matrix outcomes.

### Results and Best-Model Observation

The performance of all five classification models is compared using the selected evaluation metrics.

The following observations are considered when identifying the best-performing model:

- Highest Accuracy: Model with the highest overall correct classification rate.
- Highest ROC-AUC: Model with the strongest ability to distinguish between the two classes.
- Highest Precision: Model with fewer false-positive fraud predictions.
- Highest Recall: Model that identifies the largest proportion of actual fraudulent transactions.
- Highest F1-score: Model providing a balance between Precision and Recall.
- Highest PR-AUC: Model with the strongest Precision-Recall performance.
- Highest MCC: Model with the strongest overall classification agreement.

### Overall Winner for the Dataset

The overall best-performing model will be selected based on the evaluation results, considering the importance of identifying fraudulent transactions while controlling false-positive predictions.

Since fraud detection involves highly imbalanced data, Recall, PR-AUC, F1-score and MCC are important indicators in addition to Accuracy and ROC-AUC.

The final model selection will depend on the actual evaluation results and the intended fraud detection objective.

## e. Streamlit Application

The FraudShield AI web application provides the following functionalities:

- Dataset upload through CSV files.
- Machine learning model selection.
- Evaluation using multiple performance metrics.
- Confusion matrix visualization.
- Classification report.
- Downloadable prediction results.
- Dataset exploration and model comparison.
- Project methodology and workflow explanation.
- Batch prediction with progress tracking.

The application loads saved trained models and evaluates uploaded test data without retraining.

**Live Application:**  
https://fraudshield-ai-sharmi.streamlit.app/

## f. Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## g. Project Structure

```text
FraudShield-AI/
│
├── app.py
├── requirements.txt
├── model_comparison.csv
│
├── dataset/
│   └── creditcard.csv (local dataset; not uploaded to GitHub)
│
├── notebooks/
│   └── FraudShield_AI_Assignment2.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
│
└── test_data.csv
```


**GitHub:** https://github.com/sharmi-24-10/FraudShield-AI  
**Live Application:** https://fraudshield-ai-sharmi.streamlit.app/