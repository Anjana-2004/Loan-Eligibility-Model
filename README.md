# Loan Approval Prediction System

![Homepage Banner](banner.png)

## Project Overview
This project is a Machine Learning–based Loan Approval Prediction System that predicts whether a loan application will be Approved or Rejected based on applicant details.  

The motivation behind this project is to automate the loan approval process, which traditionally requires manual checks of applicant background and creditworthiness. By leveraging data-driven models, financial institutions can reduce human error, speed up decision-making, and maintain consistency in approvals.  

The system analyzes multiple applicant features such as income, dependents, requested loan amount, loan term, credit score, and assets to determine the likelihood of approval. Different machine learning algorithms were tested to identify the most reliable predictor of loan approval status.

---

## Dataset
- **Source**: `loan_approval_dataset.csv`  
- **Target Variable**: `loan_status` (Approved / Rejected)  

### Features
- **Numerical**:  
  - `no_of_dependents` – Number of dependents supported by the applicant.  
  - `income_annum` – Annual income of the applicant.  
  - `loan_amount` – Loan amount requested.  
  - `loan_term` – Loan repayment term in months.  
  - `cibil_score` – Credit score of the applicant.  
  - `residential_assets_value` – Value of residential assets owned.  
  - `commercial_assets_value` – Value of commercial assets owned.  
  - `luxury_assets_value` – Value of luxury assets owned.  
  - `bank_asset_value` – Value of bank assets held by the applicant.  

- **Categorical**:  
  - `education` – Applicant’s education status (Graduate / Not Graduate).  
  - `self_employed` – Whether the applicant is self-employed (Yes / No).  

These features were carefully chosen as they directly contribute to assessing the applicant’s repayment ability and financial stability.

---

## Data Preprocessing
Data preprocessing was a crucial step to ensure the dataset was clean and suitable for training machine learning models.  

1. **Dropped Irrelevant Column**  
   - Removed `loan_id` since it does not contribute to prediction.  

2. **Handling Missing Values**  
   - Numerical features imputed with the **mean** of their respective columns.  
   - Categorical features imputed with the **most frequent value (mode)**.  

3. **Encoding Categorical Variables**  
   - Converted `education` and `self_employed` into numeric form using **Label Encoding** to make them usable by ML algorithms.  

4. **Feature Scaling**  
   - Standardized numerical columns using **StandardScaler** so that all numerical features are on the same scale and no feature dominates due to higher magnitude.  

5. **Train-Test Split**  
   - Dataset split into **80% training** and **20% testing** to evaluate model performance fairly.  

---

## Models Implemented
Several machine learning models were tested to identify the best-performing algorithm:  

1. **Logistic Regression** – A baseline linear model for binary classification.  
2. **Decision Tree Classifier** – A tree-based model that splits data into decision rules.  
3. **Random Forest Classifier** – An ensemble of multiple decision trees to reduce overfitting and improve accuracy.  
4. **XGBoost Classifier** – A powerful gradient boosting model known for high performance in classification problems.  

Random Forest and XGBoost delivered the most consistent and accurate results, proving to be the most reliable models for this dataset.

---

## Model Evaluation
To measure performance, the following metrics were used:  

- **Accuracy Score** – Overall percentage of correct predictions.  
- **Confusion Matrix** – Visual representation of correct and incorrect classifications.  
- **Classification Report** – Detailed precision, recall, and F1-score for each class.  
- **Cross-Validation** – Ensured models were robust across different subsets of data.  
- **Hyperparameter Tuning** – GridSearchCV was applied to Random Forest to optimize depth, number of trees, and minimum samples required for splits.  

---

## Results
- **Logistic Regression**: Established a baseline performance but limited due to its linear nature.  
- **Decision Tree**: Provided better interpretability but showed signs of overfitting.  
- **Random Forest**: Achieved high accuracy and maintained robustness, making it a strong candidate.  
- **XGBoost**: Delivered the best overall performance, with high accuracy and consistent results across validation runs.  

---

## Project Workflow
1. Load and explore dataset.  
2. Preprocess data (handle missing values, encode categorical data, scale features).  
3. Split data into training and testing sets.  
4. Train models (Logistic Regression, Decision Tree, Random Forest, XGBoost).  
5. Evaluate models using accuracy, classification report, and confusion matrix.  
6. Perform cross-validation for reliability.  
7. Apply hyperparameter tuning for Random Forest.  
8. Select the best-performing model (XGBoost / Random Forest).  
9. Use the model to predict outcomes for new loan applicants.  

---

## Prediction for New Applicants
The system allows loan approval prediction for new applicants. Example:

```python
applicant_data = {
    'no_of_dependents': 2,
    'income_annum': 500000,
    'loan_amount': 200000,
    'loan_term': 12,
    'cibil_score': 750,
    'residential_assets_value': 100000,
    'commercial_assets_value': 50000,
    'luxury_assets_value': 25000,
    'bank_asset_value': 150000,
    'education': 'Graduate',
    'self_employed': 'No'
}

predict_loan_approval(applicant_data)
# Output: Approved / Rejected
