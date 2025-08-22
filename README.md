# Loan Approval Prediction System

![Homepage Banner](C:\Users\Anjana\OneDrive\Desktop\Loan Approval Prediction)

## Project Overview
This project is a Machine Learning–based Loan Approval Prediction System that predicts whether a loan application will be Approved or Rejected based on applicant details.  
It aims to assist financial institutions in making data-driven decisions by analyzing key applicant information such as income, dependents, loan amount, credit score, and asset values.

---

## Dataset
- **Source**: `loan_approval_dataset.csv`  
- **Target Variable**: `loan_status` (Approved / Rejected)  
- **Features**:  
  - Numerical:  
    - `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`,  
      `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`  
  - Categorical:  
    - `education` (Graduate / Not Graduate)  
    - `self_employed` (Yes / No)

---

## Data Preprocessing
1. Dropped irrelevant column: `loan_id`.  
2. Handled missing values:  
   - Numerical features imputed with **mean**.  
   - Categorical features imputed with **mode**.  
3. Encoding categorical variables using **Label Encoding**.  
4. Standardized numerical features using **StandardScaler**.  
5. Split the dataset into **Training (80%)** and **Testing (20%)** sets.  

---

## Models Implemented
The following algorithms were trained and evaluated:

1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **Random Forest Classifier**  
4. **XGBoost Classifier**

Each model was compared using accuracy, confusion matrix, and classification reports.  
Random Forest and XGBoost provided the most reliable performance.

---

## Model Evaluation
- Metrics used:  
  - Accuracy Score  
  - Confusion Matrix (visualized using heatmap)  
  - Classification Report (Precision, Recall, F1-score)  
- Cross-validation was applied for better generalization.  
- Hyperparameter tuning was conducted using `GridSearchCV` for Random Forest to identify the best parameters.

---

## Results
- Logistic Regression: Baseline performance  
- Decision Tree: Improved interpretability, moderate accuracy  
- Random Forest: High accuracy and robustness  
- XGBoost: Achieved the most consistent and strong predictive performance  

---

## Project Workflow
1. Load dataset  
2. Data preprocessing (missing values, encoding, scaling)  
3. Train-test split  
4. Train multiple ML algorithms  
5. Evaluate models  
6. Hyperparameter tuning  
7. Final model selection (XGBoost / Random Forest)  
8. Prediction on new applicants  

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
