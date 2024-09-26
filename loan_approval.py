# Loan Approval Prediction Project with Specified Features

import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('loan_approval_dataset.csv')

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Check for null values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Data Preprocessing
# Drop 'loan_id' as it is not useful for prediction
data = data.drop('loan_id', axis=1)

# Identify numerical and categorical columns
numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                  'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                  'luxury_assets_value', 'bank_asset_value']

categorical_cols = ['education', 'self_employed']

# Handle missing values
# Impute numerical columns with mean
imputer_num = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

# Impute categorical columns with mode
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Verify that there are no missing values
print("\nMissing values after imputation:")
print(data.isnull().sum())

# Encoding Categorical Variables
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Feature Scaling
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Define features and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Encode the target variable if it's categorical
if y.dtype == 'object':
    y = le.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation Function
def model_evaluation(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    # Classification Report
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model.__class__.__name__} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Logistic Regression
print("Logistic Regression Results:")
lr = LogisticRegression()
model_evaluation(lr, X_train, X_test, y_train, y_test)

# Decision Tree Classifier
print("Decision Tree Classifier Results:")
dt = DecisionTreeClassifier(random_state=42)
model_evaluation(dt, X_train, X_test, y_train, y_test)

# Random Forest Classifier
print("Random Forest Classifier Results:")
rf = RandomForestClassifier(random_state=42)
model_evaluation(rf, X_train, X_test, y_train, y_test)

# Cross-Validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean() * 100:.2f}%")

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# Evaluate the best model
print("Best Random Forest Classifier Results:")
model_evaluation(best_rf, X_train, X_test, y_train, y_test)

# Function to predict loan approval for a new applicant
def predict_loan_approval(applicant_data):
    # Create a DataFrame from the applicant data
    applicant_df = pd.DataFrame([applicant_data], columns=X.columns)
    # Encode categorical variables
    for col in categorical_cols:
        le.fit(data[col])
        applicant_df[col] = le.transform(applicant_df[col])
    # Feature scaling
    applicant_df[numerical_cols] = scaler.transform(applicant_df[numerical_cols])
    # Predict using the best model
    prediction = best_rf.predict(applicant_df)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'
    print(f"\nLoan Application Prediction: {result}")


