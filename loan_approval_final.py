import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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

# Manually encode 'education' column into 0 and 1 (Graduate = 1, Not Graduate = 0)
le_education = LabelEncoder()
data['education'] = le_education.fit_transform(data['education'].str.strip())  # Remove extra spaces

# Label encode 'self_employed'
le_self_employed = LabelEncoder()
data['self_employed'] = le_self_employed.fit_transform(data['self_employed'].str.strip())

# Save the label encoders
with open('label_encoder_education.pkl', 'wb') as f:
    pickle.dump(le_education, f)

with open('label_encoder_self_employed.pkl', 'wb') as f:
    pickle.dump(le_self_employed, f)

# Feature Scaling
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Define features and target variable
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Encode the target variable if it's categorical
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
print("Training Decision Tree Classifier...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluate the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the scaler and model
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(dt, f)

# Save the label encoder for target variable (if needed)
if 'le_target' in locals():
    with open('label_encoder_target.pkl', 'wb') as f:
        pickle.dump(le_target, f)
