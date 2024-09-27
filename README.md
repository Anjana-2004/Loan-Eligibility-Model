# Loan-Eligibility-Model
The Loan Eligibility Model uses a Decision Tree Classifier to predict loan approval based on features like income, loan amount, CIBIL score, and education level. It preprocesses data by handling missing values and encoding variables, providing interpretable predictions for assessing loan eligibility for new applicants.

## Features
- User-friendly form to input loan details.
- Integration of a trained machine learning model for loan approval prediction.
- Real-time result of loan approval status (Approved/Rejected) based on user inputs.

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.x
- Flask
- scikit-learn
- TensorFlow (if used for the deep learning model)
- pandas

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/loan-approval-flask-app.git
cd loan-approval-flask-app
```

### 2. Install Dependencies
Before running the application, install the necessary dependencies.

### 3. Run the Application
Start the Flask development server by running the following command:
```bash
python app.py
```

The application should now be running at `http://127.0.0.1:5000/` in your web browser.

## Application Structure

- **app.py**: Contains the Flask application routes and logic.
- **loan_approval.py**: The script where the machine learning model is trained and loaded.
- **loan_approval_final.py**: This script might have the final trained model or additional tweaks for the prediction.
- **templates/**: 
  - **index.html**: The web page that presents the form for users to input loan details.
  - **result.html**: Displays the prediction results ('Approved' or 'Rejected') after form submission.
- **static/**: Contains static files such as CSS or images (if any).

## Usage
1. Open the web browser and navigate to `http://127.0.0.1:5000/`.
2. Fill in the loan details in the provided form (income, loan amount, etc.) on the **index.html** page.
3. Submit the form to receive a prediction: whether the loan will be 'Approved' or 'Rejected' on the **result.html** page.

## Model Details
The machine learning model predicts loan approval using various parameters:
- **Income**
- **Loan Amount**
- **Loan Term**
- **Credit History**
- **Property Area**
- ...and other relevant factors.

