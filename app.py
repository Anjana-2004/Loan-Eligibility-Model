from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and encoders
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_education = pickle.load(open('label_encoder_education.pkl', 'rb'))
le_self_employed = pickle.load(open('label_encoder_self_employed.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    education = request.form['education'].strip()  # e.g., "Graduate" or "Not Graduate"
    self_employed = request.form['self_employed'].strip()  # e.g., "Yes" or "No"
    no_of_dependents = float(request.form['no_of_dependents'])
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    cibil_score = float(request.form['cibil_score'])
    residential_assets_value = float(request.form['residential_assets_value'])
    commercial_assets_value = float(request.form['commercial_assets_value'])
    luxury_assets_value = float(request.form['luxury_assets_value'])
    bank_asset_value = float(request.form['bank_asset_value'])

    # Encode categorical features
    education_encoded = le_education.transform([education])[0]
    self_employed_encoded = le_self_employed.transform([self_employed])[0]

    # Prepare data for scaling (only numerical columns)
    numerical_data = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                                residential_assets_value, commercial_assets_value, luxury_assets_value,
                                bank_asset_value]])

    # Scale the numerical data
    numerical_data_scaled = scaler.transform(numerical_data)

    # Combine scaled numerical data with encoded categorical data
    input_data_scaled = np.hstack([numerical_data_scaled, [[education_encoded, self_employed_encoded]]])

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Convert prediction to readable output (Assume 0 = Rejected, 1 = Approved)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'

    return render_template('result.html', prediction_text=f'Loan Status: {result}')

if __name__ == "__main__":
    app.run(debug=True)
