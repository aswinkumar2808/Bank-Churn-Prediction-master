import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load your machine learning model
rf_model = joblib.load('models/nate_random_forest.sav')

def decode(pred):
    if pred == 1:
        return 'Customer Exits'
    else:
        return 'Customer Stays'

@app.route('/')
def home():
    return render_template('index.html', predictions=None)

@app.route('/predict_from_file', methods=['POST'])
def predict_from_file():
    if 'user_file' in request.files:
        user_file = request.files['user_file']
        data = pd.read_csv(user_file)
        data.rename(columns={'Numbe of Accounts': 'Number of Accounts'}, inplace=True)
        predictions = []
        for idx, row in data.iterrows():
            user_data = row[['CreditScore', 'Country', 'Gender', 'Age', 'Tenure', 'Balance', 'Number of Accounts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
            prediction = rf_model.predict([user_data])[0]
            predictions.append(decode(prediction))
        data['Prediction'] = predictions
        output_filename = 'predicted_' + user_file.filename
        data.to_csv(output_filename, index=False)
        return f'Predictions saved to {output_filename}'
    return 'No file uploaded'

if __name__ == "__main__":
    app.run(debug=True)
