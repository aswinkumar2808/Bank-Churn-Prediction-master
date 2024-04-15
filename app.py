import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
rf_model = joblib.load('models/nate_random_forest.sav')
loaded_models = {'rf': rf_model}

def decode(pred):
    if pred == 1:
        return 'Customer Exits'
    else:
        return 'Customer Stays'

@app.route('/')
def home():
    return render_template('index.html', predictions={})

@app.route('/predict_from_file', methods=['POST'])
def predict_from_file():
    predictions = {}
    if 'user_file' in request.files:
        user_file = request.files['user_file']
        data = pd.read_csv(user_file)
        for idx, row in data.iterrows():
            user_data = row[['CreditScore', 'Country', 'Gender', 'Age', 'Tenure', 'Balance', 'Number of Accounts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
            prediction = rf_model.predict([user_data])[0]
            predictions[idx+1] = {'name': row['Name'], 'prediction': decode(prediction)}
    return render_template('index.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
