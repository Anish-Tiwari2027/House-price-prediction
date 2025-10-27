from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # allows frontend (like React) to connect

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return "House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from frontend
    area = data['Area']
    bedrooms = data['Bedrooms']
    age = data['Age']

    # Prepare data for model
    input_data = pd.DataFrame([[area, bedrooms, age]], columns=['Area', 'Bedrooms', 'Age'])

    # Predict
    prediction = model.predict(input_data)[0]
    return jsonify({'Predicted_Price (lakhs)': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
