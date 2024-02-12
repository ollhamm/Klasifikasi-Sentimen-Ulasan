from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Added CORS middleware to allow requests from the frontend

# Load model that have been trained
model = joblib.load('models/sentiment_classifier.pkl')

# Route request from frontend
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = app.make_default_options_response()
    else:
        # Handle actual POST request
        data = request.get_json()
        review = data.get('review', '')  # Use .get() to avoid KeyError
        prediction = int(model.predict([review])[0])
        sentiment = "Negative" if prediction == 1 else "Positive" # Positive or Negative 
        response = jsonify({'sentiment': sentiment})

    response.headers['Access-Control-Allow-Origin'] = '*'  # allows access from all origins
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST'

    return response

if __name__ == '__main__':
    app.run(debug=True)
