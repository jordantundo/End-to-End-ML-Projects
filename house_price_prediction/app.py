from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X)
        
        # Return prediction
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
