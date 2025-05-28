import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mematikan warning TF
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Menonaktifkan GPU

from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Create static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load model dan scaler
model = load_model('diabetes_model.h5')
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Extract values from form
        data = [float(request.form[feature]) for feature in features]
        
        # Convert to numpy array and reshape
        input_array = np.array([data])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = float(prediction[0][0])
        
        result = {
            'probability': round(probability * 100, 2),
            'prediction': 'Positive' if probability > 0.5 else 'Negative'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 