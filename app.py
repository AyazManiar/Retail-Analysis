from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

# Load the saved model
loaded_model = joblib.load('customer_segmentation_model.pkl')


# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.get_json()

        # Validate the input
        if 'Total_Purchases' not in data or 'Total_Amount' not in data:
            return jsonify({'error': 'Missing required keys'}), 400

        total_purchases = data['Total_Purchases']
        total_amount = data['Total_Amount']

        # Create a DataFrame for input data (simulating how you process data)
        input_data = pd.DataFrame({
            'Total_Purchases': [total_purchases],
            'Total_Amount': [total_amount]
        })

        # Predict the cluster using the model
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape for prediction
        prediction = loaded_model.predict(input_data_reshaped)

        # Convert numerical predictions to human-readable labels
        if prediction == 0:
            predicted_cluster = 'Low Value Customer'
        elif prediction == 1:
            predicted_cluster = 'Mid Value Customer'
        else:
            predicted_cluster = 'High Value Customer'

        return jsonify({'Predicted_Cluster': predicted_cluster})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
