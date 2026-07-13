
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model pipeline
try:
    with open('model.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Convert input data to DataFrame, ensuring correct order and columns
        # It's crucial that the input features match the training features
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model_pipeline.predict(input_df)

        # Return the prediction
        return jsonify({'predicted_units_sold': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development: app.run(debug=True)
    # For deployment, listen on all public IPs
    app.run(host='0.0.0.0', port=5000)
