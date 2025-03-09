from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from pyts.image import GramianAngularField
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import threading
import time
import random

app = Flask(__name__)

# Load the pre-trained ResNet + BiLSTM model
bilstm_model = tf.keras.models.load_model("backend/model/bilstm_ica_model.h5")

# Prepare GAF transformation
gaf = GramianAngularField(method='summation', image_size=100)

# Standardizer and ICA for feature extraction
scaler = StandardScaler()
ica = FastICA(n_components=128, random_state=42)

# Global storage for sensor data
sensor_data = []

# Function to generate random sensor data (simulating incoming data from a sensor)
def generate_random_sensor_data():
    while True:
        data = {
            "sensor_id": "Sensor_001",
            "timestamp": time.time(),
            "aX": random.randint(-20000, 20000),
            "aY": random.randint(-20000, 20000),
            "aZ": random.randint(-20000, 20000),
        }
        # Send the random data to Flask API
        sensor_data.append([data['aX'], data['aY'], data['aZ']])
        time.sleep(0.1)  # Simulate data collection every 100ms

# Function to preprocess the data (convert to GAF, normalize, ICA)
def preprocess_data(sensor_batch):
    # Convert the batch into GAF images
    gaf_images = gaf.fit_transform(sensor_batch)
    
    # Reshape and standardize
    gaf_images = gaf_images.reshape(-1, 100, 100, 1)  # Reshape for CNN input (height, width, channels)
    gaf_images = np.repeat(gaf_images, 3, axis=-1)  # Convert grayscale to 3 channels (RGB)
    
    # Extract deep features using the pre-trained model
    features = bilstm_model.predict(gaf_images)  # ResNet + BiLSTM model output
    
    # Apply ICA to reduce features to 64 dimensions
    features = ica.fit_transform(features)
    return features

# Function to process every 100 sensor readings
def process_data():
    global sensor_data
    while True:
        if len(sensor_data) >= 100:
            # Extract a batch of 100 readings
            batch = np.array(sensor_data[:100])
            sensor_data = sensor_data[100:]  # Remove processed data
            
            # Preprocess data (convert to GAF, extract features)
            features = preprocess_data(batch)
            
            # Average the features (we can also use a different method to aggregate features if needed)
            avg_features = np.mean(features, axis=0)

            # Predict using the trained BiLSTM model
            prediction = bilstm_model.predict(np.expand_dims(avg_features, axis=0))
            result = "Earthquake Detected" if np.argmax(prediction) == 1 else "No Earthquake"

            print(f"Prediction: {result}")

        time.sleep(1)  # Process every second

# Start data generation thread
threading.Thread(target=generate_random_sensor_data, daemon=True).start()

# Start data processing thread
threading.Thread(target=process_data, daemon=True).start()

# Endpoint to receive sensor data
@app.route('/sensor-data', methods=['POST'])
def receive_data():
    data = request.json
    sensor_data.append([data['aX'], data['aY'], data['aZ']])
    return jsonify({"status": "Received"}), 200

# Endpoint to get current sensor stats
@app.route('/get-stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_samples": len(sensor_data),
        "latest_reading": sensor_data[-1] if sensor_data else "No Data"
    })

if __name__ == '__main__':
    app.run(debug=True)
