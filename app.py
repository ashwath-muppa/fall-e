import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import time
import json
import os
from datetime import datetime

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Global variables
SEQUENCE_LENGTH = 30  # Reduced from 100
NUM_FEATURES = 3
DEFAULT_SAMPLING_RATE = 100  # Hz
PREDICTION_THRESHOLD = 0.6

# Define locations with their specific characteristics
LOCATIONS = {
    'san_francisco': {
        'name': 'San Francisco, CA',
        'description': 'Located near the San Andreas Fault',
        'earthquake_probability': 0.25,  # Higher probability
        'frequency_range': (3.0, 6.0),   # Higher frequency for fault zones
        'amplitude_range': (0.7, 1.8),   # Higher amplitude
        'status': 'online'
    },
    'tokyo': {
        'name': 'Tokyo, Japan',
        'description': 'Located in the Pacific Ring of Fire',
        'earthquake_probability': 0.30,   # Higher probability
        'frequency_range': (2.5, 5.5),    
        'amplitude_range': (0.6, 1.7),
        'status': 'online'
    },
    'mexico_city': {
        'name': 'Mexico City, Mexico',
        'description': 'Built on an ancient lake bed, amplifies seismic waves',
        'earthquake_probability': 0.20,
        'frequency_range': (2.0, 5.0),
        'amplitude_range': (0.5, 1.6),
        'status': 'online'
    },
    'istanbul': {
        'name': 'Istanbul, Turkey',
        'description': 'North Anatolian Fault Zone',
        'earthquake_probability': 0.22,
        'frequency_range': (2.2, 5.2),
        'amplitude_range': (0.6, 1.5),
        'status': 'online'
    },
    'los_angeles': {
        'name': 'Los Angeles, CA',
        'description': 'Southern San Andreas Fault system',
        'earthquake_probability': 0.23,
        'frequency_range': (2.8, 5.8),
        'amplitude_range': (0.6, 1.6),
        'status': 'online'
    },
    'christchurch': {
        'name': 'Christchurch, New Zealand',
        'description': 'Located on the Pacific and Australian plate boundary',
        'earthquake_probability': 0.18,
        'frequency_range': (1.8, 4.8),
        'amplitude_range': (0.4, 1.4),
        'status': 'online'
    },
    'kathmandu': {
        'name': 'Kathmandu, Nepal',
        'description': 'Located in the Himalayan seismic belt',
        'earthquake_probability': 0.15,
        'frequency_range': (1.5, 4.5),
        'amplitude_range': (0.4, 1.3),
        'status': 'maintenance'  # Example of a station in maintenance
    },
    'quito': {
        'name': 'Quito, Ecuador',
        'description': 'Located in the Andean Volcanic Belt',
        'earthquake_probability': 0.17,
        'frequency_range': (1.7, 4.7),
        'amplitude_range': (0.4, 1.4),
        'status': 'online'
    }
}

# Mock model until a real one is loaded
class MockBiLSTMModel:
    def predict(self, data):
        # Simulates prediction with some randomness but favors no earthquake
        # Returns shape (1, 2) for binary classification
        random_val = np.random.random()
        if random_val > 0.8:  # 20% chance of earthquake
            return np.array([[0.3, 0.7]])
        else:
            return np.array([[0.9, 0.1]])

# Try to load model or use mock model
try:
    model = load_model('model/bilstm_model.h5')
    print("Loaded BiLSTM model successfully")
except Exception as e:
    print(f"Could not load model: {e}")
    print("Using mock model instead")
    model = MockBiLSTMModel()

# Simplify data generation by reducing complexity
def generate_realistic_seismic_data(sequence_length=30, is_earthquake=False, location='san_francisco'):
    """
    Generate realistic seismic data with proper seismic wave characteristics
    
    Parameters:
        sequence_length: Number of data points to generate
        is_earthquake: Whether to generate earthquake or background data
        location: The location ID to use specific parameters
    
    Returns:
        NumPy array of shape (sequence_length, NUM_FEATURES) representing
        3-component seismic data (vertical, N-S, E-W components)
    """
    location_params = LOCATIONS.get(location, LOCATIONS['san_francisco'])
    
    # Create time array with proper sampling rate (e.g., 100 Hz)
    dt = 1.0 / DEFAULT_SAMPLING_RATE
    t = np.linspace(0, dt * (sequence_length - 1), sequence_length)
    
    # Initialize 3-component data (vertical, north-south, east-west)
    data = np.zeros((sequence_length, NUM_FEATURES))
    
    if is_earthquake:
        # Get location-specific parameters
        freq_min, freq_max = location_params['frequency_range']
        amp_min, amp_max = location_params['amplitude_range']
        
        # Generate parameters for P-wave, S-wave, and surface waves
        p_wave_amplitude = np.random.uniform(amp_min * 0.6, amp_max * 0.6)
        s_wave_amplitude = np.random.uniform(amp_min * 0.8, amp_max * 0.8)
        surface_wave_amplitude = np.random.uniform(amp_min, amp_max)
        
        # Wave frequencies
        p_wave_freq = np.random.uniform(freq_max * 0.8, freq_max)  # P-waves have higher frequency
        s_wave_freq = np.random.uniform(freq_min * 1.2, freq_max * 0.8)  # S-waves mid-frequency
        surface_wave_freq = np.random.uniform(freq_min, freq_min * 1.2)  # Surface waves lower frequency
        
        # Arrival times (P-wave first, then S-wave, then surface waves)
        p_arrival = int(sequence_length * 0.2)  # P-wave arrives at ~20% of the sequence
        s_arrival = int(sequence_length * 0.4)  # S-wave arrives at ~40% of the sequence
        surface_arrival = int(sequence_length * 0.6)  # Surface waves arrive at ~60% of the sequence
        
        # Generate realistic wave envelopes
        for i in range(sequence_length):
            # Gaussian envelope functions for each wave type
            if i >= p_arrival:
                p_envelope = np.exp(-0.5 * ((i - p_arrival) / (sequence_length * 0.15)) ** 2)
                p_wave = p_wave_amplitude * p_envelope * np.sin(2 * np.pi * p_wave_freq * t[i])
            else:
                p_wave = 0
                
            if i >= s_arrival:
                s_envelope = np.exp(-0.3 * ((i - s_arrival) / (sequence_length * 0.2)) ** 2)
                s_wave = s_wave_amplitude * s_envelope * np.sin(2 * np.pi * s_wave_freq * t[i])
            else:
                s_wave = 0
                
            if i >= surface_arrival:
                surface_envelope = np.exp(-0.1 * ((i - surface_arrival) / (sequence_length * 0.3)) ** 2)
                surface_wave = surface_wave_amplitude * surface_envelope * np.sin(2 * np.pi * surface_wave_freq * t[i])
            else:
                surface_wave = 0
            
            # Combine waves with different orientations for each component
            # Vertical component (stronger P-waves, weaker S and surface)
            data[i, 0] = 1.0 * p_wave + 0.3 * s_wave + 0.5 * surface_wave
            # North-South component (stronger S-waves)
            data[i, 1] = 0.2 * p_wave + 1.0 * s_wave + 0.7 * surface_wave
            # East-West component (stronger surface waves)
            data[i, 2] = 0.2 * p_wave + 0.7 * s_wave + 1.0 * surface_wave
        
        # Add reasonable measurement uncertainty/sensor noise
        noise_level = np.random.uniform(0.02, 0.05)
        data += noise_level * np.random.randn(sequence_length, NUM_FEATURES)
            
    else:
        # Background microseismic noise and ambient vibrations
        # Different frequencies for different noise sources (traffic, wind, ocean, human activity)
        noise_sources = [
            (0.01, 0.5),    # Very low frequency background noise
            (1.0, 5.0),     # Medium frequency ambient vibrations
            (8.0, 12.0)     # Higher frequency local noise
        ]
        
        # Generate superposition of different noise sources
        for freq_range in noise_sources:
            freq_min, freq_max = freq_range
            amplitude = np.random.uniform(0.01, 0.08)  # Lower amplitude for background noise
            freq = np.random.uniform(freq_min, freq_max)
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Add slightly different noise to each component
            data[:, 0] += amplitude * np.sin(2 * np.pi * freq * t + phase)
            data[:, 1] += amplitude * 0.8 * np.sin(2 * np.pi * freq * 1.1 * t + phase + 0.5)
            data[:, 2] += amplitude * 0.9 * np.sin(2 * np.pi * freq * 0.9 * t + phase + 1.0)
        
        # Add random sensor noise (Gaussian)
        data += 0.02 * np.random.randn(sequence_length, NUM_FEATURES)
    
    return data

def preprocess_and_predict(data):
    """
    Preprocess input data and make prediction using the loaded model
    """
    # Reshape data for model input
    processed_data = np.array(data).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    
    # Generate prediction
    prediction = model.predict(processed_data)
    
    # Calculate confidence
    confidence = float(np.max(prediction))
    
    # Get the predicted class
    predicted_class = int(np.argmax(prediction))
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'prediction_text': "Earthquake" if predicted_class == 1 else "No Earthquake",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def index():
    return render_template('index.html', locations=LOCATIONS)

@app.route('/locations', methods=['GET'])
def get_locations():
    """
    Return the list of available locations
    """
    return jsonify(LOCATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate sample data and make prediction for a specific location
    """
    # Parse request
    data = request.json if request.is_json else {}
    should_generate_earthquake = data.get('force_earthquake', False)
    location_id = data.get('location', 'san_francisco')
    
    # Check if location exists
    if location_id not in LOCATIONS:
        return jsonify({'error': 'Invalid location ID'}), 400
        
    # Check if location is in maintenance
    if LOCATIONS[location_id]['status'] != 'online':
        return jsonify({
            'error': 'Location offline',
            'message': f"The {LOCATIONS[location_id]['name']} monitoring station is currently {LOCATIONS[location_id]['status']}."
        }), 503
    
    # Generate realistic seismic data
    seismic_data = generate_realistic_seismic_data(
        is_earthquake=should_generate_earthquake,
        location=location_id
    )
    
    # Make prediction
    result = preprocess_and_predict(seismic_data)
    
    # Add the generated data and location info to the response
    result['data'] = seismic_data.tolist()
    result['location'] = {
        'id': location_id,
        'name': LOCATIONS[location_id]['name'],
        'description': LOCATIONS[location_id]['description']
    }
    
    # Return JSON response
    return jsonify(result)

@app.route('/stream-data', methods=['GET'])
def stream_data():
    """
    Generate streaming data for real-time visualization
    """
    # Get location from query parameters
    location_id = request.args.get('location', 'san_francisco')
    
    # Check if location exists
    if location_id not in LOCATIONS:
        return jsonify({'error': 'Invalid location ID'}), 400
        
    # Check if location is in maintenance
    if LOCATIONS[location_id]['status'] != 'online':
        return jsonify({
            'error': 'Location offline',
            'message': f"The {LOCATIONS[location_id]['name']} monitoring station is currently {LOCATIONS[location_id]['status']}."
        }), 503
    
    # Determine if we should generate earthquake data (for demo purposes)
    should_generate_earthquake = request.args.get('force_earthquake', 'false').lower() == 'true'
    
    # Generate data
    seismic_data = generate_realistic_seismic_data(
        is_earthquake=should_generate_earthquake,
        location=location_id
    )
    
    # Return data points as JSON
    return jsonify({
        'data': seismic_data.tolist(),
        'timestamp': time.time(),
        'location': {
            'id': location_id,
            'name': LOCATIONS[location_id]['name']
        }
    })

# New endpoint to get statistics for a location
@app.route('/location-stats/<location_id>', methods=['GET'])
def location_stats(location_id):
    """
    Return historical statistics for a specific location
    """
    # Check if location exists
    if location_id not in LOCATIONS:
        return jsonify({'error': 'Invalid location ID'}), 400
    
    # In a real application, these would be fetched from a database
    # For demo purposes, generate random stats
    location = LOCATIONS[location_id]
    
    # Generate mock historical data
    last_month_events = np.random.randint(1, 10)
    avg_magnitude = round(np.random.uniform(2.0, 5.5), 1)
    risk_level = "High" if location['earthquake_probability'] > 0.25 else \
                "Medium" if location['earthquake_probability'] > 0.15 else "Low"
    
    stats = {
        'location_id': location_id,
        'location_name': location['name'],
        'description': location['description'],
        'status': location['status'],
        'last_month_events': last_month_events,
        'average_magnitude': avg_magnitude,
        'risk_level': risk_level,
        'historical_data': [
            {
                'date': (datetime.now().replace(day=1) - datetime.timedelta(days=30*i)).strftime('%Y-%m'),
                'events': np.random.randint(0, 12),
                'max_magnitude': round(np.random.uniform(2.0, 6.0), 1)
            } for i in range(1, 7)  # Last 6 months
        ]
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)