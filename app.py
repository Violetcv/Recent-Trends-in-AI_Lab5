"""
Flask Backend API for Air Quality Monitoring System
Provides REST API endpoints for the frontend
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for frontend requests

# Load data and model on startup
def load_resources():
    """Load data and model"""
    try:
        locations = pd.read_csv('data/sensor_locations.csv')
        readings = pd.read_csv('data/air_quality_readings.csv')
        readings['timestamp'] = pd.to_datetime(readings['timestamp'])
        
        with open('models/air_quality_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        return locations, readings, model_data
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None

locations, readings, model_data = load_resources()

# Serve frontend
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/css/<path:path>')
def send_css(path):
    """Serve CSS files"""
    return send_from_directory('frontend/css', path)

@app.route('/js/<path:path>')
def send_js(path):
    """Serve JavaScript files"""
    return send_from_directory('frontend/js', path)

@app.route('/visualizations/<path:path>')
def send_visualizations(path):
    """Serve visualization images"""
    return send_from_directory('visualizations', path)

# API Endpoints
@app.route('/api/status')
def status():
    """API status check"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': readings is not None,
        'model_loaded': model_data is not None
    })

@app.route('/api/overview')
def get_overview():
    """Get overview statistics"""
    if readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    latest = readings.groupby('sensor_id').tail(1)
    
    return jsonify({
        'total_sensors': len(locations),
        'total_readings': len(readings),
        'avg_pm25': float(latest['pm25'].mean()),
        'avg_aqi': float(latest['aqi'].mean()),
        'max_aqi': float(latest['aqi'].max()),
        'date_range': {
            'start': readings['timestamp'].min().isoformat(),
            'end': readings['timestamp'].max().isoformat()
        },
        'pollutant_stats': {
            'pm25': {
                'min': float(readings['pm25'].min()),
                'max': float(readings['pm25'].max()),
                'mean': float(readings['pm25'].mean()),
                'std': float(readings['pm25'].std())
            },
            'pm10': {
                'min': float(readings['pm10'].min()),
                'max': float(readings['pm10'].max()),
                'mean': float(readings['pm10'].mean())
            },
            'no2': {
                'min': float(readings['no2'].min()),
                'max': float(readings['no2'].max()),
                'mean': float(readings['no2'].mean())
            }
        }
    })

@app.route('/api/sensors')
def get_sensors():
    """Get sensor locations and latest readings"""
    if locations is None or readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    latest = readings.groupby('sensor_id').tail(1)
    
    sensors_data = []
    for _, sensor in locations.iterrows():
        sensor_readings = latest[latest['sensor_id'] == sensor['sensor_id']]
        
        if len(sensor_readings) > 0:
            reading = sensor_readings.iloc[0]
            sensors_data.append({
                'id': sensor['sensor_id'],
                'name': sensor['location_name'],
                'latitude': float(sensor['latitude']),
                'longitude': float(sensor['longitude']),
                'city': sensor.get('city', 'Unknown'),
                'country': sensor.get('country', 'Unknown'),
                'readings': {
                    'pm25': float(reading['pm25']),
                    'pm10': float(reading['pm10']),
                    'no2': float(reading['no2']),
                    'o3': float(reading['o3']),
                    'co': float(reading['co']),
                    'aqi': int(reading['aqi']),
                    'temperature': float(reading['temperature']),
                    'humidity': float(reading['humidity']),
                    'timestamp': reading['timestamp'].isoformat()
                }
            })
    
    return jsonify({'sensors': sensors_data})

@app.route('/api/timeseries/<sensor_id>')
def get_timeseries(sensor_id):
    """Get time series data for a specific sensor"""
    if readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    sensor_data = readings[readings['sensor_id'] == sensor_id].sort_values('timestamp')
    
    if len(sensor_data) == 0:
        return jsonify({'error': 'Sensor not found'}), 404
    
    # Limit to last 500 points for performance
    sensor_data = sensor_data.tail(500)
    
    return jsonify({
        'sensor_id': sensor_id,
        'data': {
            'timestamps': sensor_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'pm25': sensor_data['pm25'].tolist(),
            'pm10': sensor_data['pm10'].tolist(),
            'no2': sensor_data['no2'].tolist(),
            'o3': sensor_data['o3'].tolist(),
            'aqi': sensor_data['aqi'].tolist()
        }
    })

@app.route('/api/hourly_pattern')
def get_hourly_pattern():
    """Get average pollution by hour of day"""
    if readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    readings['hour'] = pd.to_datetime(readings['timestamp']).dt.hour
    hourly_avg = readings.groupby('hour').agg({
        'pm25': 'mean',
        'pm10': 'mean',
        'no2': 'mean',
        'aqi': 'mean'
    }).reset_index()
    
    return jsonify({
        'hours': hourly_avg['hour'].tolist(),
        'pm25': hourly_avg['pm25'].tolist(),
        'pm10': hourly_avg['pm10'].tolist(),
        'no2': hourly_avg['no2'].tolist(),
        'aqi': hourly_avg['aqi'].tolist()
    })

@app.route('/api/model_info')
def get_model_info():
    """Get ML model information"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    feature_importance = model_data['feature_importance']
    
    return jsonify({
        'model_type': 'Random Forest Regressor',
        'target': 'PM2.5',
        'train_samples': 1496,
        'test_samples': 375,
        'training_date': '2024-11-17',
        'performance': {
            'r2_score': 0.457,
            'mae': 11.85,
            'rmse': 15.86
        },
        'feature_importance': feature_importance.to_dict('records')
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction for given location and conditions"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Prepare features
        features_df = pd.DataFrame([{
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'temperature': data.get('temperature', 25.0),
            'humidity': data.get('humidity', 60.0),
            'hour': data.get('hour', 12),
            'day_of_week': data.get('day_of_week', 2),
            'is_weekend': data.get('is_weekend', 0),
            'dist_from_center': data.get('dist_from_center', 0.0)
        }])
        
        # Scale and predict
        model = model_data['model']
        scaler = model_data['scaler']
        
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_pm25': float(prediction),
            'location': {
                'latitude': data['latitude'],
                'longitude': data['longitude']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/heatmap_data')
def get_heatmap_data():
    """Get data for heatmap visualization"""
    if readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    latest = readings.groupby('sensor_id').tail(1)
    
    heatmap_data = []
    for _, row in latest.iterrows():
        heatmap_data.append({
            'lat': float(row['latitude']),
            'lng': float(row['longitude']),
            'value': float(row['pm25']),
            'aqi': int(row['aqi'])
        })
    
    return jsonify({'heatmap': heatmap_data})

@app.route('/api/statistics')
def get_statistics():
    """Get comprehensive statistics"""
    if readings is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    return jsonify({
        'correlations': readings[['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']].corr().to_dict(),
        'distributions': {
            'pm25': {
                'histogram': np.histogram(readings['pm25'].dropna(), bins=20)[0].tolist(),
                'bins': np.histogram(readings['pm25'].dropna(), bins=20)[1].tolist()
            }
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🌍 Air Quality Monitoring System - Web Server")
    print("=" * 60)
    print(f"Frontend: http://localhost:5001")
    print(f"API: http://localhost:5001/api/")
    print("=" * 60)
    app.run(debug=True, port=5001)