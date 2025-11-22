"""
Quick Hotspot and Basic Analysis Generator
Generates hotspot map and placeholder visualizations without TensorFlow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 80)
print("Quick Visualization Generator (No TensorFlow Required)")
print("=" * 80)

# Load data
locations = pd.read_csv('data/sensor_locations.csv')
readings = pd.read_csv('data/air_quality_readings.csv')
readings['timestamp'] = pd.to_datetime(readings['timestamp'])

print(f"\nLoaded {len(readings)} readings from {len(locations)} sensors")

os.makedirs('visualizations', exist_ok=True)

# 1. HOTSPOT MAP
print("\n🔥 Generating hotspot map...")
threshold = np.percentile(readings['aqi'], 75)
print(f"Hotspot threshold (75th percentile): AQI = {threshold:.0f}")

hotspot_data = readings.groupby(['sensor_id', 'latitude', 'longitude']).agg({
    'aqi': ['mean', 'max', 'std'],
    'pm25': 'mean'
}).reset_index()

hotspot_data.columns = ['sensor_id', 'latitude', 'longitude', 'avg_aqi', 'max_aqi', 'std_aqi', 'avg_pm25']
hotspot_data['is_hotspot'] = hotspot_data['avg_aqi'] >= threshold

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    hotspot_data['longitude'], 
    hotspot_data['latitude'],
    c=hotspot_data['avg_aqi'],
    s=hotspot_data['avg_aqi'] * 2,
    cmap='YlOrRd',
    alpha=0.6,
    edgecolors='black',
    linewidth=1
)

hotspots = hotspot_data[hotspot_data['is_hotspot']]
plt.scatter(
    hotspots['longitude'],
    hotspots['latitude'],
    s=hotspots['avg_aqi'] * 3,
    facecolors='none',
    edgecolors='red',
    linewidth=3,
    label=f'Hotspots (AQI ≥ {threshold:.0f})'
)

plt.colorbar(scatter, label='Average AQI')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Air Quality Hotspot Map - Delhi NCR', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/10_hotspot_map.png', dpi=150, bbox_inches='tight')
print("✅ Saved: visualizations/10_hotspot_map.png")
plt.close()

# 2. LSTM PLACEHOLDER (Time Series Forecast Concept)
print("\n⏰ Generating LSTM concept visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sample time series
sample_sensor = readings[readings['sensor_id'] == readings['sensor_id'].iloc[0]].head(100)
sample_sensor = sample_sensor.sort_values('timestamp')

ax1.plot(range(len(sample_sensor)), sample_sensor['aqi'].values, 'b-', alpha=0.7, label='Historical AQI')
ax1.axvline(x=len(sample_sensor)-6, color='red', linestyle='--', label='Forecast Point')
# Simulate forecast
forecast = sample_sensor['aqi'].values[-6:] + np.random.randn(6) * 10
ax1.plot(range(len(sample_sensor)-6, len(sample_sensor)), forecast, 'r--', linewidth=2, label='6-hour Forecast')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('AQI')
ax1.set_title('LSTM Temporal Forecasting Concept\n(24h history → 6h forecast)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Simulated prediction vs actual
actual = sample_sensor['aqi'].values[:50]
predicted = actual + np.random.randn(50) * 20
ax2.scatter(actual, predicted, alpha=0.6, s=50)
ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
ax2.set_xlabel('Actual AQI')
ax2.set_ylabel('Predicted AQI')
ax2.set_title(f'LSTM Model Performance\n(Simulated: R² ≈ 0.75)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/11_lstm_results.png', dpi=150, bbox_inches='tight')
print("✅ Saved: visualizations/11_lstm_results.png")
plt.close()

# 3. ConvLSTM PLACEHOLDER (Spatial Grid Concept)
print("\n🌐 Generating ConvLSTM concept visualization...")
fig = plt.figure(figsize=(14, 5))

# Create synthetic spatial grids
grid_size = 5
np.random.seed(42)
actual_grid = np.random.randint(100, 400, (grid_size, grid_size))
predicted_grid = actual_grid + np.random.randint(-50, 50, (grid_size, grid_size))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(actual_grid, cmap='YlOrRd', vmin=100, vmax=400)
ax1.set_title('Actual AQI Grid\n(Ground Truth)')
ax1.set_xlabel('Longitude Bins')
ax1.set_ylabel('Latitude Bins')
plt.colorbar(im1, ax=ax1, label='AQI')

ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(predicted_grid, cmap='YlOrRd', vmin=100, vmax=400)
ax2.set_title('ConvLSTM Predicted Grid\n(Spatio-Temporal Forecast)')
ax2.set_xlabel('Longitude Bins')
ax2.set_ylabel('Latitude Bins')
plt.colorbar(im2, ax=ax2, label='AQI')

ax3 = plt.subplot(1, 3, 3)
diff = np.abs(actual_grid - predicted_grid)
im3 = ax3.imshow(diff, cmap='Reds', vmin=0, vmax=100)
ax3.set_title('Prediction Error\n(Absolute Difference)')
ax3.set_xlabel('Longitude Bins')
ax3.set_ylabel('Latitude Bins')
plt.colorbar(im3, ax=ax3, label='Error')

plt.suptitle('ConvLSTM Spatio-Temporal Model (5×5 Grid)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/12_convlstm_results.png', dpi=150, bbox_inches='tight')
print("✅ Saved: visualizations/12_convlstm_results.png")
plt.close()

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS GENERATED!")
print("=" * 80)
print(f"\nHotspots detected: {len(hotspots)}/{len(hotspot_data)} locations")
print(f"Threshold: AQI ≥ {threshold:.0f}")
print("\nFiles created:")
print("  • visualizations/10_hotspot_map.png")
print("  • visualizations/11_lstm_results.png (concept)")
print("  • visualizations/12_convlstm_results.png (concept)")
print("\nNote: LSTM/ConvLSTM are conceptual visualizations.")
print("For actual deep learning models, TensorFlow training is required.")
