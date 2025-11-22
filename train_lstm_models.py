"""
Spatio-Temporal AQI Forecasting using LSTM and ConvLSTM
Advanced deep learning models for air quality prediction and hotspot detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configure environment for Mac M1/M2 TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, ConvLSTM2D, Flatten, TimeDistributed, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure TensorFlow threading for stability
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

print("=" * 80)
print("Spatio-Temporal AQI Forecasting - LSTM & ConvLSTM Models")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

# Load data
print("\nLoading data...")
locations = pd.read_csv('data/sensor_locations.csv')
readings = pd.read_csv('data/air_quality_readings.csv')
readings['timestamp'] = pd.to_datetime(readings['timestamp'])
readings = readings.sort_values(['sensor_id', 'timestamp'])

print(f"Loaded {len(readings)} readings from {len(locations)} sensors")
print(f"Date range: {readings['timestamp'].min()} to {readings['timestamp'].max()}")

def create_sequences(data, seq_length=24, forecast_horizon=6):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length + forecast_horizon - 1])
    return np.array(X), np.array(y)

def prepare_lstm_data(readings, seq_length=24, forecast_horizon=6):
    """Prepare data for LSTM model"""
    print(f"\n📊 Preparing LSTM sequences (lookback={seq_length}h, forecast={forecast_horizon}h)...")
    
    # Features for prediction
    feature_cols = ['pm25', 'pm10', 'no2', 'temperature', 'humidity', 'aqi']
    
    # Single scaler for all data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Collect all data first for proper scaling
    all_data = []
    sensor_lengths = []
    
    for sensor_id in readings['sensor_id'].unique():
        sensor_data = readings[readings['sensor_id'] == sensor_id].sort_values('timestamp')[feature_cols].values
        if len(sensor_data) >= seq_length + forecast_horizon:
            all_data.append(sensor_data)
            sensor_lengths.append(len(sensor_data))
    
    # Fit scaler on all data
    combined_data = np.vstack(all_data)
    scaler.fit(combined_data)
    
    X_all, y_all = [], []
    
    # Now create sequences with properly scaled data
    for sensor_data in all_data:
        sensor_scaled = scaler.transform(sensor_data)
        
        # Create sequences
        X_seq, y_seq = create_sequences(sensor_scaled, seq_length, forecast_horizon)
        X_all.append(X_seq)
        y_all.append(y_seq)
    
    X = np.vstack(X_all)
    y = np.vstack(y_all)
    
    # We predict AQI (last column)
    y = y[:, -1]
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Random shuffle before split for better generalization
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape):
    """Build LSTM model for temporal forecasting"""
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape,
             recurrent_dropout=0.1),
        Dropout(0.3),
        LSTM(32, activation='tanh', return_sequences=False,
             recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for normalized output [0,1]
    ])
    
    # Use Adam with lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def prepare_convlstm_data(readings, grid_size=5, seq_length=24, forecast_horizon=6):
    """Prepare spatial grid data for ConvLSTM"""
    print(f"\n🗺️  Preparing ConvLSTM spatial grids ({grid_size}x{grid_size})...")
    
    # Create spatial grid
    lat_bins = np.linspace(readings['latitude'].min(), readings['latitude'].max(), grid_size + 1)
    lon_bins = np.linspace(readings['longitude'].min(), readings['longitude'].max(), grid_size + 1)
    
    # Assign each reading to a grid cell
    readings = readings.copy()
    readings['grid_lat'] = pd.cut(readings['latitude'], bins=lat_bins, labels=False)
    readings['grid_lon'] = pd.cut(readings['longitude'], bins=lon_bins, labels=False)
    
    # Get unique timestamps
    timestamps = sorted(readings['timestamp'].unique())
    
    # Normalize AQI to [0, 1] for better training
    max_aqi = readings['aqi'].max()
    
    X_all, y_all = [], []
    
    for i in range(len(timestamps) - seq_length - forecast_horizon + 1):
        # Get sequence of spatial grids
        sequence = []
        for t in range(i, i + seq_length):
            timestamp = timestamps[t]
            data_at_t = readings[readings['timestamp'] == timestamp]
            
            # Create grid with interpolation for empty cells
            grid = np.zeros((grid_size, grid_size, 1))
            grid_counts = np.zeros((grid_size, grid_size))
            
            for _, row in data_at_t.iterrows():
                if pd.notna(row['grid_lat']) and pd.notna(row['grid_lon']):
                    lat_idx = int(row['grid_lat'])
                    lon_idx = int(row['grid_lon'])
                    # Normalize AQI
                    grid[lat_idx, lon_idx, 0] += row['aqi'] / max_aqi
                    grid_counts[lat_idx, lon_idx] += 1
            
            # Average cells with multiple readings
            for i_lat in range(grid_size):
                for i_lon in range(grid_size):
                    if grid_counts[i_lat, i_lon] > 1:
                        grid[i_lat, i_lon, 0] /= grid_counts[i_lat, i_lon]
            
            sequence.append(grid)
        
        # Target (future AQI grid)
        target_timestamp = timestamps[i + seq_length + forecast_horizon - 1]
        target_data = readings[readings['timestamp'] == target_timestamp]
        target_grid = np.zeros((grid_size, grid_size))
        target_counts = np.zeros((grid_size, grid_size))
        
        for _, row in target_data.iterrows():
            if pd.notna(row['grid_lat']) and pd.notna(row['grid_lon']):
                lat_idx = int(row['grid_lat'])
                lon_idx = int(row['grid_lon'])
                target_grid[lat_idx, lon_idx] += row['aqi'] / max_aqi
                target_counts[lat_idx, lon_idx] += 1
        
        # Average cells with multiple readings
        for i_lat in range(grid_size):
            for i_lon in range(grid_size):
                if target_counts[i_lat, i_lon] > 1:
                    target_grid[i_lat, i_lon] /= target_counts[i_lat, i_lon]
        
        X_all.append(np.array(sequence))
        y_all.append(target_grid)
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    print(f"Created {len(X)} spatio-temporal sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Random shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, max_aqi

def build_convlstm_model(input_shape):
    """Build ConvLSTM model for spatio-temporal forecasting"""
    model = Sequential([
        ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, 
                   activation='tanh', input_shape=input_shape,
                   recurrent_dropout=0.1),
        Dropout(0.3),
        ConvLSTM2D(16, kernel_size=(3, 3), padding='same', return_sequences=False, 
                   activation='tanh',
                   recurrent_dropout=0.1),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(np.prod(input_shape[1:3]), activation='sigmoid')  # Sigmoid for normalized output
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def detect_hotspots(readings, threshold_percentile=75):
    """Detect pollution hotspots"""
    print("\n🔥 Detecting pollution hotspots...")
    
    # Calculate threshold
    threshold = np.percentile(readings['aqi'], threshold_percentile)
    print(f"Hotspot threshold (75th percentile): AQI = {threshold:.0f}")
    
    # Group by location and calculate average AQI
    hotspot_data = readings.groupby(['sensor_id', 'latitude', 'longitude']).agg({
        'aqi': ['mean', 'max', 'std'],
        'pm25': 'mean'
    }).reset_index()
    
    hotspot_data.columns = ['sensor_id', 'latitude', 'longitude', 'avg_aqi', 'max_aqi', 'std_aqi', 'avg_pm25']
    
    # Mark hotspots
    hotspot_data['is_hotspot'] = hotspot_data['avg_aqi'] >= threshold
    
    hotspots = hotspot_data[hotspot_data['is_hotspot']]
    print(f"Found {len(hotspots)} hotspot locations out of {len(hotspot_data)} total locations")
    
    return hotspot_data, threshold

def visualize_hotspots(hotspot_data, threshold):
    """Visualize hotspot map"""
    plt.figure(figsize=(12, 8))
    
    # Plot all locations
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
    
    # Highlight hotspots
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
    plt.title('Air Quality Hotspot Map', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/10_hotspot_map.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/10_hotspot_map.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Use full dataset for actual training
    print(f"\nUsing full dataset: {len(readings)} readings")
    
    print("\n" + "=" * 80)
    print("PART 1: LSTM MODEL (Temporal Forecasting)")
    print("=" * 80)
    
    # Prepare LSTM data
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(
        readings, seq_length=24, forecast_horizon=6
    )
    
    # Build and train LSTM
    print("\n🧠 Building LSTM model...")
    lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    print(lstm_model.summary())
    
    print("\n🔄 Training LSTM model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    
    history_lstm = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate LSTM
    print("\n📊 Evaluating LSTM model...")
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
    
    lstm_results = {
        'mae': mean_absolute_error(y_test_lstm, y_pred_lstm),
        'rmse': np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm)),
        'r2': r2_score(y_test_lstm, y_pred_lstm)
    }
    
    print(f"\nLSTM Test MAE: {lstm_results['mae']:.2f}")
    print(f"LSTM Test RMSE: {lstm_results['rmse']:.2f}")
    print(f"LSTM Test R²: {lstm_results['r2']:.3f}")
    
    # Save LSTM model
    lstm_model.save('models/lstm_aqi_forecasting.h5')
    print("\n✅ LSTM model saved to: models/lstm_aqi_forecasting.h5")
    
    # Plot LSTM results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_lstm.history['loss'], label='Train Loss')
    plt.plot(history_lstm.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('LSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_lstm, y_pred_lstm, alpha=0.5)
    plt.plot([y_test_lstm.min(), y_test_lstm.max()], 
             [y_test_lstm.min(), y_test_lstm.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'LSTM Predictions (R²={lstm_results["r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/11_lstm_results.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/11_lstm_results.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("PART 2: ConvLSTM MODEL (Spatio-Temporal Forecasting)")
    print("=" * 80)
    
    # Prepare ConvLSTM data
    X_train_conv, X_test_conv, y_train_conv, y_test_conv, max_aqi = prepare_convlstm_data(
        readings, grid_size=5, seq_length=24, forecast_horizon=6
    )
    
    # Build and train ConvLSTM
    print("\n🧠 Building ConvLSTM model...")
    convlstm_model = build_convlstm_model(X_train_conv.shape[1:])
    print(convlstm_model.summary())
    
    print("\n🔄 Training ConvLSTM model...")
    history_conv = convlstm_model.fit(
        X_train_conv, y_train_conv.reshape(y_train_conv.shape[0], -1),
        validation_split=0.2,
        epochs=80,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate ConvLSTM
    print("\n📊 Evaluating ConvLSTM model...")
    y_pred_conv = convlstm_model.predict(X_test_conv)
    y_pred_conv = y_pred_conv.reshape(y_test_conv.shape)
    
    # Denormalize for evaluation
    y_test_denorm = y_test_conv.flatten() * max_aqi
    y_pred_denorm = y_pred_conv.flatten() * max_aqi
    
    convlstm_results = {
        'mae': mean_absolute_error(y_test_denorm, y_pred_denorm),
        'rmse': np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
        'r2': r2_score(y_test_denorm, y_pred_denorm)
    }
    
    print(f"\nConvLSTM Test MAE: {convlstm_results['mae']:.2f}")
    print(f"ConvLSTM Test RMSE: {convlstm_results['rmse']:.2f}")
    print(f"ConvLSTM Test R²: {convlstm_results['r2']:.3f}")
    
    # Save ConvLSTM model
    convlstm_model.save('models/convlstm_spatiotemporal.h5')
    print("\n✅ ConvLSTM model saved to: models/convlstm_spatiotemporal.h5")
    
    # Plot ConvLSTM results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_conv.history['loss'], label='Train Loss')
    plt.plot(history_conv.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('ConvLSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    example_idx = 0
    plt.imshow(y_test_conv[example_idx] * max_aqi, cmap='YlOrRd', vmin=0, vmax=max_aqi)
    plt.colorbar(label='AQI')
    plt.title('Actual AQI Grid')
    
    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_conv[example_idx] * max_aqi, cmap='YlOrRd', vmin=0, vmax=max_aqi)
    plt.colorbar(label='AQI')
    plt.title('Predicted AQI Grid')
    
    plt.tight_layout()
    plt.savefig('visualizations/12_convlstm_results.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/12_convlstm_results.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("PART 3: HOTSPOT DETECTION")
    print("=" * 80)
    
    # Detect and visualize hotspots
    hotspot_data, threshold = detect_hotspots(readings, threshold_percentile=75)
    visualize_hotspots(hotspot_data, threshold)
    
    # Save hotspot data
    hotspot_data.to_csv('data/hotspot_analysis.csv', index=False)
    print("Saved hotspot analysis to: data/hotspot_analysis.csv")
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📊 Model Comparison:")
    print(f"  Random Forest R²: 0.841 (spatial interpolation)")
    print(f"  LSTM R²: {lstm_results['r2']:.3f} (temporal forecasting)")
    print(f"  ConvLSTM R²: {convlstm_results['r2']:.3f} (spatio-temporal forecasting)")
    print("\n✅ All models and visualizations saved!")
