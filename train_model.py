"""
Air Quality Spatial Interpolation Model
This script implements ML-based spatial interpolation for air quality prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.interpolate import griddata
import pickle
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class AirQualitySpatialModel:
    """ML model for spatial interpolation of air quality measurements"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        features = df[['latitude', 'longitude', 'temperature', 'humidity']].copy()
        
        # Add time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Add spatial clustering features (distance from city center)
        center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        features['dist_from_center'] = np.sqrt(
            (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
        )
        
        return features
    
    def train(self, X, y, model_type='random_forest'):
        """Train the spatial interpolation model"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        results = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return results, X_test, y_test, test_pred
    
    def predict(self, X):
        """Predict air quality for new locations"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_importance = data['feature_importance']

def load_data():
    """Load air quality data"""
    print("Loading data...")
    readings = pd.read_csv('data/air_quality_readings.csv')
    locations = pd.read_csv('data/sensor_locations.csv')
    
    print(f"Loaded {len(readings)} readings from {len(locations)} sensors")
    return readings, locations

def exploratory_data_analysis(df):
    """Perform EDA and generate visualizations"""
    print("\n=== Exploratory Data Analysis ===\n")
    
    # Basic statistics
    print("Dataset shape:", df.shape)
    print("\nPollutant statistics:")
    print(df[['pm25', 'pm10', 'no2', 'o3', 'co', 'aqi']].describe())
    
    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    pollutants = ['pm25', 'pm10', 'no2', 'o3', 'co', 'aqi']
    
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx // 3, idx % 3]
        ax.hist(df[pollutant], bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{pollutant.upper()} Distribution')
        ax.set_xlabel(pollutant.upper())
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('visualizations/01_pollutant_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/01_pollutant_distributions.png")
    plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['pm25', 'pm10', 'no2', 'o3', 'co', 'temperature', 'humidity']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Pollutant Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/02_correlation_matrix.png")
    plt.close()
    
    # 3. Temporal patterns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    hourly_avg = df.groupby('hour')['pm25'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average PM2.5 (µg/m³)')
    plt.title('Average PM2.5 Levels by Hour of Day')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/03_temporal_pattern.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/03_temporal_pattern.png")
    plt.close()
    
    # 4. Spatial distribution (scatter)
    latest = df.groupby('sensor_id').tail(1)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latest['longitude'], latest['latitude'], 
                         c=latest['pm25'], s=100, cmap='YlOrRd', 
                         edgecolors='black', linewidth=0.5, alpha=0.7)
    plt.colorbar(scatter, label='PM2.5 (µg/m³)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution of PM2.5 (Latest Readings)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/04_spatial_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/04_spatial_distribution.png")
    plt.close()

def train_models(df):
    """Train multiple models and compare performance"""
    print("\n=== Model Training ===\n")
    
    # Prepare data
    model = AirQualitySpatialModel()
    X = model.prepare_features(df)
    y = df['pm25'].values
    
    # Train Random Forest
    print("\n--- Random Forest ---")
    results_rf, X_test, y_test, pred_rf = model.train(X, y, 'random_forest')
    
    print(f"Train MAE: {results_rf['train_mae']:.2f}")
    print(f"Train RMSE: {results_rf['train_rmse']:.2f}")
    print(f"Train R²: {results_rf['train_r2']:.3f}")
    print(f"Test MAE: {results_rf['test_mae']:.2f}")
    print(f"Test RMSE: {results_rf['test_rmse']:.2f}")
    print(f"Test R²: {results_rf['test_r2']:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/air_quality_model.pkl')
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(model.feature_importance['feature'], model.feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance for PM2.5 Prediction')
    plt.tight_layout()
    plt.savefig('visualizations/05_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/05_feature_importance.png")
    plt.close()
    
    # Prediction vs Actual plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, pred_rf, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual PM2.5 (µg/m³)')
    plt.ylabel('Predicted PM2.5 (µg/m³)')
    plt.title(f'Prediction vs Actual (R² = {results_rf["test_r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/06_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/06_prediction_vs_actual.png")
    plt.close()
    
    # Residual plot
    residuals = y_test - pred_rf
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_rf, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted PM2.5 (µg/m³)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/07_residuals.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/07_residuals.png")
    plt.close()
    
    return model, results_rf

def generate_interpolation_map(model, df):
    """Generate spatial interpolation heatmap"""
    print("\n=== Generating Interpolation Map ===\n")
    
    # Get latest readings
    latest = df.groupby('sensor_id').tail(1)
    
    # Create grid for interpolation
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    grid_lat = np.linspace(lat_min, lat_max, 100)
    grid_lon = np.linspace(lon_min, lon_max, 100)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Prepare grid features for prediction
    grid_features = pd.DataFrame({
        'latitude': grid_lat_mesh.ravel(),
        'longitude': grid_lon_mesh.ravel(),
        'temperature': latest['temperature'].mean(),
        'humidity': latest['humidity'].mean(),
        'hour': 12,  # noon
        'day_of_week': 2,  # Wednesday
        'is_weekend': 0,
        'dist_from_center': np.sqrt(
            (grid_lat_mesh.ravel() - latest['latitude'].mean())**2 + 
            (grid_lon_mesh.ravel() - latest['longitude'].mean())**2
        )
    })
    
    # Predict on grid
    grid_predictions = model.predict(grid_features)
    grid_predictions_mesh = grid_predictions.reshape(grid_lat_mesh.shape)
    
    # Create interpolation map
    plt.figure(figsize=(12, 10))
    
    # Contour plot
    contour = plt.contourf(grid_lon_mesh, grid_lat_mesh, grid_predictions_mesh, 
                           levels=20, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(contour, label='Predicted PM2.5 (µg/m³)')
    
    # Overlay actual sensor locations
    plt.scatter(latest['longitude'], latest['latitude'], 
               c='black', s=50, marker='x', linewidths=2, 
               label='Sensor Locations', zorder=5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Interpolation of PM2.5 Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/08_interpolation_map.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/08_interpolation_map.png")
    plt.close()
    
    # Also create a simple IDW interpolation for comparison (if enough points)
    if len(latest) > 3:
        points = latest[['longitude', 'latitude']].values
        values = latest['pm25'].values
        
        try:
            grid_idw = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')
            
            plt.figure(figsize=(12, 10))
            contour = plt.contourf(grid_lon_mesh, grid_lat_mesh, grid_idw, 
                                   levels=20, cmap='YlOrRd', alpha=0.8)
            plt.colorbar(contour, label='PM2.5 (µg/m³)')
            plt.scatter(latest['longitude'], latest['latitude'], 
                       c='black', s=50, marker='x', linewidths=2, 
                       label='Sensor Locations', zorder=5)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('IDW Interpolation of PM2.5 Levels (Baseline)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/09_idw_interpolation.png', dpi=300, bbox_inches='tight')
            print("Saved: visualizations/09_idw_interpolation.png")
            plt.close()
        except:
            print("Skipped IDW interpolation (insufficient spatial points)")
    else:
        print("Skipped IDW interpolation (need at least 4 sensor locations)")

def main():
    """Main execution function"""
    print("=" * 60)
    print("Air Quality Spatial Interpolation - ML Model")
    print("=" * 60)
    
    # Load data
    readings, locations = load_data()
    
    # EDA
    exploratory_data_analysis(readings)
    
    # Train models
    model, results = train_models(readings)
    
    # Generate interpolation maps
    generate_interpolation_map(model, readings)
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print(f"\nFinal Model Performance:")
    print(f"  Test MAE: {results['test_mae']:.2f} µg/m³")
    print(f"  Test RMSE: {results['test_rmse']:.2f} µg/m³")
    print(f"  Test R²: {results['test_r2']:.3f}")
    print(f"\nModel saved to: models/air_quality_model.pkl")
    print(f"Visualizations saved to: visualizations/")

if __name__ == "__main__":
    main()
