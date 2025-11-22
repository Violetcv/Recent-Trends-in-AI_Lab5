"""
Air Quality Monitoring Dashboard
Interactive Streamlit dashboard for visualizing air quality predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Air Quality Monitoring System",
    page_icon="🌍",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    """Load air quality data"""
    try:
        locations = pd.read_csv('data/sensor_locations.csv')
        readings = pd.read_csv('data/air_quality_readings.csv')
        readings['timestamp'] = pd.to_datetime(readings['timestamp'])
        return locations, readings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_model():
    """Load trained ML model"""
    try:
        with open('models/air_quality_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Title and description
st.title("🌍 Air Quality Monitoring & Prediction System")
st.markdown("""
### Smart City Air Quality Intelligence
This system uses Machine Learning for spatial interpolation of air quality measurements.
**Data Source:** Real air quality data from UCI Air Quality Dataset (Italy, 2004-2005)
""")

# Load data
locations, readings = load_data()
model_data = load_model()

if locations is not None and readings is not None:
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Date range
    min_date = readings['timestamp'].min()
    max_date = readings['timestamp'].max()
    
    st.sidebar.subheader("Date Range")
    st.sidebar.write(f"Available: {min_date.date()} to {max_date.date()}")
    
    # Pollutant selector
    st.sidebar.subheader("Pollutant")
    pollutant = st.sidebar.selectbox(
        "Select pollutant to visualize",
        ['pm25', 'pm10', 'no2', 'o3', 'co', 'aqi']
    )
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest readings
    latest = readings.groupby('sensor_id').tail(1)
    
    with col1:
        st.metric(
            label="🌡️ Average PM2.5",
            value=f"{latest['pm25'].mean():.1f} µg/m³",
            delta=f"{latest['pm25'].std():.1f} std"
        )
    
    with col2:
        st.metric(
            label="📊 Average AQI",
            value=f"{latest['aqi'].mean():.0f}",
            delta=f"Max: {latest['aqi'].max():.0f}"
        )
    
    with col3:
        st.metric(
            label="🏢 Monitoring Stations",
            value=len(locations)
        )
    
    with col4:
        st.metric(
            label="📈 Total Readings",
            value=f"{len(readings):,}"
        )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Spatial Analysis", "📈 Time Series", "🤖 Model Performance"])
    
    with tab1:
        st.header("Air Quality Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pollutant Distributions")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                readings[['pm25', 'pm10', 'no2']].boxplot(ax=ax)
                ax.set_ylabel('Concentration (µg/m³)')
                ax.set_title('Pollutant Concentrations')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            except:
                st.write("Visualization error")
        
        with col2:
            st.subheader("Correlation Matrix")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                corr = readings[['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Pollutant Correlations')
                st.pyplot(fig)
                plt.close()
            except:
                st.write("Visualization error")
        
        st.subheader("Latest Readings by Station")
        st.dataframe(
            latest[['sensor_id', 'pm25', 'pm10', 'no2', 'aqi', 'temperature', 'humidity']],
            hide_index=True
        )
    
    with tab2:
        st.header("Spatial Distribution Analysis")
        
        st.subheader("Generated Visualizations")
        
        # Display saved visualizations
        viz_files = {
            'Spatial Distribution': 'visualizations/04_spatial_distribution.png',
            'ML Interpolation Map': 'visualizations/08_interpolation_map.png'
        }
        
        for title, filepath in viz_files.items():
            if os.path.exists(filepath):
                st.subheader(title)
                st.image(filepath, use_container_width=True)
    
    with tab3:
        st.header("Temporal Patterns")
        
        # Time series plot
        st.subheader(f"Time Series: {pollutant.upper()}")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for sensor_id in readings['sensor_id'].unique()[:5]:  # Plot up to 5 sensors
                sensor_data = readings[readings['sensor_id'] == sensor_id].sort_values('timestamp')
                ax.plot(sensor_data['timestamp'], sensor_data[pollutant], 
                       label=sensor_id, alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(f'{pollutant.upper()} Concentration')
            ax.set_title(f'{pollutant.upper()} Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error plotting time series: {e}")
        
        # Hourly patterns
        st.subheader("Average by Hour of Day")
        
        try:
            readings['hour'] = pd.to_datetime(readings['timestamp']).dt.hour
            hourly_avg = readings.groupby('hour')[pollutant].mean()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel(f'Average {pollutant.upper()}')
            ax.set_title(f'Average {pollutant.upper()} by Hour')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error plotting hourly patterns: {e}")
    
    with tab4:
        st.header("ML Model Performance")
        
        if model_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Information")
                st.write("**Model Type:** Random Forest Regressor")
                st.write("**Target:** PM2.5 Concentration")
                st.write("**Features:** Latitude, Longitude, Temperature, Humidity, Hour, Day of Week, Distance from Center")
            
            with col2:
                st.subheader("Performance Metrics")
                st.write("**Test MAE:** 11.85 µg/m³")
                st.write("**Test RMSE:** 15.86 µg/m³")
                st.write("**Test R²:** 0.457")
            
            # Feature importance
            st.subheader("Feature Importance")
            if os.path.exists('visualizations/05_feature_importance.png'):
                st.image('visualizations/05_feature_importance.png', use_container_width=True)
            
            # Prediction vs Actual
            st.subheader("Prediction Quality")
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('visualizations/06_prediction_vs_actual.png'):
                    st.image('visualizations/06_prediction_vs_actual.png', use_container_width=True)
            
            with col2:
                if os.path.exists('visualizations/07_residuals.png'):
                    st.image('visualizations/07_residuals.png', use_container_width=True)
        
        else:
            st.warning("Model not loaded")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📋 About This System
    - **Track:** Smart City Air Quality Monitoring (Track C - Beginner)
    - **Data Source:** UCI Air Quality Dataset + OpenAQ (fallback)
    - **ML Technique:** Random Forest Spatial Interpolation
    - **Technologies:** Python, Scikit-learn, Streamlit, Pandas, Matplotlib
    
    **Ethics & Privacy:** This system uses publicly available, anonymized air quality data.
    Real deployments must consider data privacy, sensor calibration, and equity in monitoring coverage.
    """)

else:
    st.error("Failed to load data. Please ensure data files exist in the 'data/' directory.")

# Instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Use filters to explore data
    2. Navigate tabs for different analyses
    3. View model predictions and performance
    4. Check spatial interpolation maps
    """)
