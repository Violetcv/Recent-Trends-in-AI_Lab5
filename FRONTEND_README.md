# Air Quality Monitoring System - Web Frontend

## Overview
Modern web-based frontend for the Air Quality Monitoring System with interactive visualizations, real-time data display, and ML model predictions.

## Features

### 🏠 Overview Tab
- **Real-time Statistics**: Total sensors, readings count, average PM2.5, current AQI
- **Sensors Table**: Live data from all monitoring stations with color-coded AQI status
- **Auto-refresh**: Data updates automatically

### 🗺️ Map Tab
- **Interactive Map**: Leaflet.js powered map with sensor locations
- **Pollutant Selector**: Switch between PM2.5, PM10, and NO2 visualizations
- **Color-coded Markers**: Visual representation of pollution levels (green=good, yellow=moderate, red=unhealthy)
- **Popup Details**: Click markers for detailed sensor information

### 📊 Analytics Tab
- **Time Series Chart**: Historical trends for all pollutants
- **Hourly Patterns**: Average pollution levels by hour of day
- **Statistics Dashboard**: Comprehensive statistical metrics
- **Correlation Matrix**: Pollutant relationships visualization

### 🤖 ML Model Tab
- **Model Information**: Random Forest details, training date, sample counts
- **Performance Metrics**: R² Score, MAE, RMSE
- **Feature Importance**: Visual representation of model features
- **Prediction Tool**: Interactive form to predict PM2.5 levels

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

## Technology Stack

### Backend
- **Flask 3.0.0**: Python web framework
- **Flask-CORS**: Cross-origin resource sharing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning model
- **pickle**: Model serialization

### Frontend
- **HTML5**: Structure
- **CSS3**: Modern styling with animations
- **JavaScript (ES6+)**: Interactive functionality
- **Leaflet.js 1.9.4**: Interactive mapping
- **Chart.js 4.4.0**: Data visualizations
- **Font Awesome 6.4.0**: Icons

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Health check |
| `/api/overview` | GET | Summary statistics |
| `/api/sensors` | GET | All sensor data |
| `/api/timeseries/<sensor_id>` | GET | Historical data |
| `/api/hourly_pattern` | GET | Hourly averages |
| `/api/model_info` | GET | ML model details |
| `/api/predict` | POST | Make predictions |
| `/api/heatmap_data` | GET | Spatial heatmap |
| `/api/statistics` | GET | Statistical metrics |

## File Structure

```
frontend/
├── index.html          # Main HTML structure
├── css/
│   └── styles.css      # CSS styling
└── js/
    └── app.js          # JavaScript functionality

app.py                  # Flask backend server
```

## Features in Detail

### Overview Dashboard
- **Total Sensors**: Number of active monitoring stations
- **Total Readings**: Cumulative measurements count
- **Average PM2.5**: Current average particulate matter
- **Current AQI**: Real-time Air Quality Index

### Interactive Map
- Pan and zoom capabilities
- Click markers for detailed information
- Switch between different pollutants
- Color-coded legend for easy interpretation

### Analytics Charts
- **Line Chart**: Time series with multiple pollutants
- **Bar Chart**: Hourly pattern analysis
- **Statistics**: Min, max, mean, median, std dev
- **Responsive Design**: Adapts to screen size

### ML Model Interface
- **Model Specs**: Type, samples, training date
- **Performance**: Visual metrics display
- **Feature Analysis**: Importance ranking
- **Prediction Form**: Input features to get PM2.5 prediction

## Browser Compatibility
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### Making Changes
1. Edit HTML/CSS/JS files in `frontend/` directory
2. Refresh browser (no server restart needed)
3. For backend changes, restart Flask server

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

### CORS Issues
- Ensure `flask-cors` is installed
- Check API_BASE_URL in `app.js` matches server address

### Map Not Loading
- Check internet connection (Leaflet uses CDN)
- Verify coordinates in sensor data

### Charts Not Displaying
- Open browser console (F12) for errors
- Verify Chart.js CDN is accessible

## Production Deployment

For production use:
1. Use production WSGI server (gunicorn, uwsgi)
2. Set `debug=False` in app.py
3. Configure proper CORS origins
4. Use environment variables for configuration
5. Enable HTTPS

```bash
# Example with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Performance

- **Load Time**: < 2 seconds
- **API Response**: < 100ms average
- **Chart Rendering**: Real-time updates
- **Map Performance**: Smooth panning/zooming

## Accessibility

- Semantic HTML structure
- ARIA labels for screen readers
- Keyboard navigation support
- High contrast color scheme
- Responsive text sizing

## Credits

Built for RTAI Lab 5: Smart City with AI
Track C: Air Quality Monitoring System

**Data Source**: UCI Air Quality Dataset (Italy, 2004-2005)
**ML Model**: Random Forest Spatial Interpolation
**Development**: Python, Flask, JavaScript

---

For questions or issues, refer to the main README.md in the project root.
