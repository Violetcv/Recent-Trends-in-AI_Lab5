# 📸 Frontend Screenshots & Usage Guide

## 🎯 How to Capture Screenshots for Lab Submission

### Overview Tab Screenshot
**What to Capture:**
- 4 statistics cards at the top
- Complete sensors table
- AQI status badges (colored)

**Steps:**
1. Open http://localhost:5000
2. Ensure you're on "Overview" tab (default)
3. Scroll to show all stats and table
4. Take full-page screenshot
5. Save as: `screenshot_1_overview.png`

---

### Map Tab Screenshot
**What to Capture:**
- Full interactive map
- Pollutant selector dropdown
- Color-coded markers
- Legend showing color meanings

**Steps:**
1. Click "Map" tab in navigation
2. Wait for map to load (1-2 seconds)
3. Select "PM2.5" from dropdown
4. Zoom to show markers clearly
5. Click a marker to show popup (optional)
6. Take screenshot
7. Save as: `screenshot_2_map.png`

---

### Analytics Tab Screenshot
**What to Capture:**
- Time series chart (line chart)
- Hourly pattern chart (bar chart)
- Statistics boxes
- Both charts visible

**Steps:**
1. Click "Analytics" tab
2. Wait for charts to render (2-3 seconds)
3. Scroll to show both charts
4. Take full-page screenshot
5. Save as: `screenshot_3_analytics.png`

---

### ML Model Tab Screenshot
**What to Capture:**
- Model information section
- Performance metrics (R², MAE, RMSE)
- Feature importance chart
- Prediction form

**Steps:**
1. Click "ML Model" tab
2. Wait for feature importance chart to load
3. Scroll to show all sections
4. Take screenshot before filling form
5. Save as: `screenshot_4_model_info.png`

---

### Prediction Demo Screenshot
**What to Capture:**
- Filled prediction form
- Prediction result displayed
- Result showing PM2.5 value and category

**Steps:**
1. On "ML Model" tab
2. Fill in the form with sample values:
   - PM2.5: `35.5`
   - PM10: `50.0`
   - NO2: `40.0`
   - CO: `1.5`
   - SO2: `10.0`
   - O3: `50.0`
   - Latitude: `45.0703`
   - Longitude: `7.6869`
3. Click "Predict PM2.5" button
4. Wait for result to appear (green box)
5. Take screenshot showing form + result
6. Save as: `screenshot_5_prediction.png`

---

## 📊 What Each Screenshot Demonstrates

### Screenshot 1: Overview
**Shows:**
- Real-time data monitoring capabilities
- Professional dashboard design
- Color-coded AQI system
- Tabular data presentation
- Responsive card layout

**For Report:** *"Figure 1: Overview dashboard showing real-time air quality statistics and sensor status table with AQI color coding."*

---

### Screenshot 2: Map
**Shows:**
- Geospatial visualization
- Interactive mapping (Leaflet.js)
- Pollutant-specific views
- Location-based monitoring
- Marker popup functionality

**For Report:** *"Figure 2: Interactive map visualization with color-coded markers indicating pollution levels at sensor locations."*

---

### Screenshot 3: Analytics
**Shows:**
- Time series analysis
- Hourly pattern recognition
- Multiple pollutant tracking
- Statistical metrics
- Data visualization expertise

**For Report:** *"Figure 3: Analytics dashboard featuring time series trends and hourly pollution patterns across multiple pollutants."*

---

### Screenshot 4: Model Info
**Shows:**
- ML model implementation
- Random Forest architecture
- Performance metrics visualization
- Feature importance analysis
- Model transparency

**For Report:** *"Figure 4: Machine learning model interface displaying performance metrics and feature importance rankings."*

---

### Screenshot 5: Prediction
**Shows:**
- Interactive prediction capability
- User input interface
- Real-time ML inference
- Result presentation
- Practical application

**For Report:** *"Figure 5: Prediction interface demonstrating real-time PM2.5 forecasting using the trained Random Forest model."*

---

## 🖼️ Optional: API Testing Screenshots

### Test API Status
**Command:**
```bash
curl http://localhost:5000/api/status
```

**Screenshot:** Terminal showing JSON response
**Save as:** `screenshot_api_status.png`

---

### Test API Overview
**Command:**
```bash
curl http://localhost:5000/api/overview
```

**Screenshot:** Terminal showing statistics JSON
**Save as:** `screenshot_api_overview.png`

---

## 📝 How to Add Screenshots to Report

### In REPORT.md

Add section:
```markdown
## 5. Web Frontend Implementation

### 5.1 Overview Dashboard
![Overview Dashboard](screenshot_1_overview.png)
*Figure 5.1: Real-time monitoring dashboard with statistics cards and sensor table*

### 5.2 Interactive Map
![Map Visualization](screenshot_2_map.png)
*Figure 5.2: Geospatial visualization with Leaflet.js showing sensor locations*

### 5.3 Analytics Dashboard
![Analytics](screenshot_3_analytics.png)
*Figure 5.3: Time series and hourly pattern analysis*

### 5.4 ML Model Interface
![Model Info](screenshot_4_model_info.png)
*Figure 5.4: Machine learning model details and performance metrics*

### 5.5 Prediction Demonstration
![Prediction](screenshot_5_prediction.png)
*Figure 5.5: Interactive prediction interface with sample results*
```

---

## 🎨 Screenshot Best Practices

### Image Quality
- **Resolution**: Use full HD (1920x1080) or higher
- **Format**: PNG (lossless, best quality)
- **Compression**: None or minimal
- **DPI**: 150+ for print, 72 for web

### Framing
- ✅ Include browser address bar (shows localhost:5000)
- ✅ Show full navigation tabs
- ✅ Capture complete data (don't cut off tables)
- ✅ Ensure text is readable
- ❌ Don't include personal desktop elements
- ❌ Avoid window shadows or reflections

### Annotation (Optional)
- Add arrows to highlight key features
- Label important sections
- Add brief captions
- Use red boxes for emphasis
- Keep annotations minimal and professional

---

## 🛠️ Screenshot Tools

### macOS
- **Command + Shift + 3**: Full screen
- **Command + Shift + 4**: Selection
- **Command + Shift + 5**: Screen recording (for demo video)

### Windows
- **Win + Shift + S**: Snipping tool
- **Win + Print Screen**: Full screen
- **Alt + Print Screen**: Active window

### Browser Extensions
- **Awesome Screenshot**: Full page capture
- **Fireshot**: Annotate and edit
- **Nimbus**: Screenshot + screen record

---

## 📹 Optional: Create Demo Video

### Recording Steps
1. **Start Screen Recording**
2. **Open browser** to localhost:5000
3. **Navigate Overview tab** (5 seconds)
4. **Switch to Map tab** (5 seconds)
   - Select different pollutants
   - Click a marker
5. **Switch to Analytics tab** (5 seconds)
   - Scroll to show charts
6. **Switch to ML Model tab** (10 seconds)
   - Fill prediction form
   - Click predict button
   - Show result
7. **Stop recording**

### Video Specs
- **Duration**: 30-60 seconds
- **Format**: MP4
- **Resolution**: 1080p
- **Framerate**: 30 fps
- **Size**: < 50 MB

### Video Purpose
- Demonstrates full functionality
- Shows smooth navigation
- Highlights interactivity
- Proves working implementation

---

## 📦 Screenshot Submission Package

### Folder Structure
```
screenshots/
├── screenshot_1_overview.png
├── screenshot_2_map.png
├── screenshot_3_analytics.png
├── screenshot_4_model_info.png
├── screenshot_5_prediction.png
├── screenshot_api_status.png (optional)
├── screenshot_api_overview.png (optional)
└── demo_video.mp4 (optional)
```

### Zip for Submission
```bash
cd "/Users/chhaviverma/Desktop/Semester_7/Recent Trends in AI/RTAI Lab 5"
mkdir screenshots
# (Take screenshots and save them to screenshots/)
zip -r RTAI_Lab5_Screenshots.zip screenshots/
```

---

## ✍️ Caption Templates

Use these in your report:

**Overview:**
> "The overview dashboard provides real-time monitoring of air quality metrics across all sensor locations. Statistics cards display total sensors (1), cumulative readings (1,871), average PM2.5 concentration, and current AQI. The sensors table presents detailed readings for each pollutant with color-coded AQI status indicators (Good: green, Moderate: yellow, Unhealthy: red)."

**Map:**
> "Interactive geospatial visualization implemented using Leaflet.js displays sensor locations on an OpenStreetMap base layer. Users can toggle between PM2.5, PM10, and NO2 views. Circle markers are color-coded based on pollution severity, with popup tooltips providing detailed sensor information including coordinates, pollutant concentrations, and AQI values."

**Analytics:**
> "The analytics dashboard features two primary visualizations: (1) a time series line chart tracking PM2.5, PM10, and NO2 concentrations over time, revealing temporal patterns and trends; (2) a bar chart displaying average hourly pollution levels, identifying peak pollution hours. Additional statistics boxes provide comprehensive metrics including min, max, mean, median, and standard deviation."

**ML Model:**
> "The machine learning interface presents Random Forest model specifications including training/test sample counts and training date. Performance metrics (R² = 0.457, MAE = 11.85 µg/m³, RMSE = 15.32 µg/m³) validate model accuracy. A horizontal bar chart visualizes feature importance rankings, revealing which environmental factors most significantly influence predictions."

**Prediction:**
> "Interactive prediction functionality enables users to input environmental parameters (PM2.5, PM10, NO2, CO, SO2, O3, location coordinates) and receive real-time PM2.5 predictions. The form validates inputs and displays results with predicted concentration values and corresponding AQI category classification, demonstrating practical application of the trained model."

---

## 🎯 Submission Checklist

Before submitting, verify:

- [ ] All 5 core screenshots captured
- [ ] Screenshots show full browser window
- [ ] localhost:5000 visible in address bar
- [ ] All tabs demonstrated (Overview, Map, Analytics, ML Model)
- [ ] Prediction form filled and result shown
- [ ] Images are high quality (PNG format)
- [ ] File sizes reasonable (< 5 MB each)
- [ ] Screenshots named consistently
- [ ] Optional API screenshots included
- [ ] Optional demo video recorded
- [ ] Screenshots referenced in REPORT.md
- [ ] Captions written for each screenshot
- [ ] Screenshots folder organized
- [ ] Ready to zip for submission

---

## 💡 Pro Tips

1. **Take screenshots in order** (Overview → Map → Analytics → ML Model)
2. **Use consistent window size** for all screenshots
3. **Clear browser cache** before screenshots (Ctrl+Shift+R)
4. **Close unnecessary tabs** for clean browser UI
5. **Disable browser extensions** that might show in UI
6. **Use incognito mode** for clean screenshots
7. **Verify data loaded** before taking screenshot
8. **Check timestamp** in Overview tab is recent
9. **Test on multiple browsers** if time permits
10. **Keep originals** - can crop/edit copies later

---

## 🎓 For Your Report

### Section: Web Frontend Development

**Technologies:**
- Flask 3.0.0 (Backend API)
- HTML5, CSS3, JavaScript ES6+
- Leaflet.js 1.9.4 (Mapping)
- Chart.js 4.4.0 (Visualizations)
- Font Awesome 6.4.0 (Icons)

**Features Implemented:**
1. Real-time data dashboard
2. Interactive geospatial visualization
3. Time series analytics
4. ML model interface
5. Prediction functionality
6. RESTful API architecture
7. Responsive design
8. CORS-enabled endpoints

**Code Statistics:**
- Backend: 280 lines (Python)
- Frontend HTML: 330 lines
- CSS: 550+ lines
- JavaScript: 450+ lines
- Total: ~1,600 lines of code

---

*This guide ensures comprehensive documentation of your web frontend for lab submission!*
