// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5001/api' 
    : '/api';

// Global variables
let map, charts = {};
let sensorsData = [];
let currentPollutant = 'pm25';

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 App initializing...');
    console.log('API URL:', API_BASE_URL);
    
    initNavigation();
    loadOverviewData();
    initMap();
    initCharts();
    loadAnalyticsData();
    loadModelInfo();
    setupPredictionForm();
    
    console.log('✅ App initialization complete');
});

// Navigation
function initNavigation() {
    console.log('📍 Initializing navigation...');
    const navLinks = document.querySelectorAll('.nav-link');
    console.log('Found', navLinks.length, 'navigation links');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabName = link.dataset.tab;
            console.log('🔄 Switching to tab:', tabName);
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    console.log('Switching to:', tabName);
    
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.tab === tabName) {
            link.classList.add('active');
        }
    });

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
        targetTab.classList.add('active');
        console.log('✅ Tab switched to:', tabName);
    } else {
        console.error('❌ Tab not found:', tabName);
    }

    // Resize charts when switching to analytics tab
    if (tabName === 'analytics') {
        setTimeout(() => {
            Object.values(charts).forEach(chart => chart.resize());
        }, 100);
    }

    // Update map when switching to map tab
    if (tabName === 'map') {
        setTimeout(() => {
            if (map) map.invalidateSize();
        }, 100);
    }
}

// Overview Tab
async function loadOverviewData() {
    console.log('📊 Loading overview data...');
    try {
        const [overview, sensorsResponse] = await Promise.all([
            fetchAPI('/overview'),
            fetchAPI('/sensors')
        ]);

        console.log('✅ Overview data:', overview);
        console.log('✅ Sensors data:', sensorsResponse);

        sensorsData = sensorsResponse.sensors || [];
        console.log('Found', sensorsData.length, 'sensors');

        // Update stats
        document.getElementById('total-sensors').textContent = overview.total_sensors;
        document.getElementById('total-readings').textContent = overview.total_readings.toLocaleString();
        document.getElementById('avg-pm25').textContent = overview.avg_pm25.toFixed(1);
        document.getElementById('avg-aqi').textContent = Math.round(overview.avg_aqi);

        // Update sensors table
        updateSensorsTable(sensorsData);
    } catch (error) {
        console.error('❌ Error loading overview data:', error);
        showError('Failed to load overview data: ' + error.message);
    }
}

function updateSensorsTable(sensors) {
    const tbody = document.getElementById('sensors-tbody');
    tbody.innerHTML = sensors.map(sensor => {
        const statusClass = getAQIStatus(sensor.readings.aqi);
        return `
            <tr>
                <td>${sensor.id}</td>
                <td>${sensor.name}</td>
                <td>${sensor.readings.pm25.toFixed(1)}</td>
                <td>${sensor.readings.pm10.toFixed(1)}</td>
                <td>${sensor.readings.no2.toFixed(1)}</td>
                <td><span class="status-badge status-${statusClass}">${sensor.readings.aqi}</span></td>
                <td>${new Date(sensor.readings.timestamp).toLocaleString()}</td>
            </tr>
        `;
    }).join('');
}

function getAQIStatus(aqi) {
    if (aqi <= 50) return 'good';
    if (aqi <= 100) return 'moderate';
    return 'unhealthy';
}

// Map Tab
function initMap() {
    map = L.map('map-container').setView([28.6139, 77.2090], 11);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    // Pollutant selector
    document.getElementById('pollutant-select').addEventListener('change', (e) => {
        currentPollutant = e.target.value;
        updateMapMarkers();
    });

    updateMapMarkers();
}

async function updateMapMarkers() {
    if (!map) return;

    try {
        const sensorsResponse = await fetchAPI('/sensors');
        const sensors = sensorsResponse.sensors || [];
        
        // Clear existing markers
        map.eachLayer(layer => {
            if (layer instanceof L.Marker || layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });

        // Add new markers
        sensors.forEach(sensor => {
            const value = sensor.readings[currentPollutant];
            const color = getPollutantColor(currentPollutant, value);
            
            const marker = L.circleMarker([sensor.latitude, sensor.longitude], {
                radius: 12,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map);

            marker.bindPopup(`
                <strong>${sensor.name}</strong><br>
                ${currentPollutant.toUpperCase()}: ${value.toFixed(1)} µg/m³<br>
                AQI: ${sensor.readings.aqi}<br>
                ${new Date(sensor.readings.timestamp).toLocaleString()}
            `);
        });
    } catch (error) {
        console.error('Error updating map:', error);
    }
}

function getPollutantColor(pollutant, value) {
    const thresholds = {
        pm25: [12, 35, 55, 150],
        pm10: [50, 100, 250, 350],
        no2: [40, 90, 120, 230]
    };

    const levels = thresholds[pollutant] || thresholds.pm25;
    if (value <= levels[0]) return '#10b981';
    if (value <= levels[1]) return '#84cc16';
    if (value <= levels[2]) return '#f59e0b';
    if (value <= levels[3]) return '#ef4444';
    return '#991b1b';
}

// Analytics Tab
function initCharts() {
    // Time Series Chart
    const timeCtx = document.getElementById('timeSeriesChart').getContext('2d');
    charts.timeseries = new Chart(timeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: { display: true, text: 'Pollutant Time Series' }
            },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Concentration (µg/m³)' } },
                x: { title: { display: true, text: 'Time' } }
            }
        }
    });

    // Hourly Pattern Chart
    const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
    charts.hourly = new Chart(hourlyCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: { display: true, text: 'Average Hourly Patterns' }
            },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Concentration (µg/m³)' } },
                x: { title: { display: true, text: 'Hour of Day' } }
            }
        }
    });

    // Feature Importance Chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    charts.features = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Importance Score',
                data: [],
                backgroundColor: '#3b82f6'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y',
            plugins: {
                title: { display: true, text: 'Feature Importance' }
            }
        }
    });
}

async function loadAnalyticsData() {
    try {
        // Load time series data
        const sensorsResponse = await fetchAPI('/sensors');
        const sensors = sensorsResponse.sensors || [];
        if (sensors.length > 0) {
            const sensorId = sensors[0].id;
            const timeseries = await fetchAPI(`/timeseries/${sensorId}`);
            updateTimeSeriesChart(timeseries);
        }

        // Load hourly patterns
        const hourly = await fetchAPI('/hourly_pattern');
        updateHourlyChart(hourly);

        // Load statistics
        const overview = await fetchAPI('/overview');
        updateStatistics(overview.pollutant_stats);
        
        // Calculate and display correlations
        if (sensors.length > 0) {
            const sensorId = sensors[0].id;
            const timeseries = await fetchAPI(`/timeseries/${sensorId}`);
            updateCorrelations(timeseries.data);
        }

    } catch (error) {
        console.error('Error loading analytics data:', error);
    }
}

function updateTimeSeriesChart(response) {
    const data = response.data;
    const labels = data.timestamps.map(ts => new Date(ts).toLocaleDateString());
    
    charts.timeseries.data.labels = labels;
    charts.timeseries.data.datasets = [
        {
            label: 'PM2.5',
            data: data.pm25,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.4
        },
        {
            label: 'PM10',
            data: data.pm10,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.4
        },
        {
            label: 'NO2',
            data: data.no2,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4
        }
    ];
    charts.timeseries.update();
}

function updateHourlyChart(data) {
    charts.hourly.data.labels = data.hours;
    charts.hourly.data.datasets = [
        {
            label: 'PM2.5',
            data: data.pm25,
            backgroundColor: 'rgba(239, 68, 68, 0.7)'
        },
        {
            label: 'PM10',
            data: data.pm10,
            backgroundColor: 'rgba(245, 158, 11, 0.7)'
        },
        {
            label: 'NO2',
            data: data.no2,
            backgroundColor: 'rgba(59, 130, 246, 0.7)'
        }
    ];
    charts.hourly.update();
}

function updateStatistics(stats) {
    const container = document.getElementById('stats-container');
    if (!stats) return;
    
    const flatStats = {};
    Object.keys(stats).forEach(pollutant => {
        Object.keys(stats[pollutant]).forEach(metric => {
            const key = `${pollutant.toUpperCase()}_${metric}`;
            flatStats[key] = stats[pollutant][metric];
        });
    });
    
    container.innerHTML = Object.entries(flatStats).map(([key, value]) => `
        <div class="stat-box">
            <div class="value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
            <div class="label">${key.replace(/_/g, ' ')}</div>
        </div>
    `).join('');
}

function updateCorrelations(data) {
    const container = document.getElementById('correlation-container');
    if (!data) return;
    
    const pollutants = ['pm25', 'pm10', 'no2'];
    const pollutantNames = ['PM2.5', 'PM10', 'NO2'];
    
    // Calculate correlation matrix
    const correlations = [];
    pollutants.forEach((pol1, i) => {
        const row = [];
        pollutants.forEach((pol2, j) => {
            const corr = calculateCorrelation(data[pol1], data[pol2]);
            row.push(corr);
        });
        correlations.push(row);
    });
    
    // Display as table
    let html = '<table style="width: 100%; border-collapse: collapse;">';
    html += '<tr><th></th>' + pollutantNames.map(n => `<th>${n}</th>`).join('') + '</tr>';
    correlations.forEach((row, i) => {
        html += `<tr><th>${pollutantNames[i]}</th>`;
        row.forEach(val => {
            const color = getCorrelationColor(val);
            html += `<td style="background: ${color}; padding: 10px; text-align: center; border: 1px solid #ddd;">${val.toFixed(2)}</td>`;
        });
        html += '</tr>';
    });
    html += '</table>';
    
    container.innerHTML = html;
}

function calculateCorrelation(arr1, arr2) {
    const n = arr1.length;
    const sum1 = arr1.reduce((a, b) => a + b, 0);
    const sum2 = arr2.reduce((a, b) => a + b, 0);
    const sum1Sq = arr1.reduce((a, b) => a + b * b, 0);
    const sum2Sq = arr2.reduce((a, b) => a + b * b, 0);
    const pSum = arr1.reduce((sum, val, i) => sum + val * arr2[i], 0);
    
    const num = pSum - (sum1 * sum2 / n);
    const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
    
    return den === 0 ? 0 : num / den;
}

function getCorrelationColor(value) {
    const absVal = Math.abs(value);
    if (absVal < 0.3) return '#f0f0f0';
    if (absVal < 0.5) return '#bbdefb';
    if (absVal < 0.7) return '#64b5f6';
    if (absVal < 0.9) return '#2196f3';
    return '#1565c0';
}

// Model Tab
async function loadModelInfo() {
    try {
        const modelInfo = await fetchAPI('/model_info');
        
        // Update model info
        document.getElementById('model-type').textContent = modelInfo.model_type;
        
        console.log('✅ Model info loaded:', modelInfo);

        // Update metrics
        document.getElementById('metric-r2').textContent = modelInfo.performance.r2_score.toFixed(3);
        document.getElementById('metric-mae').textContent = modelInfo.performance.mae.toFixed(2);
        document.getElementById('metric-rmse').textContent = modelInfo.performance.rmse.toFixed(2);

        // Update feature importance chart
        const features = modelInfo.feature_importance;
        charts.features.data.labels = features.map(f => f.feature);
        charts.features.data.datasets[0].data = features.map(f => f.importance);
        charts.features.update();

    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

function setupPredictionForm() {
    // Prediction form is called via onclick in HTML
    console.log('✅ Prediction form ready');
}

async function makePrediction() {
    console.log('🔮 Making prediction...');
    
    const formData = {
        latitude: parseFloat(document.getElementById('pred-lat').value),
        longitude: parseFloat(document.getElementById('pred-lon').value),
        temperature: parseFloat(document.getElementById('pred-temp').value),
        humidity: parseFloat(document.getElementById('pred-humidity').value),
        hour: parseInt(document.getElementById('pred-hour').value)
    };

    try {
        const result = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        }).then(res => res.json());

        displayPrediction(result);
    } catch (error) {
        console.error('❌ Error making prediction:', error);
        showError('Failed to make prediction: ' + error.message);
    }
}

function displayPrediction(result) {
    const resultDiv = document.getElementById('prediction-result');
    const pm25 = result.predicted_pm25 || 0;
    
    let category = 'Good';
    let color = '#10b981';
    if (pm25 > 12) { category = 'Moderate'; color = '#f59e0b'; }
    if (pm25 > 35) { category = 'Unhealthy for Sensitive'; color = '#ef4444'; }
    if (pm25 > 55) { category = 'Unhealthy'; color = '#991b1b'; }
    
    resultDiv.innerHTML = `
        <h3>Prediction Result</h3>
        <div class="result-value" style="color: ${color};">${pm25.toFixed(2)} µg/m³</div>
        <p>Category: <strong style="color: ${color};">${category}</strong></p>
        <p>Location: ${result.location.latitude.toFixed(4)}, ${result.location.longitude.toFixed(4)}</p>
    `;
    resultDiv.classList.add('show');
}

// Utility Functions
async function fetchAPI(endpoint) {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('🌐 Fetching:', url);
    
    try {
        const response = await fetch(url);
        console.log('📡 Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('✅ Data received:', endpoint, data);
        return data;
    } catch (error) {
        console.error('❌ Fetch error:', error);
        throw error;
    }
}

function updateLocationCoords() {
    const select = document.getElementById('location-select');
    const value = select.value;
    if (value) {
        const [lat, lon] = value.split(',');
        document.getElementById('pred-lat').value = lat;
        document.getElementById('pred-lon').value = lon;
    }
}

function showError(message) {
    console.error(message);
    // You can implement a toast notification here
    alert(message);
}
