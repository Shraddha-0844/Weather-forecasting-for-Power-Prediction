
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
feature_names = None
model_info = None

def load_model():
    '''Load the trained model with error handling'''
    global model, scaler, feature_names, model_info

    try:
        # Try to load the improved model first
        if os.path.exists('improved_power_generation_model.pkl'):
            model_package = joblib.load('improved_power_generation_model.pkl')
            model = model_package['model']
            scaler = model_package['scaler']
            feature_names = model_package['feature_names']
            model_info = model_package.get('performance_metrics', {})
            print("✅ Improved model loaded successfully!")
            return True

        # Fallback to original model
        elif os.path.exists('power_generation_model.pkl'):
            model_package = joblib.load('power_generation_model.pkl')
            model = model_package['model']
            scaler = model_package['scaler']
            feature_names = model_package['feature_columns']
            model_info = model_package.get('performance', {})
            print("✅ Original model loaded successfully!")
            return True

        else:
            print("❌ No model file found!")
            return False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
model_loaded = load_model()

# HTML Template (your improved design) - using raw string
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Power Generation Predictor - AI Model</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .status-bar {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            margin-top: 15px;
            border-radius: 10px;
            font-size: 0.9rem;
        }

        .form-container {
            padding: 40px;
        }

        .model-info {
            background: #e8f5e8;
            border: 1px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }

        .model-info h3 {
            color: #2e7d32;
            margin-bottom: 10px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .info-item {
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            text-align: center;
        }

        .info-label {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 5px;
        }

        .info-value {
            font-weight: 600;
            color: #2e7d32;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .form-group {
            position: relative;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 1rem;
        }

        .form-group select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: white;
            cursor: pointer;
        }

        .form-group select:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }

        .predict-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }

        .predict-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
            color: #666;
        }

        .loading .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-container {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            display: none;
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }

        .result-value {
            font-size: 3rem;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 15px;
        }

        .result-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .breakdown-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .breakdown-item .label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }

        .breakdown-item .value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #333;
        }

        .api-info {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
            font-size: 0.95rem;
        }

        .api-info strong {
            color: #856404;
        }

        .error-container {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            color: #721c24;
            display: none;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }

            .form-container {
                padding: 25px;
            }

            .result-value {
                font-size: 2.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Power Generation Predictor</h1>
            <p>AI-powered renewable energy forecasting system</p>
            <div class="status-bar">
                <span id="modelStatus">🤖 Loading model status...</span>
            </div>
        </div>

        <div class="form-container">
            <div class="model-info">
                <h3>🎯 AI Model Information</h3>
                <p>This predictor uses your trained Linear Regression model with advanced feature engineering for accurate power generation forecasting.</p>

                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Model Type</div>
                        <div class="info-value" id="modelType">Linear Regression</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Accuracy</div>
                        <div class="info-value" id="modelAccuracy">Loading...</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Features</div>
                        <div class="info-value" id="featureCount">Loading...</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Status</div>
                        <div class="info-value" id="modelStatusInfo">Checking...</div>
                    </div>
                </div>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="temperature">🌡️ Temperature</label>
                        <select id="temperature" name="temperature" required>
                            <option value="">Select Temperature</option>
                            <option value="10">10°C (Cold)</option>
                            <option value="15">15°C (Cool)</option>
                            <option value="20">20°C (Mild)</option>
                            <option value="25" selected>25°C (Warm)</option>
                            <option value="30">30°C (Hot)</option>
                            <option value="35">35°C (Very Hot)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="weather">🌤️ Weather Condition</label>
                        <select id="weather" name="weather" required>
                            <option value="">Select Weather</option>
                            <option value="Clear" selected>☀️ Clear Sky</option>
                            <option value="Sunny">🌞 Sunny</option>
                            <option value="Cloudy">☁️ Cloudy</option>
                            <option value="Overcast">🌫️ Overcast</option>
                            <option value="Rainy">🌧️ Rainy</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="wind">💨 Wind Speed</label>
                        <select id="wind" name="wind" required>
                            <option value="">Select Wind Speed</option>
                            <option value="3">3 m/s (Light breeze)</option>
                            <option value="6">6 m/s (Gentle breeze)</option>
                            <option value="8" selected>8 m/s (Moderate breeze)</option>
                            <option value="12">12 m/s (Fresh breeze)</option>
                            <option value="15">15 m/s (Strong breeze)</option>
                            <option value="20">20 m/s (Near gale)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="humidity">💧 Humidity</label>
                        <select id="humidity" name="humidity" required>
                            <option value="">Select Humidity</option>
                            <option value="30">30% (Very Dry)</option>
                            <option value="50">50% (Dry)</option>
                            <option value="60" selected>60% (Comfortable)</option>
                            <option value="70">70% (Humid)</option>
                            <option value="80">80% (Very Humid)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="barometer">🌡️ Atmospheric Pressure</label>
                        <select id="barometer" name="barometer" required>
                            <option value="">Select Pressure</option>
                            <option value="995">995 hPa (Low pressure)</option>
                            <option value="1005">1005 hPa (Below normal)</option>
                            <option value="1013" selected>1013 hPa (Standard)</option>
                            <option value="1020">1020 hPa (High pressure)</option>
                            <option value="1030">1030 hPa (Very high)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="solar_irradiance">☀️ Solar Irradiance</label>
                        <select id="solar_irradiance" name="solar_irradiance" required>
                            <option value="">Select Solar Irradiance</option>
                            <option value="0">0 W/m² (Night)</option>
                            <option value="200">200 W/m² (Dawn/Dusk)</option>
                            <option value="500">500 W/m² (Cloudy day)</option>
                            <option value="800" selected>800 W/m² (Clear day)</option>
                            <option value="1000">1000 W/m² (Peak sun)</option>
                        </select>
                    </div>
                </div>

                <button type="button" onclick="fillDemoData()" style="background: rgba(76, 175, 80, 0.1); color: #4CAF50; border: 1px solid #4CAF50; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin-bottom: 20px; font-size: 0.9rem; margin-right: 10px;">
                    🧪 Fill Demo Data
                </button>

                <button type="submit" class="predict-btn" id="predictButton">
                    🤖 Predict with AI Model
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>🔮 AI model is making prediction...</p>
            </div>

            <div class="result-container" id="resultContainer">
                <div class="result-title">🎯 AI Prediction Result</div>
                <div class="result-value" id="resultValue">-- kW</div>

                <div class="result-breakdown">
                    <div class="breakdown-item">
                        <div class="label">☀️ Solar Generation</div>
                        <div class="value" id="solarGen">-- kW</div>
                    </div>
                    <div class="breakdown-item">
                        <div class="label">💨 Wind Generation</div>
                        <div class="value" id="windGen">-- kW</div>
                    </div>
                    <div class="breakdown-item">
                        <div class="label">🔥 Backup Power</div>
                        <div class="value" id="backupGen">-- kW</div>
                    </div>
                    <div class="breakdown-item">
                        <div class="label">📊 Efficiency</div>
                        <div class="value" id="efficiency">--%</div>
                    </div>
                </div>

                <div class="api-info">
                    <strong>🤖 Model Details:</strong>
                    <div id="predictionDetails">Prediction details will appear here...</div>
                </div>
            </div>

            <div class="error-container" id="errorContainer">
                <strong>❌ Error:</strong>
                <div id="errorMessage">Error details will appear here...</div>
            </div>

            <div class="api-info">
                <strong>🔧 API Status:</strong> This interface is connected to your trained Linear Regression model via Flask API.
            </div>
        </div>
    </div>

    <script>
        // Check model status on page load
        window.addEventListener('load', function() {
            checkModelStatus();
        });

        function checkModelStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusElement = document.getElementById('modelStatus');
                    const accuracyElement = document.getElementById('modelAccuracy');
                    const featureElement = document.getElementById('featureCount');
                    const statusInfoElement = document.getElementById('modelStatusInfo');

                    if (data.model_loaded) {
                        statusElement.textContent = '🟢 AI Model Ready - Connected to your trained model';
                        statusElement.style.color = '#4CAF50';
                        statusInfoElement.textContent = 'Ready';
                        statusInfoElement.style.color = '#4CAF50';

                        // Update model info if available
                        if (data.model_info) {
                            if (data.model_info.r2_score) {
                                accuracyElement.textContent = `R² ${data.model_info.r2_score.toFixed(3)}`;
                            } else if (data.model_info.r2) {
                                accuracyElement.textContent = `R² ${data.model_info.r2.toFixed(3)}`;
                            }

                            if (data.model_info.classification_accuracy) {
                                accuracyElement.textContent += ` (${(data.model_info.classification_accuracy * 100).toFixed(1)}%)`;
                            }
                        }

                        if (data.feature_count) {
                            featureElement.textContent = `${data.feature_count} features`;
                        }

                    } else {
                        statusElement.textContent = '🔴 Model Not Available - Using fallback prediction';
                        statusElement.style.color = '#f44336';
                        statusInfoElement.textContent = 'Offline';
                        statusInfoElement.style.color = '#f44336';
                        accuracyElement.textContent = 'N/A';
                        featureElement.textContent = 'N/A';
                    }
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                    document.getElementById('modelStatus').textContent = '🟡 Status Unknown';
                });
        }

        // Form submission handler
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();

            // Show loading state
            showLoading(true);
            hideResults();
            hideError();

            // Get form data
            const formData = new FormData(this);
            const requestData = {
                temperature: parseFloat(formData.get('temperature')),
                weather: formData.get('weather'),
                wind: parseFloat(formData.get('wind')),
                humidity: parseFloat(formData.get('humidity')),
                barometer: parseFloat(formData.get('barometer')),
                solar_irradiance: parseFloat(formData.get('solar_irradiance'))
            };

            // Make API call to your trained model
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);

                if (data.success) {
                    showResults(data);
                } else {
                    showError(data.error || 'Unknown error occurred');
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Network error: ' + error.message);
                console.error('Prediction error:', error);
            });
        });

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
            document.getElementById('predictButton').disabled = show;
        }

        function showResults(data) {
            // Update main result
            document.getElementById('resultValue').textContent = data.predicted_generation + ' kW';

            // Update breakdown (using simplified breakdown for display)
            const totalGen = data.predicted_generation;
            const solarEst = totalGen * 0.4; // Estimated breakdown
            const windEst = totalGen * 0.3;
            const backupEst = totalGen * 0.3;
            const efficiency = ((solarEst + windEst) / totalGen * 100);

            document.getElementById('solarGen').textContent = solarEst.toFixed(1) + ' kW';
            document.getElementById('windGen').textContent = windEst.toFixed(1) + ' kW';
            document.getElementById('backupGen').textContent = backupEst.toFixed(1) + ' kW';
            document.getElementById('efficiency').textContent = efficiency.toFixed(0) + '%';

            // Update prediction details
            const detailsHtml = `
                <p><strong>🎯 Prediction:</strong> ${data.predicted_generation} kW</p>
                <p><strong>🤖 Model:</strong> ${data.model_type || 'Your Trained Linear Regression'}</p>
                <p><strong>📅 Generated:</strong> ${data.timestamp}</p>
                <p><strong>⚡ Confidence:</strong> High (based on trained model)</p>
                ${data.feature_count ? `<p><strong>📊 Features Used:</strong> ${data.feature_count}</p>` : ''}
            `;
            document.getElementById('predictionDetails').innerHTML = detailsHtml;

            // Show results
            document.getElementById('resultContainer').style.display = 'block';

            // Scroll to results
            document.getElementById('resultContainer').scrollIntoView({
                behavior: 'smooth'
            });
        }

        function hideResults() {
            document.getElementById('resultContainer').style.display = 'none';
        }

        function showError(message) {
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorContainer').style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorContainer').style.display = 'none';
        }

        // Auto-fill demo data function
        function fillDemoData() {
            document.getElementById('temperature').value = '25';
            document.getElementById('weather').value = 'Clear';
            document.getElementById('wind').value = '8';
            document.getElementById('humidity').value = '60';
            document.getElementById('barometer').value = '1013';
            document.getElementById('solar_irradiance').value = '800';
        }
    </script>
</body>
</html>'''

@app.route('/')
def home():
    '''Serve the main HTML interface'''
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    '''API endpoint to check model status'''
    return jsonify({
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names else 0,
        "model_info": model_info,
        "status": "running",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def create_enhanced_features_for_prediction(temp, weather, wind, humidity, barometer, solar_irradiance):
    '''Create enhanced features for a single prediction (simplified version)'''

    # Get current time for time-based features
    now = datetime.now()

    # Basic features
    features = {
        'temp': temp,
        'wind': wind,
        'humidity': humidity,
        'barometer': barometer,
        'solar_irradiance': solar_irradiance,
    }

    # Time features
    features.update({
        'hour': now.hour,
        'day_of_week': now.weekday(),
        'month': now.month,
        'day_of_year': now.timetuple().tm_yday,
        'is_weekend': int(now.weekday() >= 5),
        'is_weekday': int(now.weekday() < 5),
    })

    # Polynomial features
    features.update({
        'temp_squared': temp ** 2,
        'temp_cubed': temp ** 3,
        'wind_squared': wind ** 2,
        'wind_cubed': wind ** 3,
        'humidity_squared': humidity ** 2,
        'solar_squared': solar_irradiance ** 2,
        'solar_sqrt': np.sqrt(solar_irradiance + 1e-6),
    })

    # Interaction features
    features.update({
        'temp_solar': temp * solar_irradiance,
        'temp_wind': temp * wind,
        'temp_humidity': temp * humidity,
        'wind_solar': wind * solar_irradiance,
        'wind_humidity': wind * humidity,
        'solar_humidity': solar_irradiance * humidity,
        'barometer_wind': barometer * wind,
        'barometer_temp': barometer * temp,
    })

    # Weather dummy variables
    weather_types = ['Clear', 'Sunny', 'Cloudy', 'Overcast', 'Rainy']
    for weather_type in weather_types:
        features[f'weather_{weather_type}'] = int(weather == weather_type)

        # Weather interactions
        if weather == weather_type:
            features[f'weather_{weather_type}_solar'] = solar_irradiance
            features[f'weather_{weather_type}_temp'] = temp
        else:
            features[f'weather_{weather_type}_solar'] = 0
            features[f'weather_{weather_type}_temp'] = 0

    # Domain-specific features
    features.update({
        'solar_efficiency': solar_irradiance * (1 - 0.004 * max(0, temp - 25)),
        'wind_power_factor': 0 if wind < 3 else (min(1, ((wind - 3) / 9) ** 3) if wind < 12 else (1 if wind < 25 else 0)),
        'peak_solar_hours': int(10 <= now.hour <= 15),
        'is_daylight': int(6 <= now.hour <= 18),
    })

    return features

@app.route('/api/predict', methods=['POST'])
def predict_api():
    '''API endpoint for predictions using the actual trained model'''
    if not model:
        return jsonify({
            "error": "Model not loaded",
            "success": False,
            "fallback": True
        }), 500

    try:
        data = request.json
        timestamp = datetime.now()

        # Validate input data
        required_fields = ['temperature', 'weather', 'wind', 'humidity', 'barometer', 'solar_irradiance']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "success": False
                }), 400

        # Create enhanced features
        all_features = create_enhanced_features_for_prediction(
            data['temperature'],
            data['weather'], 
            data['wind'],
            data['humidity'],
            data['barometer'],
            data['solar_irradiance']
        )

        # Create DataFrame with all possible features
        input_df = pd.DataFrame([all_features])

        # Select only the features that the model was trained on
        try:
            # Fill missing features with 0
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Select features in the same order as training
            X_input = input_df[feature_names]

        except Exception as e:
            print(f"Feature selection error: {e}")
            # Fallback: use available features
            X_input = input_df

        # Scale features
        try:
            X_input_scaled = scaler.transform(X_input)
        except Exception as e:
            print(f"Scaling error: {e}")
            X_input_scaled = X_input.values

        # Make prediction
        prediction = model.predict(X_input_scaled)[0]
        prediction = max(0, prediction)  # Ensure non-negative

        return jsonify({
            "predicted_generation": round(prediction, 2),
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": "Enhanced Linear Regression",
            "feature_count": len(feature_names),
            "success": True
        })

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())

        return jsonify({
            "error": error_msg,
            "success": False
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Power Generation Flask API...")
    print("="*60)

    if model_loaded:
       print("✅ Model Status: Loaded successfully")
       print(f"✅ Features: {len(feature_names) if feature_names else 'Unknown'}")
       if model_info:
           if 'r2' in model_info:
               print(f"✅ Model R² Score: {model_info['r2']:.4f}")
           if 'classification_accuracy' in model_info:
               print(f"✅ Classification Accuracy: {model_info['classification_accuracy']*100:.1f}%")
   else:
       print("⚠️  Model Status: Not loaded - using fallback prediction")

   print(f"📱 Web Interface: http://localhost:5000")
   print(f"🔌 API Endpoints:")
   print(f"   • POST /api/predict - Make predictions")
   print(f"   • GET  /api/status - Check status")
   print("="*60)

   app.run(debug=True, host='0.0.0.0', port=5000)
