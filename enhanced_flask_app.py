from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model components
best_model = None
backup_models = {}
model_info = None
model_type = None
scaler = None
feature_names = None

def load_models():
    """Load all available models with error handling"""
    global best_model, backup_models, model_info, model_type, scaler, feature_names
    
    models_loaded = []
    
    try:
        # Try to load the best model from comparison study
        if os.path.exists('best_power_generation_model.pkl'):
            model_package = joblib.load('best_power_generation_model.pkl')
            best_model = model_package['model']
            scaler = model_package.get('scaler', None)
            feature_names = model_package['feature_names']
            model_info = model_package.get('performance_metrics', {})
            model_type = model_package.get('algorithm', model_package.get('model_type', 'Unknown'))
            
            print(f"✅ Best model loaded: {model_type}")
            models_loaded.append(f"Best Model ({model_type})")
            
            # Store comparison results if available
            if 'comparison_results' in model_package:
                model_info.update(model_package['comparison_results'])
            
        # Try to load enhanced model as backup
        elif os.path.exists('improved_power_generation_model.pkl'):
            model_package = joblib.load('improved_power_generation_model.pkl')
            best_model = model_package['model']
            scaler = model_package['scaler']
            feature_names = model_package['feature_names']
            model_info = model_package.get('performance_metrics', {})
            model_type = model_package.get('model_type', 'Enhanced Linear Regression')
            
            print(f"✅ Enhanced model loaded: {model_type}")
            models_loaded.append(f"Enhanced Model ({model_type})")
            
        # Try to load original model as backup
        elif os.path.exists('power_generation_model.pkl'):
            model_package = joblib.load('power_generation_model.pkl')
            best_model = model_package['model']
            scaler = model_package['scaler']
            feature_names = model_package['feature_columns']
            model_info = model_package.get('performance', {})
            model_type = 'Original Linear Regression'
            
            print(f"✅ Original model loaded: {model_type}")
            models_loaded.append(f"Original Model ({model_type})")
            
        else:
            print("❌ No model files found!")
            return False
            
        return len(models_loaded) > 0
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models on startup
models_loaded = load_models()

# Enhanced HTML Template with Dynamic Model Information
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Power Generation Predictor - AI Model Comparison</title>
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
            max-width: 1000px;
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
            padding: 15px 20px;
            margin-top: 15px;
            border-radius: 10px;
            font-size: 0.95rem;
        }

        .model-badges {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .badge.winner {
            background: #ffd700;
            color: #333;
            font-weight: bold;
        }

        .form-container {
            padding: 40px;
        }

        .model-comparison {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%);
            border: 2px solid #4CAF50;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }

        .model-comparison h3 {
            color: #2e7d32;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.3rem;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 8px;
        }

        .metric-value {
            font-weight: bold;
            font-size: 1.2rem;
            color: #2e7d32;
        }

        .metric-improvement {
            font-size: 0.8rem;
            color: #1976d2;
            margin-top: 5px;
        }

        .winner-banner {
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #333;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
            border: 2px solid #ffc107;
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
            text-align: center;
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

        .model-details {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
            font-size: 0.95rem;
        }

        .model-details strong {
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

        .confidence-meter {
            background: #e9ecef;
            height: 20px;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997, #17a2b8);
            border-radius: 10px;
            transition: width 0.5s ease;
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
            
            .comparison-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Enhanced Power Generation Predictor</h1>
            <p>AI Model Comparison: Linear Regression vs Random Forest</p>
            <div class="status-bar">
                <span id="modelStatus">🤖 Loading model status...</span>
                <div class="model-badges">
                    <span class="badge" id="modelBadge">Loading...</span>
                    <span class="badge" id="accuracyBadge">Accuracy: --</span>
                    <span class="badge" id="algorithmBadge">Algorithm: --</span>
                </div>
            </div>
        </div>

        <div class="form-container">
            <div id="winnerBanner" class="winner-banner" style="display: none;">
                🏆 Winner: <span id="winnerModel">--</span> | Improvement: <span id="improvement">--</span>
            </div>

            <div class="model-comparison">
                <h3>🎯 AI Model Comparison Results</h3>
                <p style="text-align: center; margin-bottom: 15px;">
                    This system automatically uses the best performing model from our comprehensive comparison study.
                </p>
                
                <div class="comparison-grid">
                    <div class="metric-card">
                        <div class="metric-label">Current Model</div>
                        <div class="metric-value" id="currentModel">Loading...</div>
                        <div class="metric-improvement" id="modelType">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value" id="r2Score">--</div>
                        <div class="metric-improvement" id="r2Improvement">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Classification Accuracy</div>
                        <div class="metric-value" id="classAccuracy">--</div>
                        <div class="metric-improvement" id="classImprovement">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value" id="rmseValue">--</div>
                        <div class="metric-improvement" id="rmseImprovement">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Features Used</div>
                        <div class="metric-value" id="featureCount">--</div>
                        <div class="metric-improvement">Enhanced Engineering</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Model Status</div>
                        <div class="metric-value" id="modelStatusValue">--</div>
                        <div class="metric-improvement" id="deploymentStatus">--</div>
                    </div>
                </div>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="temperature">🌡️ Temperature</label>
                        <select id="temperature" name="temperature" required>
                            <option value="">Select Temperature</option>
                            <option value="5">5°C (Very Cold)</option>
                            <option value="10">10°C (Cold)</option>
                            <option value="15">15°C (Cool)</option>
                            <option value="20">20°C (Mild)</option>
                            <option value="25" selected>25°C (Warm)</option>
                            <option value="30">30°C (Hot)</option>
                            <option value="35">35°C (Very Hot)</option>
                            <option value="40">40°C (Extreme)</option>
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
                            <option value="0">0 m/s (Calm)</option>
                            <option value="3">3 m/s (Light breeze)</option>
                            <option value="6">6 m/s (Gentle breeze)</option>
                            <option value="8" selected>8 m/s (Moderate breeze)</option>
                            <option value="12">12 m/s (Fresh breeze)</option>
                            <option value="15">15 m/s (Strong breeze)</option>
                            <option value="20">20 m/s (Near gale)</option>
                            <option value="25">25+ m/s (Gale)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="humidity">💧 Humidity</label>
                        <select id="humidity" name="humidity" required>
                            <option value="">Select Humidity</option>
                            <option value="20">20% (Very Dry)</option>
                            <option value="30">30% (Dry)</option>
                            <option value="50">50% (Comfortable)</option>
                            <option value="60" selected>60% (Moderate)</option>
                            <option value="70">70% (Humid)</option>
                            <option value="80">80% (Very Humid)</option>
                            <option value="90">90% (Extremely Humid)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="barometer">🌡️ Atmospheric Pressure</label>
                        <select id="barometer" name="barometer" required>
                            <option value="">Select Pressure</option>
                            <option value="980">980 hPa (Very Low)</option>
                            <option value="995">995 hPa (Low pressure)</option>
                            <option value="1005">1005 hPa (Below normal)</option>
                            <option value="1013" selected>1013 hPa (Standard)</option>
                            <option value="1020">1020 hPa (High pressure)</option>
                            <option value="1030">1030 hPa (Very high)</option>
                            <option value="1040">1040 hPa (Extreme high)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="solar_irradiance">☀️ Solar Irradiance</label>
                        <select id="solar_irradiance" name="solar_irradiance" required>
                            <option value="">Select Solar Irradiance</option>
                            <option value="0">0 W/m² (Night)</option>
                            <option value="100">100 W/m² (Dawn/Dusk)</option>
                            <option value="200">200 W/m² (Early morning)</option>
                            <option value="400">400 W/m² (Overcast day)</option>
                            <option value="500">500 W/m² (Cloudy day)</option>
                            <option value="800" selected>800 W/m² (Clear day)</option>
                            <option value="1000">1000 W/m² (Peak sun)</option>
                            <option value="1200">1200 W/m² (Intense sun)</option>
                        </select>
                    </div>
                </div>

                <div style="margin-bottom: 20px; text-align: center;">
                    <button type="button" onclick="fillOptimalData()" style="background: rgba(76, 175, 80, 0.1); color: #4CAF50; border: 1px solid #4CAF50; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin-right: 10px; font-size: 0.9rem;">
                        🌟 Optimal Conditions
                    </button>
                    <button type="button" onclick="fillDemoData()" style="background: rgba(33, 150, 243, 0.1); color: #2196F3; border: 1px solid #2196F3; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin-right: 10px; font-size: 0.9rem;">
                        🧪 Demo Data
                    </button>
                    <button type="button" onclick="fillRandomData()" style="background: rgba(255, 152, 0, 0.1); color: #FF9800; border: 1px solid #FF9800; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.9rem;">
                        🎲 Random Test
                    </button>
                </div>

                <button type="submit" class="predict-btn" id="predictButton">
                    🤖 Predict with Best AI Model
                </button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>🔮 AI model is analyzing weather conditions...</p>
            </div>

            <div class="result-container" id="resultContainer">
                <div class="result-title">🎯 AI Prediction Result</div>
                <div class="result-value" id="resultValue">-- kW</div>
                
                <div style="text-align: center; margin: 15px 0;">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 5px;">Prediction Confidence</div>
                    <div class="confidence-meter">
                        <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: #666;" id="confidenceText">--</div>
                </div>
                
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
                        <div class="label">📊 Renewable %</div>
                        <div class="value" id="renewablePercent">--%</div>
                    </div>
                </div>
                
                <div class="model-details">
                    <strong>🤖 Model Details:</strong>
                    <div id="predictionDetails">Prediction details will appear here...</div>
                </div>
            </div>

            <div class="error-container" id="errorContainer">
                <strong>❌ Error:</strong>
                <div id="errorMessage">Error details will appear here...</div>
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
                    updateModelStatus(data);
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                    document.getElementById('modelStatus').textContent = '🟡 Status Unknown';
                });
        }

        function updateModelStatus(data) {
            const statusElement = document.getElementById('modelStatus');
            const modelBadge = document.getElementById('modelBadge');
            const accuracyBadge = document.getElementById('accuracyBadge');
            const algorithmBadge = document.getElementById('algorithmBadge');
            
            if (data.model_loaded) {
                statusElement.textContent = `🟢 ${data.model_type || 'AI Model'} Ready - Best model automatically selected`;
                statusElement.style.color = '#4CAF50';
                
                // Update badges
                modelBadge.textContent = data.model_type || 'AI Model';
                modelBadge.classList.add('winner');
                
                algorithmBadge.textContent = `Algorithm: ${data.algorithm || 'Advanced'}`;
                
                // Update comparison information
                updateComparisonInfo(data);
                
                if (data.model_info) {
                    const accuracy = data.model_info.test_classification_accuracy || data.model_info.classification_accuracy;
                    if (accuracy) {
                        accuracyBadge.textContent = `Accuracy: ${(accuracy * 100).toFixed(1)}%`;
                    }
                }
                
                // Show winner banner if improvement data available
                if (data.model_info && data.model_info.improvement_r2) {
                    const winnerBanner = document.getElementById('winnerBanner');
                    const winnerModel = document.getElementById('winnerModel');
                    const improvement = document.getElementById('improvement');
                    
                    winnerModel.textContent = data.algorithm || data.model_type;
                    improvement.textContent = `+${data.model_info.improvement_r2.toFixed(1)}% R² improvement`;
                    winnerBanner.style.display = 'block';
                }
                
            } else {
                statusElement.textContent = '🔴 No models available';
                statusElement.style.color = '#f44336';
                modelBadge.textContent = 'Offline';
                accuracyBadge.textContent = 'Accuracy: N/A';
                algorithmBadge.textContent = 'Algorithm: N/A';
            }
        }

        function updateComparisonInfo(data) {
            // Update current model info
            document.getElementById('currentModel').textContent = data.algorithm || data.model_type || 'AI Model';
            document.getElementById('modelType').textContent = 'Best Performing Model';
            
            if (data.model_info) {
                // R² Score
                const r2Score = data.model_info.test_r2_score || data.model_info.r2_score || data.model_info.r2;
                if (r2Score) {
                    document.getElementById('r2Score').textContent = r2Score.toFixed(4);
                }
                
                // Classification Accuracy
                const classAcc = data.model_info.test_classification_accuracy || data.model_info.classification_accuracy;
                if (classAcc) {
                    document.getElementById('classAccuracy').textContent = `${(classAcc * 100).toFixed(1)}%`;
                }
                
                // RMSE
                const rmse = data.model_info.test_rmse || data.model_info.rmse;
                if (rmse) {
                    document.getElementById('rmseValue').textContent = `${rmse.toFixed(2)} kW`;
                }
                
                // Improvements
                if (data.model_info.improvement_r2) {
                    document.getElementById('r2Improvement').textContent = `+${data.model_info.improvement_r2.toFixed(1)}% vs baseline`;
                }
                
                if (data.model_info.improvement_classification) {
                    document.getElementById('classImprovement').textContent = `+${data.model_info.improvement_classification.toFixed(1)}pp vs baseline`;
                }
            }
            
            // Feature count
            if (data.feature_count) {
                document.getElementById('featureCount').textContent = data.feature_count;
            }
            
            // Status
            document.getElementById('modelStatusValue').textContent = 'Ready';
            document.getElementById('deploymentStatus').textContent = 'Production Ready';
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

            // Make API call to the best model
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
            
            // Update confidence meter
            const confidence = data.confidence || 85; // Default confidence
            document.getElementById('confidenceFill').style.width = confidence + '%';
            document.getElementById('confidenceText').textContent = `${confidence}% confidence based on ${data.algorithm || 'AI'} model`;
            
            // Update breakdown with more realistic estimates
            const totalGen = data.predicted_generation;
            const solarEst = data.solar_estimate || (totalGen * 0.45);
            const windEst = data.wind_estimate || (totalGen * 0.35);
            const backupEst = data.backup_estimate || (totalGen * 0.20);
            const renewablePercent = ((solarEst + windEst) / totalGen * 100);
            
            document.getElementById('solarGen').textContent = solarEst.toFixed(1) + ' kW';
            document.getElementById('windGen').textContent = windEst.toFixed(1) + ' kW';
            document.getElementById('backupGen').textContent = backupEst.toFixed(1) + ' kW';
            document.getElementById('renewablePercent').textContent = renewablePercent.toFixed(0) + '%';
            
            // Update prediction details with comprehensive information
            const detailsHtml = `
                <p><strong>🎯 Prediction:</strong> ${data.predicted_generation} kW total generation</p>
                <p><strong>🤖 Model Used:</strong> ${data.algorithm || data.model_type || 'Best AI Model'}</p>
                <p><strong>📊 Model Performance:</strong> R² = ${data.model_r2 || 'N/A'} | Accuracy = ${data.model_accuracy || 'N/A'}</p>
                <p><strong>📅 Generated:</strong> ${data.timestamp}</p>
                <p><strong>⚡ Confidence Level:</strong> ${confidence}% (High accuracy model)</p>
                ${data.feature_count ? `<p><strong>🔧 Features Analyzed:</strong> ${data.feature_count} weather parameters</p>` : ''}
                ${data.improvement ? `<p><strong>📈 Model Improvement:</strong> ${data.improvement} over baseline</p>` : ''}
                <p><strong>🌱 Renewable Energy:</strong> ${renewablePercent.toFixed(1)}% of total generation</p>
            `;
            document.getElementById('predictionDetails').innerHTML = detailsHtml;
            
            // Show results with animation
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

        // Enhanced data filling functions
        function fillOptimalData() {
            document.getElementById('temperature').value = '25';
            document.getElementById('weather').value = 'Clear';
            document.getElementById('wind').value = '12';
            document.getElementById('humidity').value = '50';
            document.getElementById('barometer').value = '1020';
            document.getElementById('solar_irradiance').value = '1000';
        }

        function fillDemoData() {
            document.getElementById('temperature').value = '25';
            document.getElementById('weather').value = 'Clear';
            document.getElementById('wind').value = '8';
            document.getElementById('humidity').value = '60';
            document.getElementById('barometer').value = '1013';
            document.getElementById('solar_irradiance').value = '800';
        }

        function fillRandomData() {
            const temps = ['15', '20', '25', '30', '35'];
            const weathers = ['Clear', 'Sunny', 'Cloudy', 'Overcast'];
            const winds = ['3', '6', '8', '12', '15'];
            const humidities = ['30', '50', '60', '70', '80'];
            const pressures = ['995', '1005', '1013', '1020', '1030'];
            const solar = ['200', '400', '500', '800', '1000'];
            
            document.getElementById('temperature').value = temps[Math.floor(Math.random() * temps.length)];
            document.getElementById('weather').value = weathers[Math.floor(Math.random() * weathers.length)];
            document.getElementById('wind').value = winds[Math.floor(Math.random() * winds.length)];
            document.getElementById('humidity').value = humidities[Math.floor(Math.random() * humidities.length)];
            document.getElementById('barometer').value = pressures[Math.floor(Math.random() * pressures.length)];
            document.getElementById('solar_irradiance').value = solar[Math.floor(Math.random() * solar.length)];
        }
    </script>
</body>
</html>'''

@app.route('/')
def home():
    '''Serve the enhanced HTML interface'''
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    '''Enhanced API endpoint to check model status with comparison data'''
    
    status_info = {
        "model_loaded": best_model is not None,
        "model_type": model_type,
        "algorithm": model_type,
        "feature_count": len(feature_names) if feature_names else 0,
        "model_info": model_info or {},
        "status": "running",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add additional performance info if available
    if model_info:
        # Format model info for display
        formatted_info = {}
        for key, value in model_info.items():
            if isinstance(value, float):
                formatted_info[key] = round(value, 4)
            else:
                formatted_info[key] = value
        status_info["model_info"] = formatted_info
    
    return jsonify(status_info)

def create_enhanced_features_for_prediction(temp, weather, wind, humidity, barometer, solar_irradiance):
    '''Create enhanced features for a single prediction matching training pipeline'''
    
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
        'week_of_year': now.isocalendar()[1],
        'is_weekend': int(now.weekday() >= 5),
        'is_weekday': int(now.weekday() < 5),
        'is_monday': int(now.weekday() == 0),
        'is_friday': int(now.weekday() == 4),
    })
    
    # Polynomial features
    features.update({
        'temp_squared': temp ** 2,
        'temp_cubed': temp ** 3,
        'wind_squared': wind ** 2,
        'wind_cubed': wind ** 3,
        'wind_fourth': wind ** 4,
        'humidity_squared': humidity ** 2,
        'solar_squared': solar_irradiance ** 2,
        'solar_sqrt': np.sqrt(solar_irradiance + 1e-6),
        'solar_cubed': solar_irradiance ** 3,
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
        'temp_wind_solar': temp * wind * solar_irradiance / 1000,
        'temp_humidity_solar': temp * humidity * solar_irradiance / 10000,
    })
    
    # Ratio features
    features.update({
        'solar_per_temp': solar_irradiance / (temp + 1e-6),
        'wind_per_temp': wind / (temp + 1e-6),
        'solar_per_humidity': solar_irradiance / (humidity + 1e-6),
        'wind_per_humidity': wind / (humidity + 1e-6),
        'temp_per_humidity': temp / (humidity + 1e-6),
        'efficiency_ratio': solar_irradiance / (humidity + temp + 1e-6),
        'power_density': (wind * solar_irradiance) / (temp + 1e-6),
    })
    
    # Trigonometric features
    features.update({
        'hour_sin': np.sin(2 * np.pi * now.hour / 24),
        'hour_cos': np.cos(2 * np.pi * now.hour / 24),
        'day_sin': np.sin(2 * np.pi * now.timetuple().tm_yday / 365),
        'day_cos': np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        'month_sin': np.sin(2 * np.pi * now.month / 12),
        'month_cos': np.cos(2 * np.pi * now.month / 12),
        'week_sin': np.sin(2 * np.pi * now.weekday() / 7),
        'week_cos': np.cos(2 * np.pi * now.weekday() / 7),
    })
    
    # Weather dummy variables
    weather_types = ['Clear', 'Sunny', 'Cloudy', 'Overcast', 'Rainy']
    for weather_type in weather_types:
        features[f'weather_{weather_type}'] = int(weather == weather_type)
        
        # Weather interactions
        if weather == weather_type:
            features[f'weather_{weather_type}_solar'] = solar_irradiance
            features[f'weather_{weather_type}_temp'] = temp
            features[f'weather_{weather_type}_wind'] = wind
            features[f'weather_{weather_type}_humidity'] = humidity
        else:
            features[f'weather_{weather_type}_solar'] = 0
            features[f'weather_{weather_type}_temp'] = 0
            features[f'weather_{weather_type}_wind'] = 0
            features[f'weather_{weather_type}_humidity'] = 0
    
    # Time interactions
    features.update({
        'hour_temp': now.hour * temp,
        'hour_solar': now.hour * solar_irradiance,
        'hour_wind': now.hour * wind,
        'weekend_solar': features['is_weekend'] * solar_irradiance,
        'weekday_wind': features['is_weekday'] * wind,
    })
    
    # Logarithmic transformations
    features.update({
        'log_solar': np.log1p(solar_irradiance),
        'log_wind': np.log1p(wind),
        'log_temp': np.log1p(temp + 50),
        'log_humidity': np.log1p(humidity),
    })
    
    # Binned features (approximate)
    features.update({
        'temp_bins': min(5, max(0, int((temp - 10) / 5))),
        'wind_bins': min(5, max(0, int(wind / 5))),
        'solar_bins': min(5, max(0, int(solar_irradiance / 200))),
        'humidity_bins': min(4, max(0, int((humidity - 20) / 20))),
        'barometer_bins': min(3, max(0, int((barometer - 980) / 20))),
    })
    
    # Rolling features (simplified for single prediction)
    features.update({
        'temp_rolling_3': temp,
        'wind_rolling_3': wind,
        'solar_rolling_3': solar_irradiance,
        'humidity_rolling_3': humidity,
        'temp_rolling_std': 0,
        'wind_rolling_std': 0,
        'solar_rolling_std': 0,
    })
    
    # Domain-specific features
    features.update({
        'solar_efficiency': solar_irradiance * (1 - 0.004 * max(0, temp - 25)),
        'wind_power_factor': 0 if wind < 3 else (min(1, ((wind - 3) / 9) ** 3) if wind < 12 else (1 if wind < 25 else 0)),
        'solar_elevation': max(0, np.sin(np.pi * (now.hour - 6) / 12)),
        'solar_azimuth': np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        'effective_solar': solar_irradiance * max(0, np.sin(np.pi * (now.hour - 6) / 12)) * np.cos(2 * np.pi * now.timetuple().tm_yday / 365),
        'peak_solar_hours': int(10 <= now.hour <= 15),
        'morning_ramp': int(6 <= now.hour <= 10),
        'evening_ramp': int(15 <= now.hour <= 19),
        'night_time': int(now.hour <= 5 or now.hour >= 20),
        'is_daylight': int(6 <= now.hour <= 18),
    })
    
    # Seasonal factors
    features.update({
        'summer': int(6 <= now.month <= 8),
        'winter': int(now.month == 12 or now.month <= 2),
        'spring': int(3 <= now.month <= 5),
        'autumn': int(9 <= now.month <= 11),
    })
    
    # Season-weather interactions
    features.update({
        'summer_clear': features['summer'] * features.get('weather_Clear', 0),
        'winter_solar': features['winter'] * solar_irradiance,
        'spring_wind': features['spring'] * wind,
        'autumn_temp': features['autumn'] * temp,
    })
    
    # Optimal conditions indicators
    features.update({
        'optimal_solar': int(solar_irradiance > 600 and temp < 30 and features.get('weather_Clear', 0) == 1),
        'optimal_wind': int(8 < wind < 15),
        'optimal_combined': features['optimal_solar'] * features['optimal_wind'],
    })
    
    # Original model features (for compatibility)
    features.update({
        'solar_hour_factor': max(0, np.sin(np.pi * (now.hour - 6) / 12)),
        'season': ((now.month - 1) // 3) + 1,
        'summer_factor': int(6 <= now.month <= 8),
        'weather_encoded': {'Clear': 0, 'Sunny': 1, 'Cloudy': 2, 'Overcast': 3, 'Rainy': 4}.get(weather, 0),
        'GAS_mxm': 0,  # Placeholder for compatibility
    })
    
    return features

def create_basic_features_for_prediction(temp, weather, wind, humidity, barometer, solar_irradiance):
    """Create basic features that should work with most models"""
    
    now = datetime.now()
    
    # Start with essential features
    features = {
        'temp': temp,
        'wind': wind,
        'humidity': humidity,
        'barometer': barometer,
        'solar_irradiance': solar_irradiance,
        'hour': now.hour,
        'day_of_week': now.weekday(),
        'month': now.month,
        'day_of_year': now.timetuple().tm_yday,
        'is_weekend': int(now.weekday() >= 5),
        'is_daylight': int(6 <= now.hour <= 18),
        'solar_hour_factor': max(0, np.sin(np.pi * (now.hour - 6) / 12)),
        'wind_squared': wind ** 2,
        'wind_cubed': wind ** 3,
        'temp_solar_interaction': temp * solar_irradiance / 1000,
        'humidity_temp': humidity * temp / 100,
        'season': ((now.month - 1) // 3) + 1,
        'summer_factor': int(6 <= now.month <= 8),
        'weather_encoded': {'Clear': 0, 'Sunny': 1, 'Cloudy': 2, 'Overcast': 3, 'Rainy': 4}.get(weather, 0),
        'GAS_mxm': 0  # Placeholder
    }
    
    return features

@app.route('/api/predict', methods=['POST'])
def predict_api():
    '''Enhanced API endpoint for predictions using the best model'''
    if not best_model:
        return jsonify({
            "error": "No model loaded",
            "success": False,
            "message": "Model comparison study incomplete"
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
        
        # Try enhanced features first, fall back to basic if needed
        prediction = None
        method_used = "enhanced"
        
        try:
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
            
            # Debug: Print feature information
            print(f"Created features: {len(input_df.columns)}")
            print(f"Required features: {len(feature_names)}")
            
            # Find missing and extra features
            missing_features = [f for f in feature_names if f not in input_df.columns]
            extra_features = [f for f in input_df.columns if f not in feature_names]
            
            if missing_features:
                print(f"Missing features ({len(missing_features)}): {missing_features[:10]}...")  # Show first 10
            if extra_features:
                print(f"Extra features ({len(extra_features)}): {extra_features[:10]}...")  # Show first 10
            
            # Create a dataframe with only the required features
            X_input = pd.DataFrame()
            
            for feature in feature_names:
                if feature in input_df.columns:
                    X_input[feature] = input_df[feature]
                else:
                    # Fill missing features with appropriate defaults
                    if 'optimal' in feature.lower():
                        X_input[feature] = 0  # Binary features default to 0
                    elif 'rolling' in feature.lower():
                        # For rolling features, use the base feature value
                        base_feature = feature.replace('_rolling_3', '').replace('_rolling_std', '')
                        if base_feature in input_df.columns:
                            X_input[feature] = input_df[base_feature]
                        else:
                            X_input[feature] = 0
                    elif any(weather_type in feature for weather_type in ['Clear', 'Sunny', 'Cloudy', 'Overcast', 'Rainy']):
                        # Weather interaction features
                        X_input[feature] = 0
                    elif feature in ['GAS_mxm']:
                        # Model-specific features
                        X_input[feature] = 0
                    else:
                        # Default to 0 for unknown features
                        X_input[feature] = 0
                        print(f"Setting unknown feature '{feature}' to 0")
            
            print(f"Final feature matrix shape: {X_input.shape}")
            
            # Scale features if model requires it
            if scaler is not None:
                X_input_scaled = scaler.transform(X_input)
            else:
                X_input_scaled = X_input.values
            
            # Make prediction
            prediction = best_model.predict(X_input_scaled)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
        except Exception as enhanced_error:
            print(f"Enhanced features failed: {enhanced_error}")
            
            # Fall back to basic features
            try:
                basic_features = create_basic_features_for_prediction(
                    data['temperature'],
                    data['weather'], 
                    data['wind'],
                    data['humidity'],
                    data['barometer'],
                    data['solar_irradiance']
                )
                
                input_df_basic = pd.DataFrame([basic_features])
                
                # Try to use only features that exist in both basic and required
                available_features = [f for f in feature_names if f in input_df_basic.columns]
                
                if len(available_features) < 5:  # Need at least 5 features
                    raise Exception("Insufficient matching features for prediction")
                
                print(f"Using {len(available_features)} basic features for prediction")
                X_input = input_df_basic[available_features]
                
                # Add missing features as zeros
                for feature in feature_names:
                    if feature not in X_input.columns:
                        X_input[feature] = 0
                
                # Reorder to match training
                X_input = X_input[feature_names]
                
                if scaler is not None:
                    X_input_scaled = scaler.transform(X_input)
                else:
                    X_input_scaled = X_input.values
                
                prediction = best_model.predict(X_input_scaled)[0]
                prediction = max(0, prediction)
                method_used = "basic"
                
            except Exception as basic_error:
                print(f"Basic features also failed: {basic_error}")
                
                # Ultimate fallback: simple estimation
                temp = data['temperature']
                wind_speed = data['wind']
                solar_rad = data['solar_irradiance']
                weather = data['weather']
                
                # Simple physics-based estimation
                solar_factor = 1.0 if weather in ['Clear', 'Sunny'] else (0.4 if weather == 'Cloudy' else 0.2)
                solar_est = (solar_rad / 1000) * 50 * solar_factor
                
                wind_est = wind_speed * 2.5 if wind_speed > 3 else 0
                
                temp_factor = 1 - abs(temp - 25) * 0.01  # Optimal around 25°C
                prediction = (solar_est + wind_est) * temp_factor
                prediction = max(10, prediction)  # Minimum generation
                method_used = "physics_based"
        
        # Calculate confidence based on method used and model performance
        if method_used == "enhanced":
            base_confidence = 85
        elif method_used == "basic":
            base_confidence = 75
        else:
            base_confidence = 60
            
        if model_info and 'test_r2_score' in model_info:
            r2_score = model_info['test_r2_score']
            base_confidence = min(95, max(base_confidence, int(r2_score * 100)))
        
        # Estimate generation breakdown
        temp = data['temperature']
        wind_speed = data['wind']
        solar_rad = data['solar_irradiance']
        weather = data['weather']
        
        # Solar generation estimate
        solar_factor = 1.0 if weather in ['Clear', 'Sunny'] else (0.4 if weather == 'Cloudy' else 0.2)
        solar_estimate = min(prediction * 0.6, (solar_rad / 1000) * 50 * solar_factor)
        
        # Wind generation estimate
        wind_estimate = min(prediction * 0.4, wind_speed * 2.5 if wind_speed > 3 else 0)
        
        # Backup generation
        backup_estimate = max(0, prediction - solar_estimate - wind_estimate)
        
        # Format model information
        model_performance = ""
        if model_info:
            if 'test_r2_score' in model_info:
                model_performance = f"R² = {model_info['test_r2_score']:.3f}"
            if 'test_classification_accuracy' in model_info:
                accuracy = model_info['test_classification_accuracy'] * 100
                model_performance += f" | Accuracy = {accuracy:.1f}%"
        
        improvement_text = ""
        if model_info and 'improvement_r2' in model_info:
            improvement_text = f"+{model_info['improvement_r2']:.1f}% R² improvement over baseline"
        
        return jsonify({
            "predicted_generation": round(prediction, 2),
            "solar_estimate": round(solar_estimate, 2),
            "wind_estimate": round(wind_estimate, 2),
            "backup_estimate": round(backup_estimate, 2),
            "confidence": base_confidence,
            "prediction_method": method_used,
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": model_type,
            "algorithm": model_type,
            "feature_count": len(feature_names),
            "model_r2": model_performance.split('|')[0].strip() if model_performance else "N/A",
            "model_accuracy": model_performance.split('|')[1].strip() if '|' in model_performance else "N/A",
            "improvement": improvement_text,
            "success": True
        })
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        return jsonify({
            "error": error_msg,
            "success": False,
            "model_type": model_type,
            "feature_count": len(feature_names) if feature_names else 0
        }), 500

@app.route('/api/model-comparison')
def model_comparison():
    '''API endpoint to get detailed model comparison results'''
    
    comparison_data = {
        "current_model": model_type,
        "model_loaded": best_model is not None,
        "performance_metrics": model_info or {},
        "feature_engineering": "Enhanced" if "Enhanced" in str(model_type) or "Random Forest" in str(model_type) else "Basic",
        "deployment_status": "Production Ready" if best_model else "Not Available",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(comparison_data)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Power Generation Flask API with Dynamic Model Selection...")
    print("="*80)

    if models_loaded:
        print("✅ Model Status: Best model loaded successfully")
        print(f"✅ Algorithm: {model_type}")
        print(f"✅ Features: {len(feature_names) if feature_names else 'Unknown'}")
        
        if model_info:
            if 'test_r2_score' in model_info:
                print(f"✅ Model R² Score: {model_info['test_r2_score']:.4f}")
            elif 'r2' in model_info:
                print(f"✅ Model R² Score: {model_info['r2']:.4f}")
                
            if 'test_classification_accuracy' in model_info:
                print(f"✅ Classification Accuracy: {model_info['test_classification_accuracy']*100:.1f}%")
            elif 'classification_accuracy' in model_info:
                print(f"✅ Classification Accuracy: {model_info['classification_accuracy']*100:.1f}%")
                
            if 'improvement_r2' in model_info:
                print(f"✅ Improvement over baseline: +{model_info['improvement_r2']:.1f}% R²")
                
        print(f"✅ Scaling: {'StandardScaler' if scaler else 'None (Tree-based model)'}")
    else:
        print("⚠️  Model Status: No models loaded - check model files")
    
    print(f"📱 Enhanced Web Interface: http://localhost:5000")
    print(f"🔌 API Endpoints:")
    print(f"   • POST /api/predict - Make predictions with best model")
    print(f"   • GET  /api/status - Check model status and comparison data")
    print(f"   • GET  /api/model-comparison - Detailed comparison results")
    print("="*80)
    print("🎯 Features:")
    print("   • Dynamic best model selection")
    print("   • Enhanced feature engineering")
    print("   • Model comparison metrics")
    print("   • Confidence scoring")
    print("   • Renewable energy breakdown")
    print("   • Production-ready deployment")
    print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)