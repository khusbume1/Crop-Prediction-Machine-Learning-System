# Crop-Prediction-Machine-Learning-System
Crop Prediction Machine Learning System

A comprehensive machine learning system for predicting crop yields using weather data, satellite imagery, and historical crop production data.
🌾 Overview
This project implements multiple ML models (traditional and deep learning) to predict crop yields by integrating:

Weather Data: Temperature, precipitation, humidity, growing degree days
Satellite Data: NDVI, EVI, soil moisture indices
Historical Crop Data: Yield statistics, planting areas, harvest data
Soil Data: Soil type, moisture, nutrient levels
Climate Indices: ENSO, NAO, PDO

📊 Available Data Sources
1. USDA NASS (National Agricultural Statistics Service)

Quick Stats API: Comprehensive crop statistics (1997-present)
Cropland Data Layer (CDL): 30m resolution satellite-based crop classification
VegScape: 250m vegetation condition monitoring
Crop-CASMA: Soil moisture and vegetation analytics
API Key: Required (free) - https://quickstats.nass.usda.gov/api

2. NASA & Satellite Data

MODIS: Vegetation indices (NDVI, EVI) - 250m/500m resolution
Landsat: Multi-spectral imagery - 30m resolution
Sentinel-2: High-resolution imagery - 10m resolution
SMAP: Soil moisture data
Google Earth Engine: Integrated access platform

3. Weather Data

NOAA: Historical weather data
NASA POWER: Solar radiation, temperature, precipitation
WRF-HRRR: High-resolution weather forecasts
Climate Indices: ENSO, NAO, PDO, IOD

4. Research Datasets

CropNet: Terabyte-sized multi-modal dataset (2017-2022)
GlobalCropYield5min: Historical yields (1982-2015) at 5-min resolution
FAO Statistics: Global agricultural data

🏗️ Project Structure
crop_prediction_ml/
├── data/                  # Data storage
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Processed features
│   └── external/         # External datasets
├── src/                  # Source code
│   ├── data_collection/  # Data download scripts
│   ├── preprocessing/    # Data preprocessing
│   ├── features/         # Feature engineering
│   ├── models/          # ML model implementations
│   └── evaluation/      # Model evaluation
├── notebooks/           # Jupyter notebooks for analysis
├── models/             # Saved trained models
├── outputs/            # Predictions, visualizations
├── config/             # Configuration files
└── tests/              # Unit tests
🚀 Quick Start
1. Installation
bashpip install -r requirements.txt
2. Configure API Keys
Edit config/config.yaml:
yamlapi_keys:
  usda_nass: "YOUR_API_KEY_HERE"
  nasa_earthdata: "YOUR_TOKEN_HERE"
3. Download Data
bash# Download USDA crop data
python src/data_collection/usda_collector.py --crop corn --years 2015-2023

# Download satellite data
python src/data_collection/satellite_collector.py --region midwest --years 2020-2023
4. Preprocess Data
bashpython src/preprocessing/preprocess_all.py
5. Train Models
bash# Train all models
python src/models/train_all_models.py

# Train specific model
python src/models/train_model.py --model random_forest --crop corn
6. Make Predictions
bashpython src/models/predict.py --model best_model --year 2024
🤖 Implemented Models
Traditional Machine Learning

Linear Regression: Baseline model
Random Forest: Ensemble tree-based model
XGBoost: Gradient boosting
LightGBM: Fast gradient boosting
Support Vector Regression (SVR): Kernel-based regression

Deep Learning

Multi-Layer Perceptron (MLP): Neural network for tabular data
CNN: For satellite image analysis
LSTM: For time-series weather data
CNN-LSTM Hybrid: Combined spatial and temporal features
Transformer: Attention-based architecture

Ensemble Methods

Stacking: Combines multiple models
Voting: Weighted average of predictions

📈 Model Performance
Based on research literature:

Target Accuracy: 85-95% (R² = 0.70-0.95)
Prediction Window: Up to 3 months before harvest
Best Models: XGBoost, Random Forest, CNN-LSTM

🔧 Key Features
Feature Engineering

Growing Degree Days (GDD)
Cumulative Precipitation
NDVI/EVI time-series statistics
Soil moisture indices
Historical yield trends
Climate oscillation indices

Data Preprocessing

Missing value imputation
Outlier detection and removal
Feature scaling and normalization
Temporal alignment
Spatial aggregation

Model Evaluation

Cross-validation (spatial and temporal)
Multiple metrics (RMSE, MAE, R², MAPE)
Feature importance analysis
Residual analysis
Spatial validation

📚 Research References

CropNet Dataset: Multi-modal dataset with Sentinel-2, WRF-HRRR, and USDA data
EOSDA: Commercial crop yield prediction achieving 95% accuracy
Michigan State Study: Landsat + drought index for sub-field predictions
Global Yield Mapping: ML models for maize, rice, wheat, soybean (1982-2015)

🌍 Example Use Cases

Farm Management: Field-level yield forecasting
Food Security: National/regional production estimates
Insurance: Crop loss assessment
Market Analysis: Price forecasting
Climate Research: Impact assessment

📊 Sample Results
python# Example: Corn yield prediction for 2023
Model: XGBoost
R²: 0.89
RMSE: 12.3 bushels/acre
MAE: 9.8 bushels/acre
Prediction Window: 2 months before harvest
🔬 Advanced Features

Multi-crop support: Corn, soybeans, wheat, rice, cotton
Multi-scale predictions: Field, county, state, national
Real-time updates: Integration with live weather feeds
Uncertainty quantification: Prediction intervals
Explainable AI: SHAP values for feature interpretation

🤝 Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
📝 License
MIT License - See LICENSE file for details
📧 Contact
For questions or collaborations, please open an issue on GitHub.
🙏 Acknowledgments

USDA NASS for agricultural data
NASA for satellite imagery
Research community for datasets and methodologies
