# 🚀 Quick Start Guide - Crop Yield Prediction ML System

This guide will help you get started with the crop yield prediction system in under 15 minutes.

## Prerequisites

- Python 3.8+
- pip package manager
- (Optional) Google Earth Engine account for satellite data
- (Optional) USDA NASS API key for crop data

## Step 1: Installation (5 minutes)

### 1.1 Clone/Download the Project

```bash
cd crop_prediction_ml
```

### 1.2 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- scikit-learn, xgboost, lightgbm (ML models)
- tensorflow (deep learning)
- pandas, numpy (data processing)
- earthengine-api (satellite data)
- matplotlib, seaborn (visualization)

## Step 2: Configuration (2 minutes)

### 2.1 Get API Keys (Optional but Recommended)

**USDA NASS API Key** (Free):
1. Visit https://quickstats.nass.usda.gov/api
2. Request an API key (instant)
3. Check your email for the key

**Google Earth Engine** (Free for research):
1. Sign up at https://earthengine.google.com/
2. Run `earthengine authenticate` in terminal
3. Follow the authentication flow

### 2.2 Configure Settings

```bash
# Copy template configuration
cp config/config_template.yaml config/config.yaml

# Edit with your API keys (optional)
nano config/config.yaml
```

Add your API keys:
```yaml
api_keys:
  usda_nass: "YOUR_API_KEY_HERE"
  nasa_earthdata: "YOUR_TOKEN_HERE"
```

## Step 3: Quick Demo with Synthetic Data (3 minutes)

### 3.1 Run Training Pipeline

```bash
# Train models with synthetic data
python src/train_pipeline.py --synthetic --output outputs
```

This will:
- Generate synthetic crop yield data
- Train 6 ML models (Linear Regression, Random Forest, XGBoost, LightGBM, SVR, MLP)
- Evaluate and compare models
- Save best models to `outputs/models/`
- Save results to `outputs/results_*.json`

Expected output:
```
Training Linear Regression...
Training Random Forest...
Training XGBoost...
Training LightGBM...
Training SVR...
Training MLP (Neural Network)...

Model Comparison
================
Best Model: xgboost
Best R²: 0.8543
Best RMSE: 12.34
```

### 3.2 View Results in Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/01_Crop_Yield_Analysis.ipynb
```

Run all cells to see:
- Data exploration visualizations
- Feature correlation analysis
- Model performance comparison
- Prediction visualizations

## Step 4: Use Real Data (Optional)

### 4.1 Collect USDA Crop Data

```bash
python src/data_collection/usda_collector.py
```

This will download:
- Crop yield data by county
- Planted acreage
- Production statistics
- Crop progress reports

Data saved to: `data/raw/usda/`

### 4.2 Collect Satellite Data

```bash
python src/data_collection/satellite_collector.py
```

This collects:
- MODIS NDVI/EVI (250m resolution)
- Sentinel-2 NDVI (10m resolution)
- Landsat imagery (30m resolution)
- Soil moisture from SMAP

Data saved to: `data/raw/satellite/`

### 4.3 Collect Weather Data

```bash
python src/data_collection/weather_collector.py
```

This fetches:
- Daily temperature (min, max, mean)
- Precipitation
- Humidity
- Solar radiation
- Growing Degree Days
- Drought indices

Data saved to: `data/raw/weather/`

### 4.4 Train with Real Data

```bash
# Train models with collected data
python src/train_pipeline.py --output outputs
```

## Step 5: Make Predictions (2 minutes)

### 5.1 Load Trained Model

```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('outputs/models/xgboost.pkl')
scaler = joblib.load('outputs/models/scaler_standard.pkl')

# Load your data
new_data = pd.read_csv('your_data.csv')

# Prepare features (same features used in training)
X_new = new_data[feature_columns]
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)

print(f"Predicted yield: {predictions[0]:.2f} bushels/acre")
```

### 5.2 Batch Predictions

```python
# Predict for multiple locations/years
predictions_df = pd.DataFrame({
    'county': new_data['county'],
    'year': new_data['year'],
    'predicted_yield': predictions
})

predictions_df.to_csv('outputs/predictions.csv', index=False)
```

## Common Use Cases

### Use Case 1: County-Level Yield Forecasting

```python
from src.data_collection.usda_collector import USDADataCollector
from src.data_collection.weather_collector import WeatherDataCollector

# Collect data for specific county
usda = USDADataCollector(api_key='YOUR_KEY')
weather = WeatherDataCollector()

# Get historical yields
yields = usda.get_crop_yields(
    crop='CORN',
    states=['IOWA'],
    years=[2020, 2021, 2022]
)

# Get weather data
weather_data = weather.collect_county_weather(
    latitude=42.03,  # Story County, IA
    longitude=-93.62,
    years=[2020, 2021, 2022]
)

# Train model and predict 2024 yield
```

### Use Case 2: Multi-Crop Comparison

```python
crops = ['CORN', 'SOYBEANS', 'WHEAT']
results = {}

for crop in crops:
    # Train separate model for each crop
    model = train_crop_model(crop)
    results[crop] = model.evaluate()

# Compare crop performance
print(pd.DataFrame(results))
```

### Use Case 3: Regional Analysis

```python
states = ['IOWA', 'ILLINOIS', 'NEBRASKA', 'KANSAS']

for state in states:
    # Collect and analyze data by state
    state_data = collect_state_data(state)
    model = train_model(state_data)
    predictions = model.predict_2024()
    
    # Generate state report
    create_report(state, predictions)
```

## Project Structure Overview

```
crop_prediction_ml/
├── data/                    # Data storage
│   ├── raw/                # Downloaded data
│   │   ├── usda/          # USDA crop data
│   │   ├── satellite/     # Satellite imagery
│   │   └── weather/       # Weather data
│   └── processed/         # Processed features
│
├── src/                    # Source code
│   ├── data_collection/   # Data collectors
│   │   ├── usda_collector.py
│   │   ├── satellite_collector.py
│   │   └── weather_collector.py
│   ├── models/            # ML models
│   │   └── ml_models.py
│   └── train_pipeline.py  # Training pipeline
│
├── notebooks/             # Jupyter notebooks
│   └── 01_Crop_Yield_Analysis.ipynb
│
├── models/               # Saved models
├── outputs/             # Results and predictions
├── config/              # Configuration files
└── tests/              # Unit tests
```

## Tips for Best Results

### 1. Data Quality
- Use at least 5 years of historical data
- Ensure data covers full growing season
- Include multiple weather stations for accuracy
- Verify satellite data has low cloud cover

### 2. Feature Engineering
- Calculate growing degree days (GDD)
- Include cumulative precipitation
- Add vegetation index time-series statistics
- Consider lagged features (previous year yields)

### 3. Model Selection
- Start with Random Forest (robust, interpretable)
- Try XGBoost for best accuracy
- Use ensemble methods for production
- Neural networks work well with large datasets

### 4. Validation
- Use time-based cross-validation (not random)
- Validate on out-of-sample years
- Check spatial consistency
- Monitor prediction intervals

## Troubleshooting

### Issue: API Key Not Working

```bash
# Verify API key
python -c "from src.data_collection.usda_collector import USDADataCollector; c = USDADataCollector('YOUR_KEY'); print('Success!')"
```

### Issue: Earth Engine Authentication Failed

```bash
# Re-authenticate
earthengine authenticate
```

### Issue: Out of Memory

```python
# Use batch processing
data = load_data_in_chunks(batch_size=1000)

# Or reduce features
features = select_top_features(X, n=20)
```

### Issue: Poor Model Performance

Checklist:
- [ ] Sufficient training data (>500 samples)
- [ ] Features properly scaled
- [ ] No data leakage (future info in training)
- [ ] Outliers handled
- [ ] Missing values imputed correctly

## Next Steps

1. ✅ Complete quick demo with synthetic data
2. 📊 Explore analysis notebook
3. 🔑 Get API keys for real data
4. 📡 Collect historical data for your region
5. 🤖 Train models with real data
6. 📈 Make predictions for current season
7. 📋 Generate yield reports

## Getting Help

- **Documentation**: See README.md for detailed info
- **Examples**: Check notebooks/ for more examples
- **Issues**: Create GitHub issue for bugs
- **Data Sources**: See DATA_SOURCES.md for data options

## Performance Benchmarks

Based on research literature, you can expect:

| Model | R² Score | RMSE | Training Time |
|-------|----------|------|---------------|
| Linear Regression | 0.65-0.75 | 15-20 | < 1 second |
| Random Forest | 0.80-0.88 | 10-15 | 10-30 seconds |
| XGBoost | 0.85-0.92 | 8-12 | 30-60 seconds |
| LightGBM | 0.85-0.91 | 8-13 | 20-40 seconds |
| Neural Network | 0.82-0.90 | 9-14 | 2-5 minutes |

*Note: Performance varies based on data quality and quantity*

## Success! 🎉

You're now ready to:
- Predict crop yields accurately
- Analyze agricultural trends
- Support farm management decisions
- Assess climate impacts
- Optimize resource allocation

Happy predicting! 🌾
