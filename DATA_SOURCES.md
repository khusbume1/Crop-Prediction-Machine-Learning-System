# 📊 Data Sources for Crop Yield Prediction

Comprehensive guide to available data sources for building crop prediction models.

## 🌾 Crop Production Data

### 1. USDA NASS (National Agricultural Statistics Service)

**Source**: https://www.nass.usda.gov/

**Coverage**: United States (1866-present)

**Data Types**:
- County-level crop yields
- Planted and harvested acreage
- Production volumes
- Crop progress reports (weekly)
- Crop condition ratings
- Prices and values

**API Access**:
- Quick Stats API: https://quickstats.nass.usda.gov/api
- Free API key (instant)
- Rate limit: Generous for research use
- Format: JSON, CSV

**Crops Available**:
- Major crops: Corn, Soybeans, Wheat, Cotton, Rice
- Specialty crops: Vegetables, fruits, nuts
- 100+ commodity types

**Spatial Resolution**: County, State, National

**How to Use**:
```python
from src.data_collection.usda_collector import USDADataCollector

collector = USDADataCollector(api_key='YOUR_KEY')
data = collector.get_crop_yields(
    crop='CORN',
    states=['IOWA', 'ILLINOIS'],
    years=[2015, 2016, 2017, 2018, 2019, 2020]
)
```

### 2. FAO Statistics (FAOSTAT)

**Source**: https://www.fao.org/faostat/

**Coverage**: Global (1961-present)

**Data Types**:
- Crop production by country
- Area harvested
- Yield (tons/hectare)
- Trade data
- Food balance sheets

**Access**: Web interface, bulk download

**Best For**: International comparisons, global analysis

### 3. USDA FAS (Foreign Agricultural Service)

**Source**: https://apps.fas.usda.gov/psdonline/

**Coverage**: Global production and supply

**Data Types**:
- Production forecasts
- Supply and demand
- Trade data
- Market analysis

## 🛰️ Satellite Data

### 1. MODIS (NASA)

**Source**: https://modis.gsfc.nasa.gov/

**Coverage**: Global (2000-present)

**Products**:
- **MOD13Q1**: Vegetation Indices (NDVI, EVI) - 250m, 16-day
- **MOD09A1**: Surface Reflectance - 500m, 8-day
- **MCD12Q1**: Land Cover - 500m, annual

**Access**: 
- Google Earth Engine
- USGS EarthExplorer
- NASA Earthdata

**Temporal Resolution**: Daily to 16-day composites

**Cost**: Free

### 2. Landsat (USGS)

**Source**: https://landsat.gsfc.nasa.gov/

**Coverage**: Global (1972-present)

**Products**:
- **Landsat 8/9**: OLI sensor, 11 bands
- **Landsat 7**: ETM+ sensor

**Spatial Resolution**: 30m (visible/NIR), 100m (thermal)

**Temporal Resolution**: 16-day revisit

**Access**:
- Google Earth Engine
- USGS EarthExplorer
- AWS Open Data

**Cost**: Free

### 3. Sentinel-2 (ESA)

**Source**: https://sentinel.esa.int/

**Coverage**: Global (2015-present)

**Products**:
- **Sentinel-2 Level-2A**: Surface reflectance
- 13 spectral bands

**Spatial Resolution**: 10m (RGB, NIR), 20m (red edge), 60m (atmospheric)

**Temporal Resolution**: 5-day revisit (both satellites)

**Access**:
- Google Earth Engine
- Copernicus Open Access Hub
- AWS Open Data

**Cost**: Free

**Why Sentinel-2?**
- Highest resolution free satellite data (10m)
- Frequent revisits
- Red edge bands for agriculture
- Excellent for field-level analysis

### 4. USDA Cropland Data Layer (CDL)

**Source**: https://nassgeodata.gmu.edu/CropScape/

**Coverage**: Contiguous United States (1997-present)

**Data**: Crop-specific land cover classification

**Resolution**: 30m

**Access**: CropScape portal, Google Earth Engine

**Use Case**: Identify crop types, field boundaries

## ☁️ Weather Data

### 1. NASA POWER

**Source**: https://power.larc.nasa.gov/

**Coverage**: Global (1981-present)

**Variables**:
- Temperature (min, max, mean)
- Precipitation
- Solar radiation
- Humidity
- Wind speed
- Dew point

**Resolution**: 0.5° x 0.625° (~50km)

**Access**: API (no key required)

**Cost**: Free

**API Example**:
```python
from src.data_collection.weather_collector import WeatherDataCollector

collector = WeatherDataCollector()
data = collector.get_nasa_power_data(
    latitude=42.03,
    longitude=-93.62,
    start_date='20200101',
    end_date='20201231'
)
```

### 2. NOAA Climate Data

**Source**: https://www.ncdc.noaa.gov/cdo-web/

**Coverage**: United States, high-quality stations

**Data Types**:
- **GHCN-Daily**: Daily summaries
- **Hourly**: Hourly observations
- **Normals**: 30-year averages

**Access**: API (free token required)

**Resolution**: Station-based (point data)

### 3. PRISM Climate Group

**Source**: https://prism.oregonstate.edu/

**Coverage**: Contiguous United States (1895-present)

**Variables**:
- Precipitation
- Temperature
- Dew point
- Vapor pressure

**Resolution**: 4km grid

**Quality**: High-quality, interpolated from stations

**Access**: Web download, FTP

**Cost**: Free for academic use

### 4. European Climate Data (Copernicus)

**Source**: https://climate.copernicus.eu/

**Coverage**: Global

**Products**:
- ERA5: Reanalysis data (1940-present)
- Seasonal forecasts
- Climate projections

## 💧 Soil Data

### 1. NASA SMAP (Soil Moisture)

**Source**: https://smap.jpl.nasa.gov/

**Coverage**: Global (2015-present)

**Data**: Surface and root-zone soil moisture

**Resolution**: 9km, 36km

**Access**: NASA Earthdata, Google Earth Engine

**Temporal**: Daily, 2-3 day revisit

### 2. USDA SSURGO (Soil Survey)

**Source**: https://websoilsurvey.nrcs.usda.gov/

**Coverage**: United States, county-level detail

**Data**:
- Soil type classifications
- Organic matter content
- pH levels
- Drainage characteristics
- Water holding capacity

**Resolution**: 1:24,000 scale (very detailed)

**Access**: Web Soil Survey, NRCS Geospatial Gateway

### 3. SoilGrids (Global)

**Source**: https://soilgrids.org/

**Coverage**: Global

**Data**:
- Soil organic carbon
- pH
- Texture (clay, silt, sand)
- Bulk density
- Depth to bedrock

**Resolution**: 250m

**Access**: REST API, WCS service

## 🌡️ Climate Indices

### 1. NOAA Climate Indices

**Source**: https://www.cpc.ncep.noaa.gov/data/indices/

**Indices Available**:
- **ENSO (El Niño/La Niña)**: Niño 3.4 SST Index
- **NAO**: North Atlantic Oscillation
- **PDO**: Pacific Decadal Oscillation
- **IOD**: Indian Ocean Dipole
- **AO**: Arctic Oscillation

**Coverage**: 1950-present (varies by index)

**Temporal**: Monthly

**Impact**: These affect regional weather patterns and crop yields

**Access**: Direct download (TXT files)

## 🔬 Research Datasets

### 1. CropNet Dataset

**Source**: https://github.com/fudong03/CropNet

**Coverage**: United States, 2017-2022, 2200+ counties

**Data Included**:
- Sentinel-2 imagery (10m)
- WRF-HRRR weather data
- USDA crop statistics

**Size**: Terabyte-scale

**Format**: PyTorch-ready

**Use Case**: Deep learning crop prediction

### 2. GlobalCropYield5min

**Source**: https://doi.org/10.1038/s41597-025-04650-4

**Coverage**: Global, 1982-2015

**Crops**: Maize, Rice, Wheat, Soybean

**Resolution**: 5 arc-minutes (~10km)

**Data**: Historical crop yields with climate data

### 3. AgML Datasets

**Source**: https://github.com/Project-AgML/AgML

**Content**: Curated agricultural ML datasets

**Includes**:
- Computer vision datasets
- Crop yield datasets
- Plant disease datasets
- Weed detection datasets

## 📦 Ready-to-Use Packages

### 1. Python Packages

```bash
# USDA NASS data
pip install rnassqs

# Earth Engine
pip install earthengine-api

# NASA data
pip install pynasa

# Climate data
pip install xclim
```

### 2. R Packages

```r
# USDA NASS
install.packages("rnassqs")

# Weather data
install.packages("rnoaa")

# Crop data
install.packages("FAOSTAT")
```

## 🎯 Recommended Data Combinations

### For County-Level Corn Prediction (USA):

**Essential**:
1. USDA NASS crop yields (target)
2. NASA POWER weather data
3. MODIS NDVI (satellite)

**Recommended Add-ons**:
4. Sentinel-2 for high-resolution
5. SSURGO soil data
6. Climate indices (ENSO)

**Expected Accuracy**: R² = 0.85-0.92

### For Field-Level Prediction:

**Essential**:
1. Field yield monitors (if available)
2. Sentinel-2 imagery (10m)
3. Weather station data

**Recommended**:
4. SMAP soil moisture
5. Digital elevation model
6. Management data (fertilizer, irrigation)

**Expected Accuracy**: R² = 0.88-0.95

### For Global Crop Assessment:

**Essential**:
1. FAO crop statistics
2. MODIS time-series
3. Climate reanalysis (ERA5)

**Recommended**:
4. CropNet dataset
5. Climate projections
6. Trade data

## 💰 Cost Summary

| Data Source | Cost | API Key Required | Rate Limits |
|-------------|------|------------------|-------------|
| USDA NASS | Free | Yes (free) | Generous |
| NASA POWER | Free | No | None |
| Google Earth Engine | Free* | Yes (free) | 10K req/day |
| Landsat | Free | No | None |
| Sentinel-2 | Free | No | None |
| MODIS | Free | No | None |
| NOAA Climate | Free | Yes (free) | 10K req/day |
| PRISM | Free* | No | Academic use |

*Free for research and education

## 📝 Data Collection Checklist

### Before Starting:

- [ ] Identify target crop(s)
- [ ] Define geographic scope
- [ ] Determine time period needed
- [ ] Get required API keys
- [ ] Set up Google Earth Engine account
- [ ] Install necessary packages

### Data Collection Order:

1. **Crop Data** (USDA NASS) - Target variable
2. **Weather Data** (NASA POWER) - Key features
3. **Satellite Data** (MODIS/Sentinel-2) - Vegetation monitoring
4. **Soil Data** (SSURGO) - Static features
5. **Climate Indices** (NOAA) - Large-scale patterns

### Quality Checks:

- [ ] No missing data gaps > 10%
- [ ] Temporal alignment (same time periods)
- [ ] Spatial matching (correct locations)
- [ ] Units verified and consistent
- [ ] Outliers identified and handled

## 🔗 Useful Links

### Official Documentation:
- USDA NASS API: https://quickstats.nass.usda.gov/api
- Google Earth Engine: https://developers.google.com/earth-engine
- NASA Earthdata: https://earthdata.nasa.gov/
- Copernicus Open Access: https://scihub.copernicus.eu/

### Tutorials:
- Earth Engine: https://developers.google.com/earth-engine/tutorials
- Crop Monitoring: https://eos.com/blog/crop-monitoring/
- MODIS Data: https://lpdaac.usgs.gov/data/get-started-data/

### Research Papers:
- CropNet: https://openreview.net/forum?id=lzpHNyhIbr
- Machine Learning in Agriculture: https://doi.org/10.1016/j.compag.2021.106343

## 📊 Data Update Frequency

| Data Type | Update Frequency | Latency |
|-----------|-----------------|---------|
| USDA Yields | Annual | 3-6 months after harvest |
| USDA Progress | Weekly | 1 week |
| MODIS | 16-day composite | 2-3 days |
| Sentinel-2 | 5-day revisit | 1-2 days |
| Landsat | 16-day revisit | 1-2 days |
| NASA POWER | Daily | 1 day |
| SMAP | Daily | 2-3 days |

## 🎓 Academic Resources

### Datasets for Research:
1. **Zenodo**: https://zenodo.org/ (search "crop yield")
2. **Kaggle**: https://www.kaggle.com/datasets (agricultural datasets)
3. **IEEE DataPort**: https://ieee-dataport.org/
4. **Harvard Dataverse**: https://dataverse.harvard.edu/

### Benchmark Datasets:
- **AgML**: Standardized agricultural ML datasets
- **CropNet**: Large-scale multi-modal dataset
- **Radiant MLHub**: Earth observation ML datasets

---

**Note**: Always cite data sources in publications and respect usage terms and licenses.
