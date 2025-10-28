"""
Weather Data Collector
Collects historical weather data from multiple sources
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDataCollector:
    """Collect weather data from various sources"""
    
    # NASA POWER API (free, no API key needed for basic access)
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self):
        """Initialize weather data collector"""
        self.session = requests.Session()
    
    def get_nasa_power_data(self,
                           latitude: float,
                           longitude: float,
                           start_date: str,
                           end_date: str,
                           parameters: List[str] = None) -> pd.DataFrame:
        """
        Get weather data from NASA POWER API
        
        Args:
            latitude: Latitude
            longitude: Longitude
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            parameters: List of weather parameters
            
        Returns:
            DataFrame with weather data
        """
        if parameters is None:
            parameters = [
                'T2M',          # Temperature at 2m
                'T2M_MAX',      # Maximum temperature
                'T2M_MIN',      # Minimum temperature
                'PRECTOTCORR',  # Precipitation
                'RH2M',         # Relative humidity
                'WS2M',         # Wind speed at 2m
                'ALLSKY_SFC_SW_DWN'  # Solar radiation
            ]
        
        params = {
            'parameters': ','.join(parameters),
            'community': 'AG',
            'longitude': longitude,
            'latitude': latitude,
            'start': start_date,
            'end': end_date,
            'format': 'JSON'
        }
        
        try:
            response = self.session.get(self.NASA_POWER_URL, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract parameter data
            records = []
            if 'properties' in data and 'parameter' in data['properties']:
                param_data = data['properties']['parameter']
                
                # Get dates
                dates = list(param_data[parameters[0]].keys())
                
                for date in dates:
                    record = {'date': date}
                    for param in parameters:
                        if param in param_data:
                            record[param] = param_data[param].get(date)
                    records.append(record)
                
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                
                # Rename columns to be more readable
                column_mapping = {
                    'T2M': 'temp_mean',
                    'T2M_MAX': 'temp_max',
                    'T2M_MIN': 'temp_min',
                    'PRECTOTCORR': 'precipitation',
                    'RH2M': 'humidity',
                    'WS2M': 'wind_speed',
                    'ALLSKY_SFC_SW_DWN': 'solar_radiation'
                }
                df = df.rename(columns=column_mapping)
                
                return df
            else:
                logger.warning("No data in response")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch NASA POWER data: {e}")
            return pd.DataFrame()
    
    def calculate_growing_degree_days(self,
                                     temp_max: pd.Series,
                                     temp_min: pd.Series,
                                     base_temp: float = 10.0,
                                     max_temp: float = 30.0) -> pd.Series:
        """
        Calculate Growing Degree Days (GDD)
        
        GDD = max(0, ((Tmax + Tmin)/2 - Tbase))
        With upper limit at max_temp
        
        Args:
            temp_max: Maximum daily temperature (Celsius)
            temp_min: Minimum daily temperature (Celsius)
            base_temp: Base temperature (default 10°C for corn)
            max_temp: Maximum effective temperature (default 30°C)
            
        Returns:
            Series with GDD values
        """
        # Cap temperatures
        temp_max_capped = temp_max.clip(upper=max_temp)
        temp_min_capped = temp_min.clip(upper=max_temp)
        
        # Calculate average temperature
        temp_avg = (temp_max_capped + temp_min_capped) / 2
        
        # Calculate GDD
        gdd = (temp_avg - base_temp).clip(lower=0)
        
        return gdd
    
    def calculate_cumulative_precipitation(self, precipitation: pd.Series) -> pd.Series:
        """Calculate cumulative precipitation"""
        return precipitation.cumsum()
    
    def calculate_drought_index(self,
                                precipitation: pd.Series,
                                temp_mean: pd.Series,
                                window: int = 30) -> pd.Series:
        """
        Calculate simple drought index
        Based on precipitation deficit and temperature
        
        Args:
            precipitation: Daily precipitation
            temp_mean: Mean daily temperature
            window: Rolling window in days
            
        Returns:
            Drought index (higher = more drought stress)
        """
        # Rolling precipitation
        precip_rolling = precipitation.rolling(window=window, min_periods=1).sum()
        
        # Rolling temperature
        temp_rolling = temp_mean.rolling(window=window, min_periods=1).mean()
        
        # Simple drought index: high temp + low precip = drought
        # Normalize to 0-1 scale
        drought_index = (temp_rolling / temp_rolling.max()) - (precip_rolling / precip_rolling.max())
        drought_index = (drought_index - drought_index.min()) / (drought_index.max() - drought_index.min())
        
        return drought_index
    
    def calculate_heat_stress_index(self,
                                    temp_max: pd.Series,
                                    humidity: pd.Series,
                                    threshold: float = 30.0) -> pd.Series:
        """
        Calculate heat stress index
        Days with high temperature and humidity
        
        Args:
            temp_max: Maximum daily temperature
            humidity: Relative humidity (%)
            threshold: Temperature threshold for heat stress
            
        Returns:
            Heat stress index
        """
        # Days above threshold
        heat_days = (temp_max > threshold).astype(int)
        
        # Weighted by humidity (higher humidity = more stress)
        heat_stress = heat_days * (humidity / 100)
        
        return heat_stress
    
    def get_climate_indices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get climate oscillation indices (ENSO, NAO, PDO, etc.)
        
        Note: This is a placeholder. In practice, you'd download from:
        - NOAA Climate Prediction Center
        - https://www.cpc.ncep.noaa.gov/data/indices/
        """
        # Example: Create synthetic data
        # In production, fetch from NOAA or other sources
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        df = pd.DataFrame({
            'date': date_range,
            'nino34': np.random.randn(len(date_range)),  # Replace with real data
            'nao': np.random.randn(len(date_range)),
            'pdo': np.random.randn(len(date_range)),
            'iod': np.random.randn(len(date_range))
        })
        
        logger.warning("Using synthetic climate indices. Replace with real data from NOAA.")
        
        return df
    
    def aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily weather data to monthly"""
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Aggregation functions
        agg_functions = {
            'temp_mean': 'mean',
            'temp_max': 'max',
            'temp_min': 'min',
            'precipitation': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'solar_radiation': 'mean'
        }
        
        # Only aggregate columns that exist
        agg_functions = {k: v for k, v in agg_functions.items() if k in df.columns}
        
        monthly = df.groupby(['year', 'month']).agg(agg_functions).reset_index()
        
        return monthly
    
    def aggregate_to_growing_season(self,
                                    df: pd.DataFrame,
                                    growing_months: List[int] = [4, 5, 6, 7, 8, 9]) -> pd.DataFrame:
        """
        Aggregate to growing season
        
        Args:
            df: Daily weather data
            growing_months: Months in growing season (default Apr-Sep)
        """
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Filter to growing season
        df_growing = df[df['month'].isin(growing_months)]
        
        # Aggregate by year
        agg_functions = {
            'temp_mean': 'mean',
            'temp_max': 'max',
            'temp_min': 'min',
            'precipitation': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'solar_radiation': 'mean'
        }
        
        agg_functions = {k: v for k, v in agg_functions.items() if k in df.columns}
        
        seasonal = df_growing.groupby('year').agg(agg_functions).reset_index()
        
        return seasonal
    
    def collect_county_weather(self,
                               latitude: float,
                               longitude: float,
                               years: List[int],
                               output_dir: str = "data/raw/weather") -> pd.DataFrame:
        """
        Collect comprehensive weather data for a location
        
        Args:
            latitude: Latitude of location (county centroid)
            longitude: Longitude of location
            years: List of years
            output_dir: Output directory
            
        Returns:
            DataFrame with all weather data
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_data = []
        
        for year in years:
            logger.info(f"Collecting weather data for {year}")
            
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            df_year = self.get_nasa_power_data(latitude, longitude, start_date, end_date)
            
            if not df.empty:
                # Calculate derived features
                if 'temp_max' in df_year.columns and 'temp_min' in df_year.columns:
                    df_year['gdd'] = self.calculate_growing_degree_days(
                        df_year['temp_max'],
                        df_year['temp_min']
                    )
                    df_year['gdd_cumulative'] = df_year['gdd'].cumsum()
                
                if 'precipitation' in df_year.columns:
                    df_year['precipitation_cumulative'] = self.calculate_cumulative_precipitation(
                        df_year['precipitation']
                    )
                
                if 'precipitation' in df_year.columns and 'temp_mean' in df_year.columns:
                    df_year['drought_index'] = self.calculate_drought_index(
                        df_year['precipitation'],
                        df_year['temp_mean']
                    )
                
                if 'temp_max' in df_year.columns and 'humidity' in df_year.columns:
                    df_year['heat_stress'] = self.calculate_heat_stress_index(
                        df_year['temp_max'],
                        df_year['humidity']
                    )
                
                all_data.append(df_year)
            
            time.sleep(1)  # Rate limiting
        
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            
            # Save full daily data
            filepath = f"{output_dir}/weather_daily_{int(latitude)}_{int(longitude)}.csv"
            df_all.to_csv(filepath, index=False)
            logger.info(f"Saved daily weather data: {len(df_all)} records")
            
            # Save monthly aggregation
            df_monthly = self.aggregate_to_monthly(df_all)
            filepath_monthly = f"{output_dir}/weather_monthly_{int(latitude)}_{int(longitude)}.csv"
            df_monthly.to_csv(filepath_monthly, index=False)
            logger.info(f"Saved monthly weather data: {len(df_monthly)} records")
            
            # Save growing season aggregation
            df_seasonal = self.aggregate_to_growing_season(df_all)
            filepath_seasonal = f"{output_dir}/weather_seasonal_{int(latitude)}_{int(longitude)}.csv"
            df_seasonal.to_csv(filepath_seasonal, index=False)
            logger.info(f"Saved seasonal weather data: {len(df_seasonal)} records")
            
            return df_all
        else:
            return pd.DataFrame()


def main():
    """Example usage"""
    collector = WeatherDataCollector()
    
    # Example: Story County, Iowa (approximate centroid)
    latitude = 42.03
    longitude = -93.62
    years = [2020, 2021, 2022, 2023]
    
    df = collector.collect_county_weather(
        latitude=latitude,
        longitude=longitude,
        years=years,
        output_dir='data/raw/weather'
    )
    
    logger.info("\n" + "="*50)
    logger.info("Weather Data Collection Summary")
    logger.info("="*50)
    logger.info(f"Total records: {len(df)}")
    if not df.empty:
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
