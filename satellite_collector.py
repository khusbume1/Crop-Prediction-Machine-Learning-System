"""
Satellite Data Collector
Collects vegetation indices and satellite imagery from multiple sources
Uses Google Earth Engine for efficient data access
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDataCollector:
    """Collect satellite data from various sources via Google Earth Engine"""
    
    def __init__(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.info("Run 'earthengine authenticate' first")
            raise
    
    def get_modis_ndvi(self,
                       region: ee.Geometry,
                       start_date: str,
                       end_date: str,
                       scale: int = 250) -> pd.DataFrame:
        """
        Get MODIS NDVI time series
        
        Args:
            region: Earth Engine Geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            scale: Resolution in meters (default 250m)
            
        Returns:
            DataFrame with NDVI time series
        """
        # MODIS Terra Vegetation Indices 16-Day Global 250m
        modis = ee.ImageCollection('MODIS/006/MOD13Q1') \
            .filterDate(start_date, end_date) \
            .filterBounds(region)
        
        def extract_ndvi(image):
            # NDVI band (scaled by 10000)
            ndvi = image.select('NDVI').divide(10000)
            
            # Calculate mean NDVI over region
            stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'NDVI': stats.get('NDVI')
            })
        
        # Extract time series
        features = modis.map(extract_ndvi).getInfo()
        
        # Convert to DataFrame
        data = []
        for feature in features['features']:
            props = feature['properties']
            if props.get('NDVI') is not None:
                data.append({
                    'date': props['date'],
                    'NDVI': props['NDVI']
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_modis_evi(self,
                      region: ee.Geometry,
                      start_date: str,
                      end_date: str,
                      scale: int = 250) -> pd.DataFrame:
        """Get MODIS EVI (Enhanced Vegetation Index) time series"""
        modis = ee.ImageCollection('MODIS/006/MOD13Q1') \
            .filterDate(start_date, end_date) \
            .filterBounds(region)
        
        def extract_evi(image):
            evi = image.select('EVI').divide(10000)
            
            stats = evi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'EVI': stats.get('EVI')
            })
        
        features = modis.map(extract_evi).getInfo()
        
        data = []
        for feature in features['features']:
            props = feature['properties']
            if props.get('EVI') is not None:
                data.append({
                    'date': props['date'],
                    'EVI': props['EVI']
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_landsat_ndvi(self,
                        region: ee.Geometry,
                        start_date: str,
                        end_date: str,
                        cloud_cover_max: int = 20) -> pd.DataFrame:
        """
        Get Landsat 8/9 NDVI time series
        
        Args:
            region: Earth Engine Geometry
            start_date: Start date
            end_date: End date
            cloud_cover_max: Maximum cloud cover percentage
        """
        # Landsat 8/9 Collection 2 Tier 1 TOA Reflectance
        landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA') \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
        
        def calculate_ndvi(image):
            # NDVI = (NIR - Red) / (NIR + Red)
            ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
            
            stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'NDVI': stats.get('NDVI'),
                'cloud_cover': image.get('CLOUD_COVER')
            })
        
        features = landsat.map(calculate_ndvi).getInfo()
        
        data = []
        for feature in features['features']:
            props = feature['properties']
            if props.get('NDVI') is not None:
                data.append(props)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_sentinel2_ndvi(self,
                          region: ee.Geometry,
                          start_date: str,
                          end_date: str,
                          cloud_cover_max: int = 20) -> pd.DataFrame:
        """
        Get Sentinel-2 NDVI time series (10m resolution)
        """
        sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
        
        def calculate_ndvi(image):
            # NDVI = (NIR - Red) / (NIR + Red)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
            stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'NDVI': stats.get('NDVI'),
                'cloud_cover': image.get('CLOUDY_PIXEL_PERCENTAGE')
            })
        
        features = sentinel.map(calculate_ndvi).getInfo()
        
        data = []
        for feature in features['features']:
            props = feature['properties']
            if props.get('NDVI') is not None:
                data.append(props)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_soil_moisture(self,
                         region: ee.Geometry,
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Get soil moisture data from NASA SMAP
        """
        smap = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007') \
            .filterDate(start_date, end_date) \
            .filterBounds(region)
        
        def extract_sm(image):
            sm = image.select('sm_surface')
            
            stats = sm.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=11000,
                maxPixels=1e9
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'soil_moisture': stats.get('sm_surface')
            })
        
        features = smap.map(extract_sm).getInfo()
        
        data = []
        for feature in features['features']:
            props = feature['properties']
            if props.get('soil_moisture') is not None:
                data.append(props)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def calculate_vegetation_indices(self,
                                    image: ee.Image) -> ee.Image:
        """
        Calculate multiple vegetation indices
        
        Returns image with bands: NDVI, EVI, NDWI, GNDVI
        """
        # Assuming Sentinel-2 or Landsat 8 band names
        # Adjust band names based on your satellite
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # GNDVI = (NIR - Green) / (NIR + Green)
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
        
        return image.addBands([ndvi, evi, ndwi, gndvi])
    
    def collect_county_data(self,
                           state: str,
                           county: str,
                           years: List[int],
                           output_dir: str = "data/raw/satellite") -> Dict[str, pd.DataFrame]:
        """
        Collect comprehensive satellite data for a county
        
        Args:
            state: State name
            county: County name
            years: List of years
            output_dir: Output directory
            
        Returns:
            Dictionary of DataFrames
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get county boundary
        counties = ee.FeatureCollection('TIGER/2018/Counties')
        county_feature = counties.filter(
            ee.Filter.And(
                ee.Filter.eq('STATEFP', self._get_state_fips(state)),
                ee.Filter.eq('NAME', county)
            )
        ).first()
        
        region = county_feature.geometry()
        
        datasets = {}
        
        for year in years:
            logger.info(f"Collecting satellite data for {county}, {state} - {year}")
            
            start_date = f"{year}-04-01"  # Growing season start
            end_date = f"{year}-10-31"    # Growing season end
            
            # Collect different indices
            datasets[f'modis_ndvi_{year}'] = self.get_modis_ndvi(region, start_date, end_date)
            datasets[f'modis_evi_{year}'] = self.get_modis_evi(region, start_date, end_date)
            datasets[f'sentinel2_ndvi_{year}'] = self.get_sentinel2_ndvi(region, start_date, end_date)
            datasets[f'soil_moisture_{year}'] = self.get_soil_moisture(region, start_date, end_date)
            
            # Save individual year data
            for key, df in datasets.items():
                if not df.empty:
                    filepath = f"{output_dir}/{state}_{county}_{key}.csv"
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {key}: {len(df)} records")
        
        return datasets
    
    @staticmethod
    def _get_state_fips(state: str) -> str:
        """Get FIPS code for state (simplified version)"""
        state_fips = {
            'IOWA': '19',
            'ILLINOIS': '17',
            'NEBRASKA': '31',
            'KANSAS': '20',
            'MISSOURI': '29',
            'INDIANA': '18',
            'OHIO': '39'
        }
        return state_fips.get(state.upper(), '00')
    
    @staticmethod
    def create_region_geometry(bbox: Tuple[float, float, float, float]) -> ee.Geometry:
        """
        Create Earth Engine Geometry from bounding box
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
        """
        return ee.Geometry.Rectangle(list(bbox))


def main():
    """Example usage"""
    collector = SatelliteDataCollector()
    
    # Example: Collect data for Story County, Iowa
    datasets = collector.collect_county_data(
        state='IOWA',
        county='Story',
        years=[2020, 2021, 2022],
        output_dir='data/raw/satellite'
    )
    
    logger.info("\n" + "="*50)
    logger.info("Satellite Data Collection Summary")
    logger.info("="*50)
    for key, df in datasets.items():
        if not df.empty:
            logger.info(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()
