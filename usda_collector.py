"""
USDA NASS Data Collector
Collects crop yield and agricultural data from USDA NASS Quick Stats API
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USDADataCollector:
    """Collect agricultural data from USDA NASS Quick Stats API"""
    
    BASE_URL = "http://quickstats.nass.usda.gov/api/api_GET/"
    
    def __init__(self, api_key: str):
        """
        Initialize USDA data collector
        
        Args:
            api_key: USDA NASS API key (get from https://quickstats.nass.usda.gov/api)
        """
        self.api_key = api_key
        self.session = requests.Session()
        
    def _make_request(self, params: Dict) -> pd.DataFrame:
        """Make API request and return data as DataFrame"""
        params['key'] = self.api_key
        params['format'] = 'JSON'
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                logger.warning(f"No data returned for params: {params}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()
    
    def get_crop_yields(self, 
                        crop: str,
                        states: List[str],
                        years: List[int],
                        level: str = "COUNTY") -> pd.DataFrame:
        """
        Get crop yield data
        
        Args:
            crop: Crop name (e.g., 'CORN', 'SOYBEANS', 'WHEAT')
            states: List of state names
            years: List of years to collect
            level: Geographic level ('COUNTY', 'STATE', 'NATIONAL')
            
        Returns:
            DataFrame with yield data
        """
        all_data = []
        
        for state in states:
            for year in years:
                logger.info(f"Fetching {crop} yield data for {state}, {year}")
                
                params = {
                    'source_desc': 'SURVEY',
                    'sector_desc': 'CROPS',
                    'commodity_desc': crop,
                    'statisticcat_desc': 'YIELD',
                    'agg_level_desc': level,
                    'state_name': state,
                    'year': year
                }
                
                df = self._make_request(params)
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)  # Rate limiting
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return self._clean_yield_data(result)
        else:
            return pd.DataFrame()
    
    def get_crop_production(self,
                           crop: str,
                           states: List[str],
                           years: List[int],
                           level: str = "COUNTY") -> pd.DataFrame:
        """Get crop production data (total production)"""
        all_data = []
        
        for state in states:
            for year in years:
                logger.info(f"Fetching {crop} production data for {state}, {year}")
                
                params = {
                    'source_desc': 'SURVEY',
                    'sector_desc': 'CROPS',
                    'commodity_desc': crop,
                    'statisticcat_desc': 'PRODUCTION',
                    'agg_level_desc': level,
                    'state_name': state,
                    'year': year
                }
                
                df = self._make_request(params)
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_planted_acres(self,
                         crop: str,
                         states: List[str],
                         years: List[int],
                         level: str = "COUNTY") -> pd.DataFrame:
        """Get planted acreage data"""
        all_data = []
        
        for state in states:
            for year in years:
                logger.info(f"Fetching {crop} planted acres for {state}, {year}")
                
                params = {
                    'source_desc': 'SURVEY',
                    'sector_desc': 'CROPS',
                    'commodity_desc': crop,
                    'statisticcat_desc': 'AREA PLANTED',
                    'agg_level_desc': level,
                    'state_name': state,
                    'year': year
                }
                
                df = self._make_request(params)
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_crop_progress(self,
                         crop: str,
                         states: List[str],
                         years: List[int]) -> pd.DataFrame:
        """Get crop progress reports (planting, emergence, etc.)"""
        all_data = []
        
        for state in states:
            for year in years:
                logger.info(f"Fetching {crop} progress data for {state}, {year}")
                
                params = {
                    'source_desc': 'SURVEY',
                    'sector_desc': 'CROPS',
                    'commodity_desc': crop,
                    'statisticcat_desc': 'PROGRESS',
                    'state_name': state,
                    'year': year
                }
                
                df = self._make_request(params)
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_crop_condition(self,
                          crop: str,
                          states: List[str],
                          years: List[int]) -> pd.DataFrame:
        """Get crop condition ratings (excellent, good, fair, poor)"""
        all_data = []
        
        for state in states:
            for year in years:
                logger.info(f"Fetching {crop} condition data for {state}, {year}")
                
                params = {
                    'source_desc': 'SURVEY',
                    'sector_desc': 'CROPS',
                    'commodity_desc': crop,
                    'statisticcat_desc': 'CONDITION',
                    'state_name': state,
                    'year': year
                }
                
                df = self._make_request(params)
                if not df.empty:
                    all_data.append(df)
                
                time.sleep(0.5)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _clean_yield_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize yield data"""
        if df.empty:
            return df
        
        # Convert Value to numeric
        df['Value'] = pd.to_numeric(df['Value'].str.replace(',', ''), errors='coerce')
        
        # Extract year, state, county
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['state'] = df['state_name']
        
        if 'county_name' in df.columns:
            df['county'] = df['county_name']
        
        # Keep relevant columns
        cols_to_keep = ['year', 'state', 'county', 'commodity_desc', 
                        'statisticcat_desc', 'Value', 'unit_desc']
        
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[cols_to_keep]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def collect_comprehensive_data(self,
                                  crops: List[str],
                                  states: List[str],
                                  years: List[int],
                                  output_dir: str = "data/raw/usda") -> Dict[str, pd.DataFrame]:
        """
        Collect comprehensive dataset for multiple crops
        
        Returns:
            Dictionary of DataFrames with different data types
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        datasets = {}
        
        for crop in crops:
            logger.info(f"\n{'='*50}")
            logger.info(f"Collecting data for {crop}")
            logger.info(f"{'='*50}")
            
            # Collect different types of data
            datasets[f"{crop}_yield"] = self.get_crop_yields(crop, states, years)
            datasets[f"{crop}_production"] = self.get_crop_production(crop, states, years)
            datasets[f"{crop}_planted"] = self.get_planted_acres(crop, states, years)
            datasets[f"{crop}_progress"] = self.get_crop_progress(crop, states, years)
            datasets[f"{crop}_condition"] = self.get_crop_condition(crop, states, years)
            
            # Save each dataset
            for key, df in datasets.items():
                if not df.empty:
                    filepath = f"{output_dir}/{key}.csv"
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {key}: {len(df)} records to {filepath}")
        
        return datasets


def main():
    """Example usage"""
    # Load configuration
    with open('config/config_template.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize collector
    api_key = config['api_keys']['usda_nass']
    collector = USDADataCollector(api_key)
    
    # Define parameters
    crops = ['CORN', 'SOYBEANS', 'WHEAT']
    states = ['IOWA', 'ILLINOIS', 'NEBRASKA', 'KANSAS']
    years = list(range(2015, 2024))
    
    # Collect data
    datasets = collector.collect_comprehensive_data(crops, states, years)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Data Collection Summary")
    logger.info("="*50)
    for key, df in datasets.items():
        logger.info(f"{key}: {len(df)} records")


if __name__ == "__main__":
    main()
