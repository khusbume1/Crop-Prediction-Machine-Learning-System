"""
Complete Training Pipeline for Crop Yield Prediction
End-to-end script to train and evaluate all models
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ml_models import CropYieldModels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CropPredictionPipeline:
    """Complete pipeline for crop yield prediction"""
    
    def __init__(self, config_path: str = "config/config_template.yaml"):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = CropYieldModels()
        self.results = {}
    
    def load_and_merge_data(self, data_dir: str = "data/processed") -> pd.DataFrame:
        """
        Load and merge all processed data sources
        
        Expected files in data_dir:
        - yield_data.csv (target variable)
        - weather_features.csv
        - satellite_features.csv
        - soil_features.csv (optional)
        
        Returns:
            Merged DataFrame ready for modeling
        """
        logger.info("Loading data files...")
        
        # Load yield data (target)
        yield_path = Path(data_dir) / "yield_data.csv"
        if not yield_path.exists():
            raise FileNotFoundError(f"Yield data not found at {yield_path}")
        
        df_yield = pd.read_csv(yield_path)
        logger.info(f"Loaded yield data: {len(df_yield)} records")
        
        # Load weather features
        weather_path = Path(data_dir) / "weather_features.csv"
        if weather_path.exists():
            df_weather = pd.read_csv(weather_path)
            logger.info(f"Loaded weather features: {len(df_weather)} records")
            
            # Merge on common keys (e.g., year, state, county)
            merge_keys = ['year', 'state', 'county']
            merge_keys = [k for k in merge_keys if k in df_yield.columns and k in df_weather.columns]
            
            df = pd.merge(df_yield, df_weather, on=merge_keys, how='left')
        else:
            logger.warning("Weather features not found, using yield data only")
            df = df_yield.copy()
        
        # Load satellite features
        satellite_path = Path(data_dir) / "satellite_features.csv"
        if satellite_path.exists():
            df_satellite = pd.read_csv(satellite_path)
            logger.info(f"Loaded satellite features: {len(df_satellite)} records")
            
            merge_keys = ['year', 'state', 'county']
            merge_keys = [k for k in merge_keys if k in df.columns and k in df_satellite.columns]
            
            df = pd.merge(df, df_satellite, on=merge_keys, how='left')
        
        # Load soil features (if available)
        soil_path = Path(data_dir) / "soil_features.csv"
        if soil_path.exists():
            df_soil = pd.read_csv(soil_path)
            logger.info(f"Loaded soil features: {len(df_soil)} records")
            
            merge_keys = ['state', 'county']  # Soil usually doesn't change by year
            merge_keys = [k for k in merge_keys if k in df.columns and k in df_soil.columns]
            
            df = pd.merge(df, df_soil, on=merge_keys, how='left')
        
        logger.info(f"Final merged dataset: {len(df)} records, {len(df.columns)} features")
        
        return df
    
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic data for testing when real data is not available
        """
        logger.warning("Creating synthetic data for demonstration")
        
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'year': np.random.choice(range(2015, 2024), n_samples),
            'state': np.random.choice(['IOWA', 'ILLINOIS', 'NEBRASKA', 'KANSAS'], n_samples),
            'county': np.random.choice([f'County_{i}' for i in range(20)], n_samples),
            
            # Weather features
            'temp_mean': np.random.uniform(15, 25, n_samples),
            'temp_max': np.random.uniform(25, 35, n_samples),
            'temp_min': np.random.uniform(5, 15, n_samples),
            'precipitation': np.random.uniform(400, 900, n_samples),
            'humidity': np.random.uniform(50, 80, n_samples),
            'gdd_cumulative': np.random.uniform(1500, 2500, n_samples),
            'drought_index': np.random.uniform(0, 1, n_samples),
            
            # Satellite features
            'ndvi_mean': np.random.uniform(0.6, 0.9, n_samples),
            'ndvi_max': np.random.uniform(0.7, 0.95, n_samples),
            'evi_mean': np.random.uniform(0.5, 0.8, n_samples),
            'soil_moisture_mean': np.random.uniform(0.2, 0.4, n_samples),
            
            # Soil features
            'soil_organic_matter': np.random.uniform(2, 5, n_samples),
            'soil_ph': np.random.uniform(6, 7.5, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate target (yield) with realistic relationships
        df['yield'] = (
            100 +  # Base yield
            df['precipitation'] * 0.05 +  # Positive relationship with rain
            df['gdd_cumulative'] * 0.02 +  # Positive with growing degree days
            df['ndvi_mean'] * 80 +  # Strong positive with NDVI
            df['soil_organic_matter'] * 10 +  # Positive with soil quality
            -df['drought_index'] * 30 +  # Negative with drought
            np.random.normal(0, 15, n_samples)  # Random noise
        )
        
        # Ensure realistic yield range (bushels/acre for corn)
        df['yield'] = df['yield'].clip(80, 220)
        
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """
        Separate features and target
        
        Returns:
            (X, y) tuple
        """
        # Identify target column
        target_cols = ['yield', 'Yield', 'Value', 'production']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("No target column found in data")
        
        logger.info(f"Using '{target_col}' as target variable")
        
        # Remove non-feature columns
        exclude_cols = [target_col, 'state', 'county', 'date', 'fips']
        exclude_cols = [c for c in exclude_cols if c in df.columns]
        
        X = df.drop(columns=exclude_cols)
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert categorical to numeric if needed
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Feature names: {list(X.columns)}")
        
        return X, y
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train all models and evaluate performance
        """
        logger.info("\n" + "="*70)
        logger.info("TRAINING CROP YIELD PREDICTION MODELS")
        logger.info("="*70)
        
        # Prepare data
        data = self.models.prepare_data(
            X, y,
            test_size=self.config['models']['test_size'],
            val_size=self.config['models']['validation_size']
        )
        
        logger.info(f"Train set: {len(data['X_train'])} samples")
        logger.info(f"Validation set: {len(data['X_val'])} samples")
        logger.info(f"Test set: {len(data['X_test'])} samples")
        
        # Train all models
        train_results = self.models.train_all_models(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val']
        )
        
        # Evaluate on test set
        comparison = self.models.compare_models(data['X_test'], data['y_test'])
        
        # Save results
        self.results = {
            'data_splits': {
                'train_size': len(data['X_train']),
                'val_size': len(data['X_val']),
                'test_size': len(data['X_test'])
            },
            'model_comparison': comparison.to_dict('records'),
            'best_model': comparison.iloc[0]['model'],
            'best_r2': comparison.iloc[0]['r2'],
            'best_rmse': comparison.iloc[0]['rmse']
        }
        
        return self.results
    
    def save_results(self, output_dir: str = "outputs"):
        """Save all results and models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        models_dir = output_path / "models"
        self.models.save_models(str(models_dir))
        
        # Save results
        results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results_serializable = {
                k: {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2 
                    for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in self.results.items()
            }
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Models saved to {models_dir}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("="*70)
        logger.info(f"Best Model: {self.results['best_model']}")
        logger.info(f"Best R²: {self.results['best_r2']:.4f}")
        logger.info(f"Best RMSE: {self.results['best_rmse']:.2f}")
        logger.info("="*70)
    
    def run_complete_pipeline(self, use_synthetic: bool = False):
        """Run complete training pipeline"""
        try:
            # Load data
            if use_synthetic:
                df = self.create_synthetic_data()
            else:
                df = self.load_and_merge_data()
            
            # Prepare features and target
            X, y = self.prepare_features_and_target(df)
            
            # Train and evaluate
            results = self.train_and_evaluate(X, y)
            
            # Save everything
            self.save_results()
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train crop yield prediction models')
    parser.add_argument('--config', type=str, default='config/config_template.yaml',
                      help='Path to configuration file')
    parser.add_argument('--synthetic', action='store_true',
                      help='Use synthetic data for testing')
    parser.add_argument('--output', type=str, default='outputs',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CropPredictionPipeline(args.config)
    
    # Run pipeline
    pipeline.run_complete_pipeline(use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
