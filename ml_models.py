"""
Machine Learning Models for Crop Yield Prediction
Implements multiple ML and DL models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, Tuple, List, Any
import joblib
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropYieldModels:
    """Collection of ML models for crop yield prediction"""
    
    def __init__(self, random_state: int = 42):
        """Initialize models"""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    scale: bool = True) -> Dict:
        """
        Prepare data for training
        
        Returns:
            Dictionary with train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        # Scale features
        if scale:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            if X_val is not None:
                X_val = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
            
            self.scalers['standard'] = scaler
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train linear regression model"""
        logger.info("Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['linear_regression'] = model
        
        # Feature importance (coefficients)
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['linear_regression'] = importance
        
        return {'model': model, 'feature_importance': importance}
    
    def train_random_forest(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           n_estimators: int = 200,
                           max_depth: int = 20) -> Dict:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = importance
        
        return {'model': model, 'feature_importance': importance}
    
    def train_xgboost(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame = None,
                     y_val: pd.Series = None) -> Dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=False
        )
        
        self.models['xgboost'] = model
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = importance
        
        return {'model': model, 'feature_importance': importance}
    
    def train_lightgbm(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: pd.DataFrame = None,
                      y_val: pd.Series = None) -> Dict:
        """Train LightGBM model"""
        logger.info("Training LightGBM...")
        
        params = {
            'objective': 'regression',
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = model
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['lightgbm'] = importance
        
        return {'model': model, 'feature_importance': importance}
    
    def train_svr(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train Support Vector Regression"""
        logger.info("Training SVR...")
        
        model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        model.fit(X_train, y_train)
        
        self.models['svr'] = model
        
        return {'model': model}
    
    def build_mlp(self, input_dim: int) -> keras.Model:
        """Build Multi-Layer Perceptron"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_mlp(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_val: pd.DataFrame = None,
                 y_val: pd.Series = None,
                 epochs: int = 100,
                 batch_size: int = 32) -> Dict:
        """Train MLP model"""
        logger.info("Training MLP (Neural Network)...")
        
        model = self.build_mlp(X_train.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['mlp'] = model
        
        return {'model': model, 'history': history.history}
    
    def build_lstm(self, timesteps: int, features: int) -> keras.Model:
        """Build LSTM model for time series"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def evaluate_model(self,
                      model_name: str,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with metrics
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'n_samples': len(y_test)
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame = None,
                        y_val: pd.Series = None) -> Dict:
        """Train all available models"""
        logger.info("\n" + "="*50)
        logger.info("Training All Models")
        logger.info("="*50)
        
        results = {}
        
        # Traditional ML models
        results['linear_regression'] = self.train_linear_regression(X_train, y_train)
        results['random_forest'] = self.train_random_forest(X_train, y_train)
        results['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        results['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        results['svr'] = self.train_svr(X_train, y_train)
        
        # Deep learning models
        results['mlp'] = self.train_mlp(X_train, y_train, X_val, y_val)
        
        return results
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare all trained models"""
        logger.info("\n" + "="*50)
        logger.info("Model Comparison")
        logger.info("="*50)
        
        results = []
        for model_name in self.models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results.append(metrics)
        
        comparison = pd.DataFrame(results)
        comparison = comparison.sort_values('r2', ascending=False)
        
        logger.info("\n" + comparison.to_string(index=False))
        
        return comparison
    
    def save_models(self, output_dir: str = "models"):
        """Save all trained models"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = f"{output_dir}/{model_name}.pkl"
            
            if isinstance(model, keras.Model):
                # Save Keras models differently
                model.save(f"{output_dir}/{model_name}.h5")
                logger.info(f"Saved {model_name} to {output_dir}/{model_name}.h5")
            else:
                joblib.dump(model, filepath)
                logger.info(f"Saved {model_name} to {filepath}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            filepath = f"{output_dir}/scaler_{scaler_name}.pkl"
            joblib.dump(scaler, filepath)
        
        # Save feature importance
        for model_name, importance in self.feature_importance.items():
            filepath = f"{output_dir}/feature_importance_{model_name}.csv"
            importance.to_csv(filepath, index=False)
    
    def load_model(self, model_name: str, model_dir: str = "models"):
        """Load a saved model"""
        filepath = f"{model_dir}/{model_name}.pkl"
        h5_filepath = f"{model_dir}/{model_name}.h5"
        
        if Path(h5_filepath).exists():
            model = keras.models.load_model(h5_filepath)
        else:
            model = joblib.load(filepath)
        
        self.models[model_name] = model
        return model


def main():
    """Example usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    y = pd.Series(
        X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 10 + 100,
        name='yield'
    )
    
    # Initialize and train models
    crop_models = CropYieldModels()
    
    # Prepare data
    data = crop_models.prepare_data(X, y)
    
    # Train all models
    crop_models.train_all_models(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val']
    )
    
    # Compare models
    comparison = crop_models.compare_models(data['X_test'], data['y_test'])
    
    # Save models
    crop_models.save_models()
    
    logger.info("\nBest model: " + comparison.iloc[0]['model'])


if __name__ == "__main__":
    main()
