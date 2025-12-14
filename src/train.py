"""
Model Training Module

This module provides functions for training credit risk models with MLflow tracking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "credit-risk-modeling") -> None:
    """
    Setup MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment '{experiment_name}' set up successfully")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise


def train_logistic_regression(X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              random_state: int = 42) -> Dict[str, Any]:
    """
    Train Logistic Regression model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Logistic Regression model...")
        
        with mlflow.start_run(run_name="logistic_regression"):
            # Train model
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'LogisticRegression',
                'random_state': random_state,
                'max_iter': 1000
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Logistic Regression: {str(e)}")
        raise


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        random_state: int = 42,
                        n_estimators: int = 100) -> Dict[str, Any]:
    """
    Train Random Forest model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        n_estimators: Number of trees
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Random Forest model...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'RandomForest',
                'random_state': random_state,
                'n_estimators': n_estimators
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise


def train_models(data_path: str,
                 target_col: str = 'is_high_risk',
                 test_size: float = 0.2,
                 random_state: int = 42) -> Dict[str, Any]:
    """
    Main training function that loads data, processes it, and trains multiple models.
    
    Args:
        data_path: Path to processed data CSV
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with trained models and results
    """
    try:
        logger.info("Starting model training pipeline...")
        
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        try:
            from src.data_processing import load_data
        except ImportError:
            # Fallback for relative imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.data_processing import load_data
        
        df = load_data(data_path)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Train models
        results = {}
        
        # Logistic Regression
        lr_results = train_logistic_regression(X_train, y_train, X_test, y_test, random_state)
        results['logistic_regression'] = lr_results
        
        # Random Forest
        rf_results = train_random_forest(X_train, y_train, X_test, y_test, random_state)
        results['random_forest'] = rf_results
        
        logger.info("Model training completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
