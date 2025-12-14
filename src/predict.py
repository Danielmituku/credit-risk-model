"""
Prediction Module

This module provides functions for making predictions using trained models.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_mlflow(run_id: str = None, model_name: str = None) -> Any:
    """
    Load a trained model from MLflow.
    
    Args:
        run_id: MLflow run ID (optional)
        model_name: Model name in registry (optional)
        
    Returns:
        Loaded model object
    """
    try:
        if model_name:
            logger.info(f"Loading model '{model_name}' from MLflow registry...")
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        elif run_id:
            logger.info(f"Loading model from run ID: {run_id}")
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        else:
            raise ValueError("Either run_id or model_name must be provided")
        
        logger.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {str(e)}")
        raise


def predict_risk_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """
    Predict risk probability for given features.
    
    Args:
        model: Trained model object
        features: DataFrame with features matching training data
        
    Returns:
        Array of risk probabilities
    """
    try:
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        logger.info(f"Making predictions for {len(features)} samples...")
        
        # Get probability predictions
        probabilities = model.predict_proba(features)[:, 1]
        
        logger.info("Predictions completed successfully")
        return probabilities
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


def predict_risk_class(model: Any, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Predict risk class (binary) for given features.
    
    Args:
        model: Trained model object
        features: DataFrame with features matching training data
        threshold: Probability threshold for classification
        
    Returns:
        Array of risk classes (0 or 1)
    """
    try:
        probabilities = predict_risk_probability(model, features)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    except Exception as e:
        logger.error(f"Error making class predictions: {str(e)}")
        raise


def calculate_credit_score(risk_probability: float, 
                          min_score: int = 300, 
                          max_score: int = 850) -> int:
    """
    Convert risk probability to credit score.
    
    Args:
        risk_probability: Risk probability (0-1)
        min_score: Minimum credit score
        max_score: Maximum credit score
        
    Returns:
        Credit score (300-850 scale)
    """
    try:
        # Lower risk probability = higher credit score
        # Inverse relationship: score = max - (prob * (max - min))
        score = int(max_score - (risk_probability * (max_score - min_score)))
        return max(min_score, min(max_score, score))  # Clamp to range
    
    except Exception as e:
        logger.error(f"Error calculating credit score: {str(e)}")
        raise

