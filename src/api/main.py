"""
FastAPI Application for Credit Risk Prediction

This module provides REST API endpoints for credit risk prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from typing import Optional
import os

try:
    from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse
    from src.predict import load_model_from_mlflow, predict_risk_probability, calculate_credit_score
except ImportError:
    # Fallback for relative imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse
    from src.predict import load_model_from_mlflow, predict_risk_probability, calculate_credit_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability and credit scores",
    version="1.0.0"
)

# Global model variable
model = None
model_loaded = False


def load_model():
    """Load the model on startup."""
    global model, model_loaded
    try:
        # Try to load from environment variable or use default
        model_name = os.getenv("MLFLOW_MODEL_NAME", None)
        run_id = os.getenv("MLFLOW_RUN_ID", None)
        
        if model_name or run_id:
            model = load_model_from_mlflow(run_id=run_id, model_name=model_name)
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.warning("No model specified. Set MLFLOW_MODEL_NAME or MLFLOW_RUN_ID environment variable")
            model_loaded = False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    load_model()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability and credit score for a customer.
    
    Args:
        request: PredictionRequest with customer features
        
    Returns:
        PredictionResponse with risk probability, credit score, and risk class
    """
    try:
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check model configuration."
            )
        
        # Convert request to DataFrame
        features_dict = request.dict()
        customer_id = features_dict.pop('CustomerId')
        
        # Create DataFrame with single row
        features_df = pd.DataFrame([features_dict])
        
        # Ensure correct column order (match training data)
        # This should match the feature order from training
        expected_columns = [
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count',
            'Total_Value', 'Avg_Value', 'Std_Value',
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 
            'TransactionYear', 'TransactionDayOfWeek',
            'ProductCategory', 'ChannelId', 'CountryCode', 
            'CurrencyCode', 'PricingStrategy', 'FraudResult'
        ]
        
        # Reorder columns
        features_df = features_df[expected_columns]
        
        # Make prediction
        risk_probability = predict_risk_probability(model, features_df)[0]
        credit_score = calculate_credit_score(risk_probability)
        risk_class = "high_risk" if risk_probability >= 0.5 else "low_risk"
        
        logger.info(f"Prediction for {customer_id}: prob={risk_probability:.4f}, score={credit_score}")
        
        return PredictionResponse(
            CustomerId=customer_id,
            risk_probability=float(risk_probability),
            credit_score=credit_score,
            risk_class=risk_class
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Predict credit risk for multiple customers (batch prediction).
    
    Args:
        requests: List of PredictionRequest objects
        
    Returns:
        List of PredictionResponse objects
    """
    try:
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check model configuration."
            )
        
        if len(requests) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty request list"
            )
        
        # Convert requests to DataFrame
        features_list = []
        customer_ids = []
        
        for req in requests:
            features_dict = req.dict()
            customer_ids.append(features_dict.pop('CustomerId'))
            features_list.append(features_dict)
        
        features_df = pd.DataFrame(features_list)
        
        # Make batch predictions
        risk_probabilities = predict_risk_probability(model, features_df)
        
        # Create responses
        responses = []
        for i, customer_id in enumerate(customer_ids):
            prob = risk_probabilities[i]
            score = calculate_credit_score(prob)
            risk_class = "high_risk" if prob >= 0.5 else "low_risk"
            
            responses.append(PredictionResponse(
                CustomerId=customer_id,
                risk_probability=float(prob),
                credit_score=score,
                risk_class=risk_class
            ))
        
        logger.info(f"Batch prediction completed for {len(requests)} customers")
        return responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )
