"""
Pydantic Models for API Request/Response Validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    CustomerId: str = Field(..., description="Customer ID")
    Total_Amount: float = Field(..., description="Total transaction amount")
    Avg_Amount: float = Field(..., description="Average transaction amount")
    Std_Amount: float = Field(..., description="Standard deviation of amounts")
    Transaction_Count: int = Field(..., description="Number of transactions")
    Total_Value: float = Field(..., description="Total transaction value")
    Avg_Value: float = Field(..., description="Average transaction value")
    Std_Value: float = Field(..., description="Standard deviation of values")
    TransactionHour: int = Field(..., ge=0, le=23, description="Transaction hour")
    TransactionDay: int = Field(..., ge=1, le=31, description="Transaction day")
    TransactionMonth: int = Field(..., ge=1, le=12, description="Transaction month")
    TransactionYear: int = Field(..., ge=2000, le=2100, description="Transaction year")
    TransactionDayOfWeek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Channel ID")
    CountryCode: int = Field(..., description="Country code")
    CurrencyCode: str = Field(..., description="Currency code")
    PricingStrategy: int = Field(..., description="Pricing strategy")
    FraudResult: int = Field(..., ge=0, description="Fraud result count")
    
    @validator('Total_Amount', 'Avg_Amount', 'Total_Value', 'Avg_Value')
    def validate_amounts(cls, v):
        """Validate that amounts are reasonable."""
        if not np.isfinite(v):
            raise ValueError("Amount must be a finite number")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerId": "CUST001",
                "Total_Amount": 50000.0,
                "Avg_Amount": 5000.0,
                "Std_Amount": 1000.0,
                "Transaction_Count": 10,
                "Total_Value": 50000.0,
                "Avg_Value": 5000.0,
                "Std_Value": 1000.0,
                "TransactionHour": 14,
                "TransactionDay": 15,
                "TransactionMonth": 6,
                "TransactionYear": 2024,
                "TransactionDayOfWeek": 2,
                "ProductCategory": "financial_services",
                "ChannelId": "web",
                "CountryCode": 256,
                "CurrencyCode": "UGX",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    CustomerId: str = Field(..., description="Customer ID")
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_class: str = Field(..., description="Risk class (low_risk or high_risk)")
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerId": "CUST001",
                "risk_probability": 0.25,
                "credit_score": 712,
                "risk_class": "low_risk"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }
