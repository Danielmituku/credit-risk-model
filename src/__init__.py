"""
Credit Risk Modeling Package

This package provides modules for data processing, model training, and prediction.
"""

__version__ = "1.0.0"

from src.data_processing import (
    load_data,
    create_aggregate_features,
    extract_temporal_features,
    create_preprocessing_pipeline,
    process_data
)

from src.predict import (
    load_model_from_mlflow,
    predict_risk_probability,
    predict_risk_class,
    calculate_credit_score
)

__all__ = [
    'load_data',
    'create_aggregate_features',
    'extract_temporal_features',
    'create_preprocessing_pipeline',
    'process_data',
    'load_model_from_mlflow',
    'predict_risk_probability',
    'predict_risk_class',
    'calculate_credit_score',
]

