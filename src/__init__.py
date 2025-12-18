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

from src.rfm_analysis import (
    calculate_rfm_metrics,
    create_proxy_target,
    create_rfm_target_variable
)

from src.woe_transformer import (
    WoETransformer,
    calculate_information_value
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
    'calculate_rfm_metrics',
    'create_proxy_target',
    'create_rfm_target_variable',
    'WoETransformer',
    'calculate_information_value',
]


