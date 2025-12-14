"""
Example Usage Script

This script demonstrates how to use the data processing, training, and prediction modules.
"""

import pandas as pd
import numpy as np
from src.data_processing import load_data, process_data
from src.train import train_models
from src.predict import predict_risk_probability, calculate_credit_score

def example_data_processing():
    """Example of data processing."""
    print("=" * 60)
    print("Example: Data Processing")
    print("=" * 60)
    
    try:
        # Load raw data
        df = load_data("data/raw/data.csv")
        print(f"Loaded {len(df)} transactions")
        
        # Process data
        processed_df, target = process_data(df)
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Features: {list(processed_df.columns[:5])}...")
        
        return processed_df, target
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None


def example_training():
    """Example of model training (requires processed data with target)."""
    print("\n" + "=" * 60)
    print("Example: Model Training")
    print("=" * 60)
    
    print("Note: This requires processed data with target variable.")
    print("Training will be implemented in Task 4 after proxy variable creation.")
    print("Example usage:")
    print("  from src.train import train_models")
    print("  results = train_models('data/processed/train_data.csv', target_col='is_high_risk')")


def example_prediction():
    """Example of making predictions."""
    print("\n" + "=" * 60)
    print("Example: Prediction")
    print("=" * 60)
    
    print("Example usage:")
    print("  from src.predict import load_model_from_mlflow, predict_risk_probability")
    print("  model = load_model_from_mlflow(model_name='credit-risk-model')")
    print("  probabilities = predict_risk_probability(model, features_df)")
    print("  credit_scores = [calculate_credit_score(p) for p in probabilities]")


if __name__ == "__main__":
    print("Credit Risk Modeling - Example Usage\n")
    
    # Example 1: Data Processing
    processed_df, target = example_data_processing()
    
    # Example 2: Training (info only)
    example_training()
    
    # Example 3: Prediction (info only)
    example_prediction()
    
    print("\n" + "=" * 60)
    print("For API usage, start the FastAPI server:")
    print("  uvicorn src.api.main:app --reload")
    print("=" * 60)

