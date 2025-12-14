"""
Data Processing Module

This module provides functions for feature engineering and data preprocessing
using sklearn Pipeline for reproducible transformations.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_aggregate_features(df: pd.DataFrame, customer_id_col: str = 'CustomerId') -> pd.DataFrame:
    """
    Create aggregate features at customer level.
    
    Args:
        df: Transaction-level DataFrame
        customer_id_col: Name of customer ID column
        
    Returns:
        DataFrame with aggregate features
    """
    try:
        if customer_id_col not in df.columns:
            raise ValueError(f"Column {customer_id_col} not found in DataFrame")
        
        logger.info("Creating aggregate features...")
        
        # Group by customer
        agg_features = df.groupby(customer_id_col).agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = [
            customer_id_col,
            'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count',
            'Total_Value', 'Avg_Value', 'Std_Value'
        ]
        
        # Fill NaN values with 0 for std columns (when only 1 transaction)
        agg_features['Std_Amount'] = agg_features['Std_Amount'].fillna(0)
        agg_features['Std_Value'] = agg_features['Std_Value'].fillna(0)
        
        logger.info(f"Created aggregate features for {len(agg_features)} customers")
        return agg_features
    
    except Exception as e:
        logger.error(f"Error creating aggregate features: {str(e)}")
        raise


def extract_temporal_features(df: pd.DataFrame, date_col: str = 'TransactionStartTime') -> pd.DataFrame:
    """
    Extract temporal features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of datetime column
        
    Returns:
        DataFrame with temporal features added
    """
    try:
        if date_col not in df.columns:
            raise ValueError(f"Column {date_col} not found in DataFrame")
        
        logger.info("Extracting temporal features...")
        df = df.copy()
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract temporal features
        df['TransactionHour'] = df[date_col].dt.hour
        df['TransactionDay'] = df[date_col].dt.day
        df['TransactionMonth'] = df[date_col].dt.month
        df['TransactionYear'] = df[date_col].dt.year
        df['TransactionDayOfWeek'] = df[date_col].dt.dayofweek
        
        logger.info("Temporal features extracted successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error extracting temporal features: {str(e)}")
        raise


def create_preprocessing_pipeline(numerical_cols: list, categorical_cols: list) -> Pipeline:
    """
    Create sklearn Pipeline for data preprocessing.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        sklearn Pipeline object
    """
    try:
        logger.info("Creating preprocessing pipeline...")
        
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        
        logger.info("Preprocessing pipeline created successfully")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        raise


def process_data(df: pd.DataFrame, 
                 customer_id_col: str = 'CustomerId',
                 date_col: str = 'TransactionStartTime',
                 target_col: str = None) -> tuple:
    """
    Main data processing function that orchestrates feature engineering.
    
    Args:
        df: Raw transaction DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of datetime column
        target_col: Optional target column name
        
    Returns:
        Tuple of (processed_features_df, target_series) or (processed_features_df, None)
    """
    try:
        logger.info("Starting data processing pipeline...")
        
        # Extract temporal features
        df = extract_temporal_features(df, date_col)
        
        # Create aggregate features
        agg_features = create_aggregate_features(df, customer_id_col)
        
        # Merge back temporal features (using most recent transaction)
        temporal_features = df.groupby(customer_id_col).agg({
            'TransactionHour': 'last',
            'TransactionDay': 'last',
            'TransactionMonth': 'last',
            'TransactionYear': 'last',
            'TransactionDayOfWeek': 'last',
            'ProductCategory': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'ChannelId': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'CountryCode': 'first',
            'CurrencyCode': 'first',
            'PricingStrategy': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'FraudResult': 'sum'  # Total fraud count per customer
        }).reset_index()
        
        # Merge all features
        processed_df = agg_features.merge(temporal_features, on=customer_id_col, how='left')
        
        # Extract target if provided
        target = None
        if target_col and target_col in processed_df.columns:
            target = processed_df[target_col]
            processed_df = processed_df.drop(columns=[target_col])
        
        logger.info(f"Data processing completed. Final shape: {processed_df.shape}")
        return processed_df, target
    
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

