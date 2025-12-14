"""
Unit Tests for Data Processing Module

This module contains unit tests for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    load_data,
    create_aggregate_features,
    extract_temporal_features,
    create_preprocessing_pipeline,
    process_data
)


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C1'],
            'Amount': [100, 200, 150],
            'Value': [100, 200, 150]
        })
        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        # Load data
        result = load_data(str(test_file))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'CustomerId' in result.columns
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_empty_file(self, tmp_path):
        """Test error handling for empty file."""
        test_file = tmp_path / "empty.csv"
        test_file.write_text("")
        
        with pytest.raises(ValueError):
            load_data(str(test_file))


class TestCreateAggregateFeatures:
    """Tests for create_aggregate_features function."""
    
    def test_create_aggregate_features_success(self):
        """Test successful aggregate feature creation."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'Amount': [100, 200, 150, 250],
            'Value': [100, 200, 150, 250]
        })
        
        result = create_aggregate_features(df, 'CustomerId')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two unique customers
        assert 'Total_Amount' in result.columns
        assert 'Avg_Amount' in result.columns
        assert 'Transaction_Count' in result.columns
        assert result['CustomerId'].nunique() == 2
    
    def test_create_aggregate_features_missing_column(self):
        """Test error handling for missing customer ID column."""
        df = pd.DataFrame({
            'Amount': [100, 200],
            'Value': [100, 200]
        })
        
        with pytest.raises(ValueError):
            create_aggregate_features(df, 'CustomerId')


class TestExtractTemporalFeatures:
    """Tests for extract_temporal_features function."""
    
    def test_extract_temporal_features_success(self):
        """Test successful temporal feature extraction."""
        df = pd.DataFrame({
            'TransactionStartTime': [
                '2024-01-15 14:30:00',
                '2024-02-20 09:15:00'
            ]
        })
        
        result = extract_temporal_features(df, 'TransactionStartTime')
        
        assert 'TransactionHour' in result.columns
        assert 'TransactionDay' in result.columns
        assert 'TransactionMonth' in result.columns
        assert 'TransactionYear' in result.columns
        assert 'TransactionDayOfWeek' in result.columns
        assert result['TransactionYear'].iloc[0] == 2024
    
    def test_extract_temporal_features_missing_column(self):
        """Test error handling for missing datetime column."""
        df = pd.DataFrame({
            'Amount': [100, 200]
        })
        
        with pytest.raises(ValueError):
            extract_temporal_features(df, 'TransactionStartTime')


class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline function."""
    
    def test_create_pipeline_success(self):
        """Test successful pipeline creation."""
        numerical_cols = ['Amount', 'Value']
        categorical_cols = ['ProductCategory']
        
        pipeline = create_preprocessing_pipeline(numerical_cols, categorical_cols)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    def test_pipeline_fit_transform(self):
        """Test that pipeline can fit and transform data."""
        numerical_cols = ['Amount']
        categorical_cols = ['Category']
        
        pipeline = create_preprocessing_pipeline(numerical_cols, categorical_cols)
        
        # Create test data
        X = pd.DataFrame({
            'Amount': [100, 200, 150, np.nan],
            'Category': ['A', 'B', 'A', 'B']
        })
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed is not None
        assert not np.isnan(X_transformed).any()  # No NaN after imputation


class TestProcessData:
    """Tests for process_data function."""
    
    def test_process_data_success(self):
        """Test successful data processing."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2'],
            'Amount': [100, 200, 150],
            'Value': [100, 200, 150],
            'TransactionStartTime': [
                '2024-01-15 14:30:00',
                '2024-01-16 15:00:00',
                '2024-01-17 10:00:00'
            ],
            'ProductCategory': ['A', 'A', 'B'],
            'ChannelId': ['web', 'web', 'mobile']
        })
        
        processed_df, target = process_data(df)
        
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == 2  # Two unique customers
        assert 'Total_Amount' in processed_df.columns
        assert 'TransactionHour' in processed_df.columns
    
    def test_process_data_with_target(self):
        """Test data processing with target column."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'Amount': [100, 200],
            'Value': [100, 200],
            'TransactionStartTime': ['2024-01-15', '2024-01-16'],
            'ProductCategory': ['A', 'A'],
            'is_high_risk': [0, 1]
        })
        
        processed_df, target = process_data(df, target_col='is_high_risk')
        
        assert target is not None
        assert len(target) == 2
        assert 'is_high_risk' not in processed_df.columns
