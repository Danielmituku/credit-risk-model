"""
Unit Tests for WoE Transformer Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.woe_transformer import WoETransformer, calculate_information_value


class TestWoETransformer:
    """Tests for WoETransformer class."""
    
    def test_woe_transformer_fit_transform(self):
        """Test WoE transformer fit and transform."""
        # Create test data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        transformer = WoETransformer(n_bins=5, min_samples=5)
        transformer.fit(X, y)
        
        X_transformed = transformer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
        assert len(X_transformed.columns) == len(X.columns)
    
    def test_woe_transformer_with_categorical(self):
        """Test WoE transformer with mixed data types."""
        X = pd.DataFrame({
            'numerical': np.random.randn(50),
            'categorical': ['A', 'B'] * 25
        })
        y = pd.Series(np.random.randint(0, 2, 50))
        
        transformer = WoETransformer(n_bins=5)
        transformer.fit(X, y)
        
        X_transformed = transformer.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)


class TestCalculateInformationValue:
    """Tests for calculate_information_value function."""
    
    def test_calculate_iv_success(self):
        """Test successful IV calculation."""
        # Create test data with some predictive power
        np.random.seed(42)
        X = pd.DataFrame({
            'good_feature': np.concatenate([
                np.random.randn(50) + 2,  # Class 0
                np.random.randn(50) - 2   # Class 1
            ]),
            'bad_feature': np.random.randn(100)
        })
        y = pd.Series([0] * 50 + [1] * 50)
        
        iv_df = calculate_information_value(X, y, n_bins=5)
        
        assert isinstance(iv_df, pd.DataFrame)
        assert 'Feature' in iv_df.columns
        assert 'IV' in iv_df.columns
        assert 'Strength' in iv_df.columns
        assert len(iv_df) == 2  # Two features
        
        # Good feature should have higher IV
        good_iv = iv_df[iv_df['Feature'] == 'good_feature']['IV'].iloc[0]
        bad_iv = iv_df[iv_df['Feature'] == 'bad_feature']['IV'].iloc[0]
        assert good_iv > bad_iv

