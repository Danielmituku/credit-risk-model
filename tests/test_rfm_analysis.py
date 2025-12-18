"""
Unit Tests for RFM Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rfm_analysis import (
    calculate_rfm_metrics,
    create_proxy_target,
    create_rfm_target_variable
)


class TestCalculateRFMMetrics:
    """Tests for calculate_rfm_metrics function."""
    
    def test_calculate_rfm_metrics_success(self):
        """Test successful RFM metrics calculation."""
        # Create test data
        base_date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'TransactionStartTime': [
                base_date,
                base_date + timedelta(days=5),
                base_date + timedelta(days=10),
                base_date + timedelta(days=15)
            ],
            'Value': [100, 200, 150, 250]
        })
        
        snapshot_date = base_date + timedelta(days=20)
        result = calculate_rfm_metrics(df, snapshot_date=snapshot_date)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two customers
        assert 'Recency' in result.columns
        assert 'Frequency' in result.columns
        assert 'Monetary_Total' in result.columns
        assert 'Monetary_Mean' in result.columns
        
        # Check C1: last transaction 15 days ago, 2 transactions
        c1 = result[result['CustomerId'] == 'C1'].iloc[0]
        assert c1['Frequency'] == 2
        assert c1['Monetary_Total'] == 300
    
    def test_calculate_rfm_metrics_missing_columns(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame({
            'CustomerId': ['C1'],
            'Value': [100]
        })
        
        with pytest.raises((ValueError, KeyError)):
            calculate_rfm_metrics(df)


class TestCreateProxyTarget:
    """Tests for create_proxy_target function."""
    
    def test_create_proxy_target_success(self):
        """Test successful proxy target creation."""
        # Create RFM data
        rfm_df = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3', 'C4'],
            'Recency': [100, 50, 10, 5],  # C1 most inactive
            'Frequency': [1, 2, 10, 15],  # C1 least frequent
            'Monetary_Total': [50, 100, 500, 1000]  # C1 lowest monetary
        })
        
        result = create_proxy_target(rfm_df, n_clusters=3, random_state=42)
        
        assert 'is_high_risk' in result.columns
        assert 'Cluster' in result.columns
        assert result['is_high_risk'].isin([0, 1]).all()
        
        # Check that high-risk customers are identified
        high_risk_count = result['is_high_risk'].sum()
        assert high_risk_count > 0  # At least one high-risk customer


class TestCreateRFMTargetVariable:
    """Tests for create_rfm_target_variable function."""
    
    def test_create_rfm_target_variable_success(self):
        """Test complete RFM target variable creation."""
        base_date = datetime(2024, 1, 1)
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
            'TransactionStartTime': [
                base_date,
                base_date + timedelta(days=5),
                base_date + timedelta(days=10),
                base_date + timedelta(days=15),
                base_date + timedelta(days=20)
            ],
            'Value': [100, 200, 150, 250, 300]
        })
        
        rfm_df, target_df = create_rfm_target_variable(
            df,
            n_clusters=3,
            random_state=42
        )
        
        assert isinstance(rfm_df, pd.DataFrame)
        assert isinstance(target_df, pd.DataFrame)
        assert 'is_high_risk' in rfm_df.columns
        assert 'is_high_risk' in target_df.columns
        assert len(rfm_df) == 3  # Three customers

