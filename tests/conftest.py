"""
Pytest Configuration and Fixtures

This module provides shared fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data."""
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2'],
        'Amount': [100.0, 200.0, 150.0, 250.0],
        'Value': [100.0, 200.0, 150.0, 250.0],
        'TransactionStartTime': [
            '2024-01-15 14:30:00',
            '2024-01-16 15:00:00',
            '2024-01-17 10:00:00',
            '2024-01-18 11:00:00'
        ],
        'ProductCategory': ['A', 'A', 'B', 'B'],
        'ChannelId': ['web', 'web', 'mobile', 'mobile'],
        'CountryCode': [256, 256, 256, 256],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX'],
        'PricingStrategy': [2, 2, 2, 2],
        'FraudResult': [0, 0, 0, 1]
    })


@pytest.fixture
def temp_csv_file(sample_transaction_data):
    """Fixture providing a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_transaction_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

