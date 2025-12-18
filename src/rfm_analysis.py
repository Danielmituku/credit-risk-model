"""
RFM Analysis Module

This module provides functions for calculating RFM metrics and creating
proxy target variables using K-Means clustering.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rfm_metrics(df: pd.DataFrame,
                         customer_id_col: str = 'CustomerId',
                         date_col: str = 'TransactionStartTime',
                         amount_col: str = 'Value',
                         snapshot_date: str = None) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        df: Transaction-level DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of datetime column
        amount_col: Name of amount/value column
        snapshot_date: Snapshot date for recency calculation (default: max date)
        
    Returns:
        DataFrame with RFM metrics per customer
    """
    try:
        logger.info("Calculating RFM metrics...")
        
        # Convert date column to datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Set snapshot date (default to max date in dataset)
        if snapshot_date is None:
            snapshot_date = df[date_col].max()
        else:
            snapshot_date = pd.to_datetime(snapshot_date)
        
        logger.info(f"Using snapshot date: {snapshot_date}")
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (snapshot_date - x.max()).days,  # Recency (days since last transaction)
            customer_id_col: 'count',  # Frequency (number of transactions)
            amount_col: ['sum', 'mean']  # Monetary (total and average)
        }).reset_index()
        
        # Flatten column names
        rfm.columns = [
            customer_id_col,
            'Recency',
            'Frequency',
            'Monetary_Total',
            'Monetary_Mean'
        ]
        
        # Handle edge cases
        rfm['Recency'] = rfm['Recency'].fillna(999)  # If no transactions, set high recency
        rfm['Frequency'] = rfm['Frequency'].fillna(0)
        rfm['Monetary_Total'] = rfm['Monetary_Total'].fillna(0)
        rfm['Monetary_Mean'] = rfm['Monetary_Mean'].fillna(0)
        
        logger.info(f"RFM metrics calculated for {len(rfm)} customers")
        logger.info(f"Recency: min={rfm['Recency'].min()}, max={rfm['Recency'].max()}, mean={rfm['Recency'].mean():.2f}")
        logger.info(f"Frequency: min={rfm['Frequency'].min()}, max={rfm['Frequency'].max()}, mean={rfm['Frequency'].mean():.2f}")
        logger.info(f"Monetary_Total: min={rfm['Monetary_Total'].min()}, max={rfm['Monetary_Total'].max()}, mean={rfm['Monetary_Total'].mean():.2f}")
        
        return rfm
    
    except Exception as e:
        logger.error(f"Error calculating RFM metrics: {str(e)}")
        raise


def create_proxy_target(rfm_df: pd.DataFrame,
                       customer_id_col: str = 'CustomerId',
                       n_clusters: int = 3,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Create proxy target variable using K-Means clustering on RFM metrics.
    
    High-risk customers are identified as the cluster with:
    - High Recency (inactive)
    - Low Frequency
    - Low Monetary value
    
    Args:
        rfm_df: DataFrame with RFM metrics
        customer_id_col: Name of customer ID column
        n_clusters: Number of clusters for K-Means
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with is_high_risk column added
    """
    try:
        logger.info(f"Creating proxy target using K-Means clustering (n_clusters={n_clusters})...")
        
        # Select RFM features for clustering
        rfm_features = ['Recency', 'Frequency', 'Monetary_Total']
        X = rfm_df[rfm_features].copy()
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        cluster_stats = rfm_df.groupby('Cluster')[rfm_features].mean()
        logger.info("\nCluster Characteristics:")
        logger.info(cluster_stats)
        
        # Identify high-risk cluster
        # High-risk = High Recency (inactive) + Low Frequency + Low Monetary
        # We'll use a scoring approach: higher score = higher risk
        cluster_scores = {}
        for cluster_id in range(n_clusters):
            cluster_data = cluster_stats.loc[cluster_id]
            # Risk score: High Recency (bad) + Low Frequency (bad) + Low Monetary (bad)
            # Normalize and combine
            score = (
                cluster_data['Recency'] / cluster_data['Recency'].max() +  # Higher recency = higher risk
                (1 - cluster_data['Frequency'] / cluster_data['Frequency'].max()) +  # Lower frequency = higher risk
                (1 - cluster_data['Monetary_Total'] / cluster_data['Monetary_Total'].max())  # Lower monetary = higher risk
            )
            cluster_scores[cluster_id] = score
        
        # Find cluster with highest risk score
        high_risk_cluster = max(cluster_scores, key=cluster_scores.get)
        logger.info(f"\nHigh-risk cluster identified: Cluster {high_risk_cluster}")
        logger.info(f"Risk score: {cluster_scores[high_risk_cluster]:.4f}")
        logger.info(f"Cluster characteristics:\n{cluster_stats.loc[high_risk_cluster]}")
        
        # Create binary target variable
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
        
        # Log target distribution
        risk_distribution = rfm_df['is_high_risk'].value_counts()
        logger.info(f"\nTarget variable distribution:")
        logger.info(f"High-risk (1): {risk_distribution.get(1, 0)} ({risk_distribution.get(1, 0)/len(rfm_df)*100:.2f}%)")
        logger.info(f"Low-risk (0): {risk_distribution.get(0, 0)} ({risk_distribution.get(0, 0)/len(rfm_df)*100:.2f}%)")
        
        return rfm_df
    
    except Exception as e:
        logger.error(f"Error creating proxy target: {str(e)}")
        raise


def create_rfm_target_variable(df: pd.DataFrame,
                               customer_id_col: str = 'CustomerId',
                               date_col: str = 'TransactionStartTime',
                               amount_col: str = 'Value',
                               snapshot_date: str = None,
                               n_clusters: int = 3,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to create RFM-based proxy target variable.
    
    Args:
        df: Transaction-level DataFrame
        customer_id_col: Name of customer ID column
        date_col: Name of datetime column
        amount_col: Name of amount/value column
        snapshot_date: Snapshot date for recency calculation
        n_clusters: Number of clusters for K-Means
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (rfm_df with target, original df with target merged)
    """
    try:
        logger.info("Starting RFM-based proxy target creation pipeline...")
        
        # Step 1: Calculate RFM metrics
        rfm_df = calculate_rfm_metrics(
            df, customer_id_col, date_col, amount_col, snapshot_date
        )
        
        # Step 2: Create proxy target using clustering
        rfm_df = create_proxy_target(
            rfm_df, customer_id_col, n_clusters, random_state
        )
        
        # Step 3: Merge target back to original dataframe (optional)
        # Keep only necessary columns for merging
        target_df = rfm_df[[customer_id_col, 'is_high_risk', 'Recency', 'Frequency', 'Monetary_Total', 'Monetary_Mean']]
        
        logger.info("RFM-based proxy target creation completed successfully")
        return rfm_df, target_df
    
    except Exception as e:
        logger.error(f"Error in RFM target creation pipeline: {str(e)}")
        raise

