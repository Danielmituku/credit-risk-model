"""
Weight of Evidence (WoE) and Information Value (IV) Transformer

This module provides WoE transformation and IV calculation for feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) transformer for numerical features.
    
    WoE = ln((% of non-events / % of events))
    Higher WoE indicates higher predictive power.
    """
    
    def __init__(self, n_bins: int = 10, min_samples: int = 5):
        """
        Initialize WoE transformer.
        
        Args:
            n_bins: Number of bins for discretization
            min_samples: Minimum samples per bin
        """
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.woe_dict: Dict[str, Dict] = {}
        self.feature_names: list = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit WoE transformer by calculating WoE for each feature.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        try:
            logger.info(f"Fitting WoE transformer on {len(X.columns)} features...")
            self.feature_names = list(X.columns)
            
            # Calculate WoE for each numerical feature
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    woe_map = self._calculate_woe(X[col], y)
                    self.woe_dict[col] = woe_map
                    logger.info(f"Calculated WoE for {col}: {len(woe_map)} bins")
            
            return self
        
        except Exception as e:
            logger.error(f"Error fitting WoE transformer: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using WoE values.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with WoE-transformed features
        """
        try:
            X_transformed = X.copy()
            
            for col in self.feature_names:
                if col in self.woe_dict:
                    X_transformed[col] = X[col].apply(
                        lambda x: self._get_woe_value(col, x)
                    )
            
            return X_transformed
        
        except Exception as e:
            logger.error(f"Error transforming with WoE: {str(e)}")
            raise
    
    def _calculate_woe(self, feature: pd.Series, target: pd.Series) -> Dict:
        """
        Calculate WoE for a feature by binning.
        
        Args:
            feature: Feature values
            target: Target values
            
        Returns:
            Dictionary mapping bin ranges to WoE values
        """
        try:
            # Create bins
            try:
                bins = pd.qcut(feature, q=self.n_bins, duplicates='drop', retbins=True)[1]
            except ValueError:
                # If qcut fails, use equal-width bins
                bins = np.linspace(feature.min(), feature.max(), self.n_bins + 1)
            
            # Assign bins
            feature_binned = pd.cut(feature, bins=bins, include_lowest=True, duplicates='drop')
            
            # Calculate WoE for each bin
            woe_map = {}
            for bin_label in feature_binned.cat.categories:
                bin_mask = (feature_binned == bin_label)
                bin_data = target[bin_mask]
                
                if len(bin_data) < self.min_samples:
                    continue
                
                # Calculate % of events and non-events
                total_events = target.sum()
                total_non_events = len(target) - total_events
                
                if total_events == 0 or total_non_events == 0:
                    woe_value = 0.0
                else:
                    events_in_bin = bin_data.sum()
                    non_events_in_bin = len(bin_data) - events_in_bin
                    
                    pct_events = events_in_bin / total_events if total_events > 0 else 0
                    pct_non_events = non_events_in_bin / total_non_events if total_non_events > 0 else 0
                    
                    # Calculate WoE
                    if pct_events == 0 or pct_non_events == 0:
                        woe_value = 0.0
                    else:
                        woe_value = np.log(pct_non_events / pct_events)
                
                woe_map[bin_label] = woe_value
            
            return woe_map
        
        except Exception as e:
            logger.warning(f"Error calculating WoE for feature: {str(e)}")
            return {}
    
    def _get_woe_value(self, feature_name: str, value: float) -> float:
        """
        Get WoE value for a given feature value.
        
        Args:
            feature_name: Name of the feature
            value: Feature value
            
        Returns:
            WoE value
        """
        if feature_name not in self.woe_dict:
            return 0.0
        
        woe_map = self.woe_dict[feature_name]
        
        # Find the bin that contains this value
        for bin_label, woe_value in woe_map.items():
            if bin_label.left <= value <= bin_label.right:
                return woe_value
        
        # If value is outside all bins, return the closest WoE value
        if woe_map:
            return list(woe_map.values())[0]
        return 0.0


def calculate_information_value(X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate Information Value (IV) for each feature.
    
    IV = Σ((% of non-events - % of events) * WoE)
    
    IV Interpretation:
    - < 0.02: Not useful
    - 0.02 - 0.1: Weak predictive power
    - 0.1 - 0.3: Medium predictive power
    - 0.3 - 0.5: Strong predictive power
    - > 0.5: Suspicious (may be overfitted)
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_bins: Number of bins for discretization
        
    Returns:
        DataFrame with IV values for each feature
    """
    try:
        logger.info("Calculating Information Value (IV) for features...")
        
        iv_results = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                try:
                    # Create bins
                    try:
                        bins = pd.qcut(X[col], q=n_bins, duplicates='drop', retbins=True)[1]
                    except ValueError:
                        bins = np.linspace(X[col].min(), X[col].max(), n_bins + 1)
                    
                    # Assign bins
                    feature_binned = pd.cut(X[col], bins=bins, include_lowest=True, duplicates='drop')
                    
                    # Calculate IV
                    total_events = y.sum()
                    total_non_events = len(y) - total_events
                    
                    iv = 0.0
                    for bin_label in feature_binned.cat.categories:
                        bin_mask = (feature_binned == bin_label)
                        bin_data = y[bin_mask]
                        
                        if len(bin_data) == 0:
                            continue
                        
                        events_in_bin = bin_data.sum()
                        non_events_in_bin = len(bin_data) - events_in_bin
                        
                        pct_events = events_in_bin / total_events if total_events > 0 else 0
                        pct_non_events = non_events_in_bin / total_non_events if total_non_events > 0 else 0
                        
                        if pct_events > 0 and pct_non_events > 0:
                            woe = np.log(pct_non_events / pct_events)
                            iv += (pct_non_events - pct_events) * woe
                    
                    # Interpret IV
                    if iv < 0.02:
                        strength = "Not useful"
                    elif iv < 0.1:
                        strength = "Weak"
                    elif iv < 0.3:
                        strength = "Medium"
                    elif iv < 0.5:
                        strength = "Strong"
                    else:
                        strength = "Suspicious"
                    
                    iv_results.append({
                        'Feature': col,
                        'IV': iv,
                        'Strength': strength
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating IV for {col}: {str(e)}")
                    iv_results.append({
                        'Feature': col,
                        'IV': 0.0,
                        'Strength': 'Error'
                    })
        
        iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
        logger.info(f"\nInformation Value Summary:\n{iv_df.to_string()}")
        
        return iv_df
    
    except Exception as e:
        logger.error(f"Error calculating Information Value: {str(e)}")
        raise

