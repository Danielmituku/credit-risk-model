"""
Complete Pipeline Script

This script orchestrates the entire credit risk modeling pipeline:
1. Load raw data
2. Feature engineering
3. RFM analysis and proxy target creation
4. Data preprocessing
5. Model training
"""

import pandas as pd
import logging
from pathlib import Path

from src.data_processing import load_data, process_data, apply_woe_transformation
from src.rfm_analysis import create_rfm_target_variable
from src.train import train_models, setup_mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    raw_data_path: str,
    output_data_path: str = "data/processed/processed_data.csv",
    use_woe: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Run the complete credit risk modeling pipeline.
    
    Args:
        raw_data_path: Path to raw transaction data
        output_data_path: Path to save processed data
        use_woe: Whether to apply WoE transformation
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
    """
    try:
        logger.info("=" * 80)
        logger.info("CREDIT RISK MODELING PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load raw data
        logger.info("\n[Step 1/5] Loading raw data...")
        df_raw = load_data(raw_data_path)
        logger.info(f"Loaded {len(df_raw)} transactions")
        
        # Step 2: Feature engineering
        logger.info("\n[Step 2/5] Feature engineering...")
        df_features, _ = process_data(df_raw)
        logger.info(f"Created {len(df_features.columns)} features")
        
        # Step 3: RFM analysis and proxy target creation
        logger.info("\n[Step 3/5] Creating RFM-based proxy target variable...")
        rfm_df, target_df = create_rfm_target_variable(
            df_raw,
            n_clusters=3,
            random_state=random_state
        )
        
        # Merge target with features
        df_processed = df_features.merge(
            target_df[['CustomerId', 'is_high_risk']],
            on='CustomerId',
            how='left'
        )
        
        # Check target distribution
        target_dist = df_processed['is_high_risk'].value_counts()
        logger.info(f"Target distribution: {dict(target_dist)}")
        
        # Step 4: Save processed data
        logger.info("\n[Step 4/5] Saving processed data...")
        Path(output_data_path).parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_data_path, index=False)
        logger.info(f"Processed data saved to {output_data_path}")
        
        # Step 5: Model training
        logger.info("\n[Step 5/5] Training models...")
        setup_mlflow("credit-risk-modeling")
        results = train_models(
            output_data_path,
            target_col='is_high_risk',
            test_size=test_size,
            random_state=random_state
        )
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nModel Performance Summary:")
        for model_name, result in results.items():
            metrics = result['metrics']
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info("\n✓ Pipeline execution completed!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        raw_data_path = sys.argv[1]
    else:
        raw_data_path = "data/raw/data.csv"
    
    run_complete_pipeline(raw_data_path)

