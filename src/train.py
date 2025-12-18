"""
Model Training Module

This module provides functions for training credit risk models with MLflow tracking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any, Optional
import os

# Try importing XGBoost and LightGBM (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "credit-risk-modeling") -> None:
    """
    Setup MLflow experiment tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment '{experiment_name}' set up successfully")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise


def train_logistic_regression(X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              random_state: int = 42) -> Dict[str, Any]:
    """
    Train Logistic Regression model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Logistic Regression model...")
        
        with mlflow.start_run(run_name="logistic_regression"):
            # Train model
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'LogisticRegression',
                'random_state': random_state,
                'max_iter': 1000
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Logistic Regression: {str(e)}")
        raise


def train_decision_tree(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       random_state: int = 42,
                       max_depth: Optional[int] = None) -> Dict[str, Any]:
    """
    Train Decision Tree model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        max_depth: Maximum depth of the tree
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Decision Tree model...")
        
        with mlflow.start_run(run_name="decision_tree"):
            # Train model
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'DecisionTree',
                'random_state': random_state,
                'max_depth': max_depth
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Decision Tree: {str(e)}")
        raise


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        random_state: int = 42,
                        n_estimators: int = 100) -> Dict[str, Any]:
    """
    Train Random Forest model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        n_estimators: Number of trees
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Random Forest model...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'RandomForest',
                'random_state': random_state,
                'n_estimators': n_estimators
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise


def train_gradient_boosting(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           random_state: int = 42,
                           n_estimators: int = 100,
                           learning_rate: float = 0.1) -> Dict[str, Any]:
    """
    Train Gradient Boosting model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        random_state: Random state for reproducibility
        n_estimators: Number of boosting stages
        learning_rate: Learning rate
        
    Returns:
        Dictionary with model and metrics
    """
    try:
        logger.info("Training Gradient Boosting model...")
        
        with mlflow.start_run(run_name="gradient_boosting"):
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
            }
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'GradientBoosting',
                'random_state': random_state,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Model trained. Metrics: {metrics}")
            return {'model': model, 'metrics': metrics}
    
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {str(e)}")
        raise


def hyperparameter_tuning_grid_search(model, param_grid: Dict, 
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     cv: int = 5,
                                     scoring: str = 'roc_auc') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Grid Search.
    
    Args:
        model: Model instance
        param_grid: Dictionary of hyperparameters to search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with best model and results
    """
    try:
        logger.info(f"Performing Grid Search for {type(model).__name__}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    except Exception as e:
        logger.error(f"Error in Grid Search: {str(e)}")
        raise


def hyperparameter_tuning_random_search(model, param_distributions: Dict,
                                      X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      n_iter: int = 50,
                                      cv: int = 5,
                                      scoring: str = 'roc_auc',
                                      random_state: int = 42) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Random Search.
    
    Args:
        model: Model instance
        param_distributions: Dictionary of hyperparameter distributions
        X_train: Training features
        y_train: Training target
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        scoring: Scoring metric
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with best model and results
    """
    try:
        logger.info(f"Performing Random Search for {type(model).__name__}...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return {
            'best_model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    except Exception as e:
        logger.error(f"Error in Random Search: {str(e)}")
        raise


def register_best_model(model, model_name: str, metrics: Dict[str, float],
                       stage: str = "Production") -> None:
    """
    Register the best model in MLflow Model Registry.
    
    Args:
        model: Trained model
        model_name: Name for the model in registry
        metrics: Model metrics
        stage: Model stage (Production, Staging, etc.)
    """
    try:
        logger.info(f"Registering model '{model_name}' to MLflow Model Registry...")
        
        # Get the current run
        run_id = mlflow.active_run().info.run_id
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Transition to stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage=stage
        )
        
        logger.info(f"Model '{model_name}' registered successfully (version {registered_model.version})")
        logger.info(f"Model transitioned to '{stage}' stage")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise


def train_models(data_path: str,
                 target_col: str = 'is_high_risk',
                 test_size: float = 0.2,
                 random_state: int = 42) -> Dict[str, Any]:
    """
    Main training function that loads data, processes it, and trains multiple models.
    
    Args:
        data_path: Path to processed data CSV
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with trained models and results
    """
    try:
        logger.info("Starting model training pipeline...")
        
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        try:
            from src.data_processing import load_data
        except ImportError:
            # Fallback for relative imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.data_processing import load_data
        
        df = load_data(data_path)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Train models
        results = {}
        
        # Logistic Regression
        lr_results = train_logistic_regression(X_train, y_train, X_test, y_test, random_state)
        results['logistic_regression'] = lr_results
        
        # Decision Tree
        dt_results = train_decision_tree(X_train, y_train, X_test, y_test, random_state)
        results['decision_tree'] = dt_results
        
        # Random Forest
        rf_results = train_random_forest(X_train, y_train, X_test, y_test, random_state)
        results['random_forest'] = rf_results
        
        # Gradient Boosting
        gb_results = train_gradient_boosting(X_train, y_train, X_test, y_test, random_state)
        results['gradient_boosting'] = gb_results
        
        # Find best model based on ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]['metrics']
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        # Register best model
        try:
            with mlflow.start_run(run_name=f"best_model_{best_model_name}"):
                mlflow.log_params({
                    'best_model_type': best_model_name,
                    'best_roc_auc': best_metrics['roc_auc']
                })
                mlflow.log_metrics(best_metrics)
                mlflow.sklearn.log_model(best_model, "model")
                register_best_model(best_model, "CreditRiskModel", best_metrics)
        except Exception as e:
            logger.warning(f"Could not register model in registry: {str(e)}")
        
        logger.info("Model training completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
