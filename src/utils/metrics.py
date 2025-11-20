"""
Evaluation Metrics
Comprehensive metrics for model evaluation and monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from loguru import logger

class MetricsCalculator:
    """Calculate and track various evaluation metrics"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics_history = []
        logger.info("MetricsCalculator initialized")
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)
            average: Averaging method for multi-class metrics
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC AUC (if probabilities provided)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba,
                    multi_class='ovr',
                    average=average
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            metrics[f'precision_class_{cls}'] = per_class_precision[i]
            metrics[f'recall_class_{cls}'] = per_class_recall[i]
            metrics[f'f1_class_{cls}'] = per_class_f1[i]
        
        logger.info(f"Classification metrics calculated: {metrics}")
        
        return metrics
    
    def calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
        
        Returns:
            Tuple of (confusion matrix array, confusion matrix DataFrame)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = np.unique(y_true)
        
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels]
        )
        
        logger.info("Confusion matrix calculated")
        
        return cm, cm_df
    
    def calculate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None
    ) -> Dict:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
        
        Returns:
            Classification report as dictionary
        """
        report = classification_report(
            y_true, y_pred,
            target_names=labels,
            output_dict=True
        )
        
        logger.info("Classification report generated")
        
        return report
    
    def track_metrics(
        self,
        metrics: Dict[str, float],
        stage: str = "training",
        iteration: Optional[int] = None
    ) -> None:
        """
        Track metrics over time
        
        Args:
            metrics: Dictionary of metrics
            stage: Stage of pipeline (training, validation, test)
            iteration: Iteration number (optional)
        """
        entry = {
            'stage': stage,
            'iteration': iteration,
            **metrics
        }
        
        self.metrics_history.append(entry)
        logger.info(f"Tracked metrics for {stage} stage")
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get historical metrics as DataFrame
        
        Returns:
            DataFrame with all tracked metrics
        """
        if not self.metrics_history:
            logger.warning("No metrics history available")
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def calculate_confidence_metrics(
        self,
        y_proba: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate confidence-related metrics
        
        Args:
            y_proba: Predicted probabilities
            confidence_threshold: Threshold for high-confidence predictions
        
        Returns:
            Dictionary of confidence metrics
        """
        # Get max probability for each prediction
        max_probs = np.max(y_proba, axis=1)
        
        metrics = {
            'mean_confidence': np.mean(max_probs),
            'median_confidence': np.median(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'high_confidence_ratio': np.mean(max_probs >= confidence_threshold),
            'confidence_std': np.std(max_probs)
        }
        
        logger.info(f"Confidence metrics: mean={metrics['mean_confidence']:.3f}, high_conf_ratio={metrics['high_confidence_ratio']:.3f}")
        
        return metrics
    
    def calculate_ensemble_agreement(
        self,
        predictions: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate agreement between multiple model predictions
        
        Args:
            predictions: List of prediction arrays from different models
        
        Returns:
            Dictionary of agreement metrics
        """
        predictions_array = np.array(predictions)
        
        # Calculate agreement rate
        agreement_mask = np.all(predictions_array == predictions_array[0], axis=0)
        agreement_rate = np.mean(agreement_mask)
        
        # Calculate pairwise agreement
        n_models = len(predictions)
        pairwise_agreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pairwise_agreement = np.mean(predictions[i] == predictions[j])
                pairwise_agreements.append(pairwise_agreement)
        
        metrics = {
            'full_agreement_rate': agreement_rate,
            'mean_pairwise_agreement': np.mean(pairwise_agreements),
            'min_pairwise_agreement': np.min(pairwise_agreements),
            'max_pairwise_agreement': np.max(pairwise_agreements)
        }
        
        logger.info(f"Ensemble agreement metrics: full_agreement={metrics['full_agreement_rate']:.3f}")
        
        return metrics