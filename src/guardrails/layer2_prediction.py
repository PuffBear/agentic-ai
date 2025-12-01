"""
Layer 2: Prediction Validation Guardrail
Validates model predictions for consistency, confidence, and hallucinations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class PredictionValidationGuardrail:
    """
    Layer 2: Prediction Validation
    
    Validates:
    - Cross-model consistency (hallucination detection)
    - Confidence thresholds
    - Anomaly detection in predictions
    - Prediction distribution sanity checks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize prediction validation guardrail
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.model_agreement_threshold = self.config.get('model_agreement_threshold', 0.8)
        self.max_entropy_threshold = self.config.get('max_entropy_threshold', 1.5)
        
        # Expected classes
        self.expected_classes = ['High', 'Medium', 'Low']
        
        logger.info("Layer 2: Prediction Validation Guardrail initialized")
    
    def validate(self, prediction_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Main validation method for predictions
        
        Args:
            prediction_data: Dictionary containing:
                - predictions: Array of predictions
                - probabilities: Probability distributions (optional)
                - confidence: Confidence scores (optional)
                - individual_predictions: Dict of predictions from each model (optional)
                
        Returns:
            Tuple of (is_valid, error_message, validation_details)
        """
        validation_details = {
            'confidence_valid': False,
            'consistency_valid': False,
            'distribution_valid': False,
            'hallucination_detected': False,
            'errors': [],
            'warnings': []
        }
        
        # Extract data
        predictions = prediction_data.get('predictions')
        probabilities = prediction_data.get('probabilities')
        confidence = prediction_data.get('confidence')
        individual_preds = prediction_data.get('individual_predictions')
        
        if predictions is None:
            error_msg = "No predictions provided"
            validation_details['errors'].append(error_msg)
            return False, error_msg, validation_details
        
        # 1. Validate prediction format
        format_valid, format_msg = self._validate_format(predictions, probabilities)
        if not format_valid:
            validation_details['errors'].append(format_msg)
            return False, format_msg, validation_details
        
        # 2. Validate confidence scores
        if confidence is not None:
            conf_valid, conf_msg = self._validate_confidence(confidence)
            validation_details['confidence_valid'] = conf_valid
            if not conf_valid:
                validation_details['errors'].append(conf_msg)
                return False, conf_msg, validation_details
        else:
            # Calculate confidence from probabilities if available
            if probabilities is not None:
                confidence = np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else np.array([np.max(probabilities)])
                conf_valid, conf_msg = self._validate_confidence(confidence)
                validation_details['confidence_valid'] = conf_valid
                if not conf_valid:
                    validation_details['warnings'].append(conf_msg)
        
        # 3. Check cross-model consistency (hallucination detection)
        if individual_preds is not None:
            consistency_valid, consistency_msg, hallucination_rate = self._validate_consistency(
                individual_preds, predictions
            )
            validation_details['consistency_valid'] = consistency_valid
            validation_details['hallucination_rate'] = hallucination_rate
            
            if not consistency_valid:
                validation_details['hallucination_detected'] = True
                validation_details['errors'].append(consistency_msg)
                logger.warning(f"Hallucination detected: {consistency_msg}")
                return False, consistency_msg, validation_details
        
        # 4. Validate prediction distribution
        dist_valid, dist_msg = self._validate_distribution(predictions, probabilities)
        validation_details['distribution_valid'] = dist_valid
        if not dist_valid:
            validation_details['warnings'].append(dist_msg)
        
        # 5. Check for anomalous predictions
        anomaly_detected, anomaly_msg = self._detect_anomalies(predictions, probabilities)
        if anomaly_detected:
            validation_details['warnings'].append(anomaly_msg)
            logger.warning(f"Prediction anomaly: {anomaly_msg}")
        
        # All critical checks passed
        logger.info("✓ Prediction validation passed")
        return True, "Prediction validation passed", validation_details
    
    def _validate_format(self, predictions: np.ndarray, 
                        probabilities: np.ndarray = None) -> Tuple[bool, str]:
        """
        Validate prediction format and structure
        
        Args:
            predictions: Prediction array
            probabilities: Probability array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if predictions are valid
        if not isinstance(predictions, (np.ndarray, list)):
            return False, "Predictions must be numpy array or list"
        
        predictions = np.array(predictions)
        
        # Check for NaN or infinite values (only for numeric types)
        if predictions.dtype.kind in ['i', 'f', 'c']:  # integer, float, or complex
            if np.any(np.isnan(predictions)):
                return False, "Predictions contain NaN values"
        
        # Validate prediction classes
        unique_preds = np.unique(predictions)
        invalid_classes = [p for p in unique_preds if p not in self.expected_classes]
        if invalid_classes:
            return False, f"Invalid prediction classes: {invalid_classes}"
        
        # Validate probabilities if provided
        if probabilities is not None:
            probabilities = np.array(probabilities)
            
            # Check shape
            if len(probabilities.shape) == 2:
                if probabilities.shape[1] != len(self.expected_classes):
                    return False, f"Probability shape mismatch: expected {len(self.expected_classes)} classes"
                
                # Check if probabilities sum to 1
                prob_sums = np.sum(probabilities, axis=1)
                if not np.allclose(prob_sums, 1.0, rtol=1e-2):
                    return False, "Probabilities don't sum to 1"
            
            # Check for NaN or negative values
            if np.any(np.isnan(probabilities)) or np.any(probabilities < 0):
                return False, "Probabilities contain invalid values"
        
        return True, "Format validation passed"
    
    def _validate_confidence(self, confidence: np.ndarray) -> Tuple[bool, str]:
        """
        Validate confidence scores
        
        Args:
            confidence: Confidence score array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        confidence = np.array(confidence)
        
        # Check for invalid values
        if np.any(np.isnan(confidence)) or np.any(confidence < 0) or np.any(confidence > 1):
            return False, "Confidence scores contain invalid values (must be in [0, 1])"
        
        # Check average confidence
        avg_confidence = np.mean(confidence)
        
        if avg_confidence < self.confidence_threshold:
            return False, f"Average confidence {avg_confidence:.3f} below threshold {self.confidence_threshold}"
        
        # Check for too many low-confidence predictions
        low_confidence_ratio = (confidence < self.confidence_threshold).mean()
        if low_confidence_ratio > 0.3:  # More than 30% low confidence
            logger.warning(f"High proportion of low-confidence predictions: {low_confidence_ratio:.2%}")
        
        return True, "Confidence validation passed"
    
    def _validate_consistency(self, individual_preds: Dict[str, np.ndarray],
                            ensemble_pred: np.ndarray) -> Tuple[bool, str, float]:
        """
        Validate cross-model consistency (hallucination detection)
        
        Args:
            individual_preds: Dictionary of predictions from each model
            ensemble_pred: Ensemble prediction
            
        Returns:
            Tuple of (is_valid, error_message, hallucination_rate)
        """
        # Extract individual predictions
        pred_arrays = list(individual_preds.values())
        
        if len(pred_arrays) < 2:
            return True, "Not enough models for consistency check", 0.0
        
        # Check agreement between all models
        pred_matrix = np.array(pred_arrays)
        
        # Calculate agreement rate (all models agree)
        full_agreement = np.all(pred_matrix == pred_matrix[0], axis=0)
        agreement_rate = full_agreement.mean()
        hallucination_rate = 1 - agreement_rate
        
        # Check if agreement rate meets threshold
        if agreement_rate < self.model_agreement_threshold:
            error_msg = (f"Low model agreement: {agreement_rate:.2%} "
                        f"(threshold: {self.model_agreement_threshold:.0%}). "
                        f"Hallucination rate: {hallucination_rate:.2%}")
            return False, error_msg, hallucination_rate
        
        # Check for systematic disagreements
        disagreement_indices = np.where(~full_agreement)[0]
        if len(disagreement_indices) > 0:
            logger.info(f"Models disagree on {len(disagreement_indices)} predictions")
        
        return True, "Consistency validation passed", hallucination_rate
    
    def _validate_distribution(self, predictions: np.ndarray,
                              probabilities: np.ndarray = None) -> Tuple[bool, str]:
        """
        Validate prediction distribution sanity
        
        Args:
            predictions: Prediction array
            probabilities: Probability array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        predictions = np.array(predictions)
        
        # Check class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts / len(predictions)))
        
        # Check for extreme imbalances (>95% one class)
        max_class_ratio = max(distribution.values())
        if max_class_ratio > 0.95:
            msg = f"Extreme class imbalance: {max_class_ratio:.1%} in one class"
            logger.warning(msg)
            return False, msg
        
        # Check if all classes are represented (if we have enough samples)
        if len(predictions) > 10:
            missing_classes = set(self.expected_classes) - set(unique)
            if missing_classes:
                msg = f"Missing classes in predictions: {missing_classes}"
                logger.warning(msg)
        
        # Check entropy of probabilities if available
        if probabilities is not None:
            probabilities = np.array(probabilities)
            if len(probabilities.shape) == 2:
                # Calculate entropy
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                avg_entropy = np.mean(entropy)
                
                # High entropy = uncertain predictions
                if avg_entropy > self.max_entropy_threshold:
                    msg = f"High prediction entropy: {avg_entropy:.3f} (threshold: {self.max_entropy_threshold})"
                    logger.warning(msg)
        
        return True, "Distribution validation passed"
    
    def _detect_anomalies(self, predictions: np.ndarray,
                         probabilities: np.ndarray = None) -> Tuple[bool, str]:
        """
        Detect anomalies in predictions
        
        Args:
            predictions: Prediction array
            probabilities: Probability array
            
        Returns:
            Tuple of (anomaly_detected, message)
        """
        anomalies = []
        
        # Check for sudden changes in batch predictions
        if len(predictions) > 10:
            # Check if distribution changes drastically within batch
            mid_point = len(predictions) // 2
            first_half = predictions[:mid_point]
            second_half = predictions[mid_point:]
            
            dist1 = pd.Series(first_half).value_counts(normalize=True)
            dist2 = pd.Series(second_half).value_counts(normalize=True)
            
            # Calculate distribution difference
            all_classes = set(dist1.index) | set(dist2.index)
            diff = sum(abs(dist1.get(c, 0) - dist2.get(c, 0)) for c in all_classes)
            
            if diff > 0.4:  # >40% distribution change
                anomalies.append(f"Distribution shift within batch: {diff:.2%}")
        
        # Check for extreme probabilities (all near 1.0 or all near 0.33)
        if probabilities is not None:
            probabilities = np.array(probabilities)
            if len(probabilities.shape) == 2:
                max_probs = np.max(probabilities, axis=1)
                
                # All predictions very certain
                if np.mean(max_probs > 0.99) > 0.9:
                    anomalies.append("Suspiciously high confidence across all predictions")
                
                # All predictions very uncertain
                if np.mean(max_probs < 0.4) > 0.9:
                    anomalies.append("Suspiciously low confidence across all predictions")
        
        if anomalies:
            return True, "; ".join(anomalies)
        
        return False, "No anomalies detected"
    
    def get_validation_report(self, validation_details: Dict[str, Any]) -> str:
        """
        Generate human-readable validation report
        
        Args:
            validation_details: Validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PREDICTION VALIDATION REPORT (Layer 2)")
        report.append("=" * 60)
        report.append(f"Confidence Valid: {validation_details.get('confidence_valid', 'N/A')}")
        report.append(f"Consistency Valid: {validation_details.get('consistency_valid', 'N/A')}")
        report.append(f"Distribution Valid: {validation_details.get('distribution_valid', 'N/A')}")
        report.append(f"Hallucination Detected: {validation_details.get('hallucination_detected', 'N/A')}")
        
        if 'hallucination_rate' in validation_details:
            report.append(f"Hallucination Rate: {validation_details['hallucination_rate']:.2%}")
        
        if validation_details.get('errors'):
            report.append("\nErrors:")
            for error in validation_details['errors']:
                report.append(f"  - {error}")
        
        if validation_details.get('warnings'):
            report.append("\nWarnings:")
            for warning in validation_details['warnings']:
                report.append(f"  - {warning}")
        
        report.append("=" * 60)
        return "\n".join(report)
