"""
Drift Detector - Monitors model drift using statistical tests
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects model and data drift using:
    - Kolmogorov-Smirnov test (KS test)
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    - Performance degradation monitoring
    """
    
    def __init__(self, 
                 ks_threshold: float = 0.05,
                 psi_threshold: float = 0.2,
                 performance_threshold: float = 0.1):
        """
        Initialize drift detector
        
        Args:
            ks_threshold: P-value threshold for KS test (default: 0.05)
            psi_threshold: PSI threshold (default: 0.2, >0.25 = significant drift)
            performance_threshold: Maximum allowed performance drop (default: 0.1)
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.performance_threshold = performance_threshold
        
        # Store reference distributions
        self.reference_data = None
        self.reference_predictions = None
        self.baseline_performance = None
        
        logger.info("Drift detector initialized")
    
    def set_reference(self, 
                     X_reference: np.ndarray,
                     predictions_reference: np.ndarray = None,
                     performance_reference: float = None):
        """
        Set reference/baseline data for drift detection
        
        Args:
            X_reference: Reference feature data
            predictions_reference: Reference model predictions
            performance_reference: Baseline model performance (e.g., accuracy)
        """
        self.reference_data = X_reference
        self.reference_predictions = predictions_reference
        self.baseline_performance = performance_reference
        
        logger.info(f"Reference data set: {len(X_reference)} samples")
    
    def detect_drift(self, 
                    X_current: np.ndarray,
                    predictions_current: np.ndarray = None,
                    performance_current: float = None) -> Dict[str, Any]:
        """
        Detect drift in features, predictions, and performance
        
        Args:
            X_current: Current feature data
            predictions_current: Current model predictions
            performance_current: Current model performance
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        results = {
            'drift_detected': False,
            'feature_drift': {},
            'prediction_drift': None,
            'performance_drift': None,
            'timestamp': pd.Timestamp.now()
        }
        
        # 1. Feature drift (KS test for each feature)
        logger.info("Checking feature drift...")
        feature_drift_results = self._detect_feature_drift(X_current)
        results['feature_drift'] = feature_drift_results
        
        # Check if any feature has significant drift
        drifted_features = [
            f for f, res in feature_drift_results.items() 
            if res['drift_detected']
        ]
        
        if drifted_features:
            results['drift_detected'] = True
            logger.warning(f"Drift detected in {len(drifted_features)} features: {drifted_features[:5]}")
        
        # 2. Prediction drift (PSI)
        if predictions_current is not None and self.reference_predictions is not None:
            logger.info("Checking prediction drift...")
            prediction_drift = self._detect_prediction_drift(predictions_current)
            results['prediction_drift'] = prediction_drift
            
            if prediction_drift['drift_detected']:
                results['drift_detected'] = True
                logger.warning(f"Prediction drift detected: PSI = {prediction_drift['psi']:.4f}")
        
        # 3. Performance drift
        if performance_current is not None and self.baseline_performance is not None:
            logger.info("Checking performance drift...")
            performance_drift = self._detect_performance_drift(performance_current)
            results['performance_drift'] = performance_drift
            
            if performance_drift['drift_detected']:
                results['drift_detected'] = True
                logger.warning(f"Performance drift detected: drop = {performance_drift['drop']:.4f}")
        
        return results
    
    def _detect_feature_drift(self, X_current: np.ndarray) -> Dict[str, Dict]:
        """
        Detect drift in individual features using KS test
        
        Args:
            X_current: Current feature data
            
        Returns:
            Dictionary with drift results for each feature
        """
        n_features = X_current.shape[1]
        results = {}
        
        for i in range(n_features):
            feature_name = f"feature_{i}"
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                X_current[:, i]
            )
            
            drift_detected = p_value < self.ks_threshold
            
            results[feature_name] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected
            }
        
        return results
    
    def _detect_prediction_drift(self, predictions_current: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in predictions using PSI
        
        Args:
            predictions_current: Current predictions
            
        Returns:
            Dictionary with PSI and drift status
        """
        # Calculate PSI
        psi = self._calculate_psi(
            self.reference_predictions,
            predictions_current
        )
        
        drift_detected = psi > self.psi_threshold
        
        return {
            'psi': float(psi),
            'threshold': self.psi_threshold,
            'drift_detected': drift_detected
        }
    
    def _detect_performance_drift(self, performance_current: float) -> Dict[str, Any]:
        """
        Detect performance degradation
        
        Args:
            performance_current: Current model performance
            
        Returns:
            Dictionary with performance drift results
        """
        performance_drop = self.baseline_performance - performance_current
        drift_detected = performance_drop > self.performance_threshold
        
        return {
            'baseline_performance': self.baseline_performance,
            'current_performance': performance_current,
            'drop': float(performance_drop),
            'threshold': self.performance_threshold,
            'drift_detected': drift_detected
        }
    
    def _calculate_psi(self, 
                       reference: np.ndarray, 
                       current: np.ndarray,
                       bins: int = 10) -> float:
        """
        Calculate Population Stability Index
        
        Args:
            reference: Reference predictions
            current: Current predictions
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # For classification, use value counts
        if reference.dtype in [np.int32, np.int64, object, str]:
            ref_counts = pd.Series(reference).value_counts(normalize=True)
            curr_counts = pd.Series(current).value_counts(normalize=True)
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            psi = 0
            for cat in all_categories:
                ref_pct = ref_counts.get(cat, 1e-10)
                curr_pct = curr_counts.get(cat, 1e-10)
                psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
            
            return psi
        
        # For continuous predictions, bin them
        else:
            # Create bins based on reference distribution
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Get distributions
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            curr_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize
            ref_pct = ref_hist / len(reference) + 1e-10
            curr_pct = curr_hist / len(current) + 1e-10
            
            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            return psi
    
    def calculate_js_divergence(self, 
                                reference: np.ndarray, 
                                current: np.ndarray,
                                bins: int = 10) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins
            
        Returns:
            JS divergence value
        """
        # Create histograms
        _, bin_edges = np.histogram(reference, bins=bins)
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to probabilities
        ref_prob = ref_hist / len(reference) + 1e-10
        curr_prob = curr_hist / len(current) + 1e-10
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, curr_prob)
        
        return float(js_div)
    
    def get_drift_report(self, drift_results: Dict[str, Any]) -> str:
        """
        Generate human-readable drift report
        
        Args:
            drift_results: Results from detect_drift()
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DRIFT DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {drift_results['timestamp']}")
        report.append(f"Overall Drift Detected: {drift_results['drift_detected']}")
        report.append("")
        
        # Feature drift
        feature_drift = drift_results['feature_drift']
        drifted_features = [
            f for f, res in feature_drift.items() 
            if res['drift_detected']
        ]
        
        report.append(f"Feature Drift: {len(drifted_features)} / {len(feature_drift)} features")
        if drifted_features:
            report.append("Drifted features:")
            for feat in drifted_features[:10]:  # Show top 10
                res = feature_drift[feat]
                report.append(f"  - {feat}: KS={res['ks_statistic']:.4f}, p={res['p_value']:.4f}")
        
        report.append("")
        
        # Prediction drift
        if drift_results['prediction_drift']:
            pred_drift = drift_results['prediction_drift']
            report.append(f"Prediction Drift (PSI): {pred_drift['psi']:.4f}")
            report.append(f"  Threshold: {pred_drift['threshold']}")
            report.append(f"  Drift Detected: {pred_drift['drift_detected']}")
            report.append("")
        
        # Performance drift
        if drift_results['performance_drift']:
            perf_drift = drift_results['performance_drift']
            report.append(f"Performance Drift:")
            report.append(f"  Baseline: {perf_drift['baseline_performance']:.4f}")
            report.append(f"  Current: {perf_drift['current_performance']:.4f}")
            report.append(f"  Drop: {perf_drift['drop']:.4f}")
            report.append(f"  Drift Detected: {perf_drift['drift_detected']}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
