"""
Monitoring Agent (Agent 5)
Detects drift, monitors performance, triggers alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
from scipy import stats
from .base_agent import BaseAgent

class MonitoringAgent(BaseAgent):
    """Agent 5: Monitors system health and performance"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        super().__init__(
            agent_name="monitoring_agent",
            config_path=config_path
        )
        
        # Store baseline statistics
        self.baseline_stats = None
        self.performance_history = []
        self.drift_alerts = []
        
        # Thresholds
        self.drift_threshold = 0.05  # 5% significance level
        self.performance_threshold = 0.85  # Alert if accuracy drops below 85%
        
        self.logger.info("MonitoringAgent initialized")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input"""
        required_keys = ['mode']
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        mode = input_data['mode']
        if mode not in ['set_baseline', 'check_drift', 'monitor_performance', 'get_alerts']:
            self.logger.error(f"Invalid mode: {mode}")
            return False
        
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output"""
        return True
    
    def _calculate_feature_stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate feature statistics"""
        return {
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
            'min': X.min(axis=0).tolist(),
            'max': X.max(axis=0).tolist(),
            'n_samples': len(X)
        }
    
    def _detect_drift(
        self,
        baseline_data: np.ndarray,
        current_data: np.ndarray
    ) -> Tuple[bool, List[int], float]:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            baseline_data: Reference data
            current_data: New data to check
        
        Returns:
            (drift_detected, drifted_features, max_drift_score)
        """
        n_features = baseline_data.shape[1]
        drifted_features = []
        drift_scores = []
        
        for i in range(n_features):
            # KS test for each feature
            statistic, p_value = stats.ks_2samp(
                baseline_data[:, i],
                current_data[:, i]
            )
            
            drift_scores.append(statistic)
            
            # If p-value < threshold, drift detected
            if p_value < self.drift_threshold:
                drifted_features.append(i)
        
        drift_detected = len(drifted_features) > 0
        max_drift_score = max(drift_scores) if drift_scores else 0.0
        
        return drift_detected, drifted_features, max_drift_score
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing logic"""
        mode = input_data['mode']
        
        if mode == 'set_baseline':
            # Set baseline statistics from training data
            X = input_data['X']
            y = input_data.get('y', None)
            
            self.baseline_stats = {
                'feature_stats': self._calculate_feature_stats(X),
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X)
            }
            
            if y is not None:
                unique, counts = np.unique(y, return_counts=True)
                self.baseline_stats['target_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
            self.logger.info(f"Baseline set with {len(X)} samples")
            
            return {
                'mode': 'set_baseline',
                'baseline_set': True,
                'n_samples': len(X)
            }
        
        elif mode == 'check_drift':
            # Check for data drift
            if self.baseline_stats is None:
                return {
                    'mode': 'check_drift',
                    'error': 'Baseline not set. Call set_baseline first.'
                }
            
            X_new = input_data['X']
            X_baseline = input_data.get('X_baseline', None)
            
            if X_baseline is None:
                return {
                    'mode': 'check_drift',
                    'error': 'Need baseline data for comparison'
                }
            
            # Detect drift
            drift_detected, drifted_features, max_drift_score = self._detect_drift(
                X_baseline,
                X_new
            )
            
            drift_result = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': drift_detected,
                'drifted_features': drifted_features,
                'n_drifted_features': len(drifted_features),
                'max_drift_score': max_drift_score,
                'drift_percentage': (len(drifted_features) / X_new.shape[1] * 100)
            }
            
            # Log alert if drift detected
            if drift_detected:
                alert = {
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH' if len(drifted_features) > X_new.shape[1] * 0.3 else 'MEDIUM',
                    'message': f'Drift detected in {len(drifted_features)} features',
                    **drift_result
                }
                self.drift_alerts.append(alert)
                self.logger.warning(f"⚠️ DATA DRIFT: {len(drifted_features)} features drifted")
            
            return {
                'mode': 'check_drift',
                **drift_result
            }
        
        elif mode == 'monitor_performance':
            # Monitor model performance
            y_true = input_data['y_true']
            y_pred = input_data['y_pred']
            confidence = input_data.get('confidence', None)
            
            # Calculate metrics
            accuracy = (y_true == y_pred).mean()
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'n_samples': len(y_true),
                'avg_confidence': confidence.mean() if confidence is not None else None
            }
            
            self.performance_history.append(performance_record)
            
            # Check for performance degradation
            if accuracy < self.performance_threshold:
                alert = {
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'HIGH',
                    'message': f'Accuracy dropped to {accuracy:.2%}',
                    **performance_record
                }
                self.drift_alerts.append(alert)
                self.logger.warning(f"⚠️ PERFORMANCE ALERT: Accuracy = {accuracy:.2%}")
            
            # Recommend retraining if performance dropped significantly
            recommend_retrain = False
            if len(self.performance_history) > 1:
                recent_avg = np.mean([p['accuracy'] for p in self.performance_history[-5:]])
                if recent_avg < self.performance_threshold:
                    recommend_retrain = True
            
            return {
                'mode': 'monitor_performance',
                'performance': performance_record,
                'recommend_retrain': recommend_retrain,
                'total_records': len(self.performance_history)
            }
        
        elif mode == 'get_alerts':
            # Return all alerts
            return {
                'mode': 'get_alerts',
                'alerts': self.drift_alerts,
                'total_alerts': len(self.drift_alerts)
            }
