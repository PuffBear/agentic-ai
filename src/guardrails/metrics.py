"""
Guardrail Metrics - Track and analyze guardrail performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class GuardrailMetrics:
    """
    Track and analyze guardrail performance metrics:
    - False positive/negative rates
    - Layer-by-layer effectiveness
    - Response times
    - Violation patterns
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.validation_history = []
        self.violation_history = []
        self.performance_metrics = {
            'layer1': {'total': 0, 'passed': 0, 'failed': 0, 'response_times': []},
            'layer2': {'total': 0, 'passed': 0, 'failed': 0, 'response_times': []},
            'layer3': {'total': 0, 'passed': 0, 'failed': 0, 'response_times': []}
        }
        
        logger.info("Guardrail metrics tracker initialized")
    
    def record_validation(self, layer: str, is_valid: bool, 
                         response_time: float, details: Dict[str, Any] = None):
        """
        Record a validation event
        
        Args:
            layer: 'layer1', 'layer2', or 'layer3'
            is_valid: Whether validation passed
            response_time: Time taken for validation (seconds)
            details: Additional validation details
        """
        if layer not in self.performance_metrics:
            logger.warning(f"Unknown layer: {layer}")
            return
        
        # Update counters
        self.performance_metrics[layer]['total'] += 1
        if is_valid:
            self.performance_metrics[layer]['passed'] += 1
        else:
            self.performance_metrics[layer]['failed'] += 1
        
        # Record response time
        self.performance_metrics[layer]['response_times'].append(response_time)
        
        # Store full record
        record = {
            'timestamp': datetime.now(),
            'layer': layer,
            'is_valid': is_valid,
            'response_time': response_time,
            'details': details or {}
        }
        self.validation_history.append(record)
        
        # If validation failed, record violation
        if not is_valid:
            self.violation_history.append(record)
    
    def get_layer_metrics(self, layer: str) -> Dict[str, Any]:
        """
        Get metrics for a specific layer
        
        Args:
            layer: Layer name
            
        Returns:
            Dictionary of metrics
        """
        if layer not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[layer]
        
        total = metrics['total']
        if total == 0:
            return {
                'total_validations': 0,
                'pass_rate': 0.0,
                'fail_rate': 0.0,
                'avg_response_time': 0.0
            }
        
        response_times = metrics['response_times']
        
        return {
            'total_validations': total,
            'passed': metrics['passed'],
            'failed': metrics['failed'],
            'pass_rate': metrics['passed'] / total,
            'fail_rate': metrics['failed'] / total,
            'avg_response_time': np.mean(response_times) if response_times else 0.0,
            'max_response_time': np.max(response_times) if response_times else 0.0,
            'min_response_time': np.min(response_times) if response_times else 0.0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0.0
        }
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall guardrail system metrics
        
        Returns:
            Dictionary of overall metrics
        """
        total = sum(m['total'] for m in self.performance_metrics.values())
        passed = sum(m['passed'] for m in self.performance_metrics.values())
        failed = sum(m['failed'] for m in self.performance_metrics.values())
        
        all_response_times = []
        for m in self.performance_metrics.values():
            all_response_times.extend(m['response_times'])
        
        return {
            'total_validations': total,
            'total_passed': passed,
            'total_failed': failed,
            'overall_pass_rate': passed / total if total > 0 else 0.0,
            'overall_fail_rate': failed / total if total > 0 else 0.0,
            'avg_response_time': np.mean(all_response_times) if all_response_times else 0.0,
            'total_violations': len(self.violation_history),
            'layers': {
                'layer1': self.get_layer_metrics('layer1'),
                'layer2': self.get_layer_metrics('layer2'),
                'layer3': self.get_layer_metrics('layer3')
            }
        }
    
    def get_violation_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze violation patterns
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            Dictionary of violation patterns
        """
        if not self.violation_history:
            return {'total_violations': 0, 'patterns': []}
        
        # Filter violations in time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_violations = [
            v for v in self.violation_history 
            if v['timestamp'] > cutoff_time
        ]
        
        # Count violations by layer
        layer_counts = {}
        for v in recent_violations:
            layer = v['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Extract violation reasons
        violation_reasons = []
        for v in recent_violations:
            details = v.get('details', {})
            errors = details.get('errors', [])
            for error in errors:
                violation_reasons.append(error)
        
        # Count reason frequencies
        reason_counts = pd.Series(violation_reasons).value_counts()
        
        return {
            'total_violations': len(recent_violations),
            'time_window_hours': time_window_hours,
            'violations_by_layer': layer_counts,
            'top_violation_reasons': dict(reason_counts.head(10)),
            'violation_rate': len(recent_violations) / time_window_hours if time_window_hours > 0 else 0
        }
    
    def calculate_false_positive_rate(self, ground_truth: List[bool],
                                     predictions: List[bool]) -> float:
        """
        Calculate false positive rate
        
        Args:
            ground_truth: True labels (True = actually valid)
            predictions: Guardrail predictions (True = passed validation)
            
        Returns:
            False positive rate
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Lengths must match")
        
        # False positive: Guardrail said invalid (False) but actually valid (True)
        false_positives = sum(1 for gt, pred in zip(ground_truth, predictions)
                             if gt and not pred)
        
        # All actually invalid cases
        negatives = sum(1 for gt in ground_truth if not gt)
        
        if negatives == 0:
            return 0.0
        
        return false_positives / negatives
    
    def calculate_false_negative_rate(self, ground_truth: List[bool],
                                     predictions: List[bool]) -> float:
        """
        Calculate false negative rate
        
        Args:
            ground_truth: True labels (True = actually valid)
            predictions: Guardrail predictions (True = passed validation)
            
        Returns:
            False negative rate
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Lengths must match")
        
        # False negative: Guardrail said valid (True) but actually invalid (False)
        false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions)
                             if not gt and pred)
        
        # All actually valid cases
        positives = sum(1 for gt in ground_truth if gt)
        
        if positives == 0:
            return 0.0
        
        return false_negatives / positives
    
    def get_effectiveness_report(self) -> str:
        """
        Generate effectiveness report
        
        Returns:
            Formatted report string
        """
        overall = self.get_overall_metrics()
        violations = self.get_violation_patterns()
        
        report = []
        report.append("=" * 70)
        report.append("GUARDRAIL SYSTEM EFFECTIVENESS REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall stats
        report.append("OVERALL METRICS")
        report.append("-" * 70)
        report.append(f"Total Validations: {overall['total_validations']}")
        report.append(f"Pass Rate: {overall['overall_pass_rate']:.1%}")
        report.append(f"Fail Rate: {overall['overall_fail_rate']:.1%}")
        report.append(f"Avg Response Time: {overall['avg_response_time']:.4f}s")
        report.append(f"Total Violations: {overall['total_violations']}")
        report.append("")
        
        # Layer-by-layer
        report.append("LAYER-BY-LAYER PERFORMANCE")
        report.append("-" * 70)
        for layer_name in ['layer1', 'layer2', 'layer3']:
            layer_metrics = overall['layers'][layer_name]
            report.append(f"\n{layer_name.upper()} (Input/Prediction/Action Validation):")
            report.append(f"  Total: {layer_metrics['total_validations']}")
            report.append(f"  Pass Rate: {layer_metrics['pass_rate']:.1%}")
            report.append(f"  Avg Response Time: {layer_metrics['avg_response_time']:.4f}s")
            report.append(f"  P95 Response Time: {layer_metrics['p95_response_time']:.4f}s")
        
        report.append("")
        
        # Violations
        report.append("RECENT VIOLATIONS (Last 24 Hours)")
        report.append("-" * 70)
        report.append(f"Total: {violations['total_violations']}")
        report.append(f"Rate: {violations['violation_rate']:.2f} violations/hour")
        
        if violations.get('violations_by_layer'):
            report.append("\nBy Layer:")
            for layer, count in violations['violations_by_layer'].items():
                report.append(f"  {layer}: {count}")
        
        if violations.get('top_violation_reasons'):
            report.append("\nTop Violation Reasons:")
            for reason, count in list(violations['top_violation_reasons'].items())[:5]:
                report.append(f"  - {reason[:60]}... ({count})")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to file
        
        Args:
            filepath: Path to export file
        """
        metrics = {
            'overall': self.get_overall_metrics(),
            'violations': self.get_violation_patterns(),
            'validation_history': [
                {
                    'timestamp': v['timestamp'].isoformat(),
                    'layer': v['layer'],
                    'is_valid': v['is_valid'],
                    'response_time': v['response_time']
                }
                for v in self.validation_history
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.validation_history = []
        self.violation_history = []
        for layer in self.performance_metrics:
            self.performance_metrics[layer] = {
                'total': 0, 'passed': 0, 'failed': 0, 'response_times': []
            }
        logger.info("Metrics reset")
