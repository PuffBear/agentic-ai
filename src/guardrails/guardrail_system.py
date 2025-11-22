"""
Multi-Layered Guardrail System
Rule-based validation (no LLM needed)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

class GuardrailSystem:
    """
    3-Layer Guardrail System:
    Layer 1: Input Validation
    Layer 2: Prediction Validation (Hallucination Detection)
    Layer 3: Action Validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger("GuardrailSystem")
        self.validation_history = []
        
        # Thresholds
        self.min_confidence = 0.6
        self.min_model_agreement = 0.7
        self.max_action_cost = 20.0
        
        self.logger.info("GuardrailSystem initialized with 3 layers")
    
    def layer_1_input_validation(
        self,
        player_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Layer 1: Validate input data quality
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Check for required fields
        required_fields = ['age', 'playtime_hours', 'sessions_per_week', 'player_level']
        for field in required_fields:
            if field not in player_data:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # Check for valid ranges
        age = player_data.get('age', 0)
        if age < 13 or age > 100:
            issues.append(f"Invalid age: {age} (must be 13-100)")
        
        playtime = player_data.get('playtime_hours', 0)
        if playtime < 0 or playtime > 1000:
            issues.append(f"Invalid playtime: {playtime} (must be 0-1000)")
        
        sessions = player_data.get('sessions_per_week', 0)
        if sessions < 0 or sessions > 50:
            issues.append(f"Invalid sessions: {sessions} (must be 0-50)")
        
        level = player_data.get('player_level', 0)
        if level < 0 or level > 100:
            issues.append(f"Invalid level: {level} (must be 0-100)")
        
        # Check for anomalies
        if playtime > 100 and sessions < 1:
            issues.append("Anomaly: High playtime but no sessions")
        
        if level > 50 and playtime < 10:
            issues.append("Anomaly: High level but low playtime")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def layer_2_prediction_validation(
        self,
        prediction: str,
        confidence: float,
        model_agreement: float,
        probabilities: np.ndarray
    ) -> Tuple[bool, str, List[str]]:
        """
        Layer 2: Validate prediction quality (Hallucination Detection)
        
        Returns:
            (is_valid, risk_level, concerns)
        """
        concerns = []
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            concerns.append(f"Low confidence: {confidence:.1%} < {self.min_confidence:.1%}")
        
        # Check model agreement (hallucination detection)
        if model_agreement < self.min_model_agreement:
            concerns.append(f"Low model agreement: {model_agreement:.1%} (hallucination risk)")
        
        # Check prediction uncertainty
        prob_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        normalized_entropy = prob_entropy / max_entropy
        
        if normalized_entropy > 0.8:
            concerns.append(f"High prediction uncertainty: {normalized_entropy:.2f}")
        
        # Determine risk level
        if confidence >= 0.8 and model_agreement >= 0.85:
            risk_level = 'LOW'
        elif confidence >= 0.6 and model_agreement >= 0.7:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        is_valid = risk_level != 'HIGH'
        
        return is_valid, risk_level, concerns
    
    def layer_3_action_validation(
        self,
        action: Dict[str, Any],
        prediction: str,
        confidence: float,
        player_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Layer 3: Validate action appropriateness
        
        Returns:
            (is_valid, concerns)
        """
        concerns = []
        action_name = action['name']
        action_cost = action['cost']
        
        # Check cost threshold
        if action_cost > self.max_action_cost:
            concerns.append(f"Action cost ${action_cost} exceeds max ${self.max_action_cost}")
        
        # Check action-prediction alignment
        if prediction == 'High' and action_name in ['discount_10', 'discount_20']:
            concerns.append("Unnecessary: Offering discount to already engaged player")
        
        if prediction == 'Low' and action_name == 'no_action':
            concerns.append("Missed opportunity: No action for at-risk player")
        
        # Cost-benefit analysis
        if action_cost > 5 and confidence < 0.7:
            concerns.append(f"High cost ${action_cost} with low confidence {confidence:.1%}")
        
        # Check for over-messaging risk
        sessions = player_data.get('sessions_per_week', 0)
        if sessions > 15 and action_name in ['notification', 'content_recommend']:
            concerns.append("Over-messaging risk: Player already highly active")
        
        is_valid = len(concerns) == 0
        
        return is_valid, concerns
    
    def validate_full_pipeline(
        self,
        player_data: Dict[str, Any],
        prediction: str,
        confidence: float,
        model_agreement: float,
        probabilities: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run all 3 layers of validation
        
        Returns:
            Comprehensive validation result
        """
        timestamp = datetime.now().isoformat()
        
        # Layer 1: Input Validation
        input_valid, input_issues = self.layer_1_input_validation(player_data)
        
        # Layer 2: Prediction Validation
        pred_valid, risk_level, pred_concerns = self.layer_2_prediction_validation(
            prediction, confidence, model_agreement, probabilities
        )
        
        # Layer 3: Action Validation
        action_valid, action_concerns = self.layer_3_action_validation(
            action, prediction, confidence, player_data
        )
        
        # Overall decision
        all_valid = input_valid and pred_valid and action_valid
        
        # Aggregate all concerns
        all_concerns = []
        if input_issues:
            all_concerns.extend([f"[Input] {issue}" for issue in input_issues])
        if pred_concerns:
            all_concerns.extend([f"[Prediction] {concern}" for concern in pred_concerns])
        if action_concerns:
            all_concerns.extend([f"[Action] {concern}" for concern in action_concerns])
        
        # Determine overall risk
        if not all_valid:
            if not input_valid:
                overall_risk = 'CRITICAL'
            elif risk_level == 'HIGH':
                overall_risk = 'HIGH'
            else:
                overall_risk = 'MEDIUM'
        else:
            overall_risk = risk_level
        
        validation_result = {
            'timestamp': timestamp,
            'approved': all_valid,
            'overall_risk': overall_risk,
            'layers': {
                'layer_1_input': {
                    'valid': input_valid,
                    'issues': input_issues
                },
                'layer_2_prediction': {
                    'valid': pred_valid,
                    'risk_level': risk_level,
                    'concerns': pred_concerns
                },
                'layer_3_action': {
                    'valid': action_valid,
                    'concerns': action_concerns
                }
            },
            'all_concerns': all_concerns,
            'recommendation': 'EXECUTE' if all_valid else 'BLOCK'
        }
        
        # Log result
        self.validation_history.append(validation_result)
        
        if all_valid:
            self.logger.info(f"✅ APPROVED: {action['name']} for {prediction} engagement")
        else:
            self.logger.warning(
                f"❌ BLOCKED: {action['name']} - Risk: {overall_risk}, "
                f"Concerns: {len(all_concerns)}"
            )
        
        return validation_result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics on validations"""
        if not self.validation_history:
            return {'message': 'No validations yet'}
        
        total = len(self.validation_history)
        approved = sum(1 for v in self.validation_history if v['approved'])
        blocked = total - approved
        
        risk_levels = [v['overall_risk'] for v in self.validation_history]
        
        return {
            'total_validations': total,
            'approved': approved,
            'blocked': blocked,
            'approval_rate': approved / total,
            'risk_distribution': {
                'LOW': risk_levels.count('LOW'),
                'MEDIUM': risk_levels.count('MEDIUM'),
                'HIGH': risk_levels.count('HIGH'),
                'CRITICAL': risk_levels.count('CRITICAL')
            }
        }
