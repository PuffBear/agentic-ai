"""
Layer 3: Action Validation Guardrail
Validates recommended actions for safety and business logic compliance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ActionValidationGuardrail:
    """
    Layer 3: Action Validation
    
    Validates:
    - Rule-based safety constraints
    - High-risk decision flagging
    - Business logic compliance
    - Action appropriateness for context
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize action validation guardrail
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Valid actions
        self.valid_actions = [
            'no_action',
            'send_discount_offer',
            'send_push_notification',
            'recommend_content',
            'adjust_difficulty',
            'send_achievement_hint',
            'offer_tutorial',
            'send_reengagement_email'
        ]
        
        # High-risk actions (require extra validation)
        self.high_risk_actions = [
            'send_discount_offer',
            'adjust_difficulty'
        ]
        
        # Risk threshold
        self.risk_threshold = self.config.get('risk_threshold', 0.7)
        
        # Business rules
        self.max_actions_per_week = self.config.get('max_actions_per_week', 3)
        self.min_days_between_actions = self.config.get('min_days_between_actions', 2)
        
        logger.info("Layer 3: Action Validation Guardrail initialized")
    
    def validate(self, action_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Main validation method for actions
        
        Args:
            action_data: Dictionary containing:
                - action: Recommended action
                - player_context: Player information
                - prediction: Prediction result
                - confidence: Action confidence score
                - expected_impact: Expected outcome
                
        Returns:
            Tuple of (is_valid, error_message, validation_details)
        """
        validation_details = {
            'action_valid': False,
            'safety_valid': False,
            'business_rules_valid': False,
            'risk_acceptable': False,
            'errors': [],
            'warnings': [],
            'risk_level': 0.0
        }
        
        # Extract data
        action = action_data.get('action')
        player_context = action_data.get('player_context', {})
        prediction = action_data.get('prediction')
        confidence = action_data.get('confidence', 0.5)
        
        if action is None:
            error_msg = "No action provided"
            validation_details['errors'].append(error_msg)
            return False, error_msg, validation_details
        
        # 1. Validate action format
        format_valid, format_msg = self._validate_action_format(action)
        validation_details['action_valid'] = format_valid
        if not format_valid:
            validation_details['errors'].append(format_msg)
            return False, format_msg, validation_details
        
        # 2. Assess risk level
        risk_level = self._assess_risk(action, player_context, confidence)
        validation_details['risk_level'] = risk_level
        
        # 3. Check safety constraints
        safety_valid, safety_msg = self._check_safety(action, player_context, risk_level)
        validation_details['safety_valid'] = safety_valid
        if not safety_valid:
            validation_details['errors'].append(safety_msg)
            logger.warning(f"Safety check failed: {safety_msg}")
            return False, safety_msg, validation_details
        
        # 4. Validate business rules
        business_valid, business_msg = self._validate_business_rules(action, player_context)
        validation_details['business_rules_valid'] = business_valid
        if not business_valid:
            validation_details['warnings'].append(business_msg)
            logger.warning(f"Business rule warning: {business_msg}")
        
        # 5. Check action appropriateness
        appropriate, appropriate_msg = self._check_appropriateness(action, player_context, prediction)
        if not appropriate:
            validation_details['warnings'].append(appropriate_msg)
            logger.warning(f"Action appropriateness warning: {appropriate_msg}")
        
        # 6. Final risk decision
        if risk_level > self.risk_threshold:
            validation_details['risk_acceptable'] = False
            error_msg = f"Risk level {risk_level:.2f} exceeds threshold {self.risk_threshold}"
            validation_details['errors'].append(error_msg)
            logger.warning(error_msg)
            
            # High-risk actions can still be flagged for human review
            if action in self.high_risk_actions:
                validation_details['requires_human_review'] = True
                logger.info(f"Action flagged for human review: {action}")
        else:
            validation_details['risk_acceptable'] = True
        
        # All critical checks passed
        logger.info(f"✓ Action validation passed: {action} (risk: {risk_level:.2f})")
        return True, "Action validation passed", validation_details
    
    def _validate_action_format(self, action: str) -> Tuple[bool, str]:
        """
        Validate action format
        
        Args:
            action: Action string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(action, str):
            return False, f"Action must be string, got {type(action)}"
        
        if action not in self.valid_actions:
            return False, f"Invalid action '{action}'. Valid actions: {self.valid_actions}"
        
        return True, "Action format valid"
    
    def _assess_risk(self, action: str, player_context: Dict[str, Any], 
                     confidence: float) -> float:
        """
        Assess risk level of action
        
        Args:
            action: Recommended action
            player_context: Player information
            confidence: Action confidence score
            
        Returns:
            Risk score (0-1, higher = more risky)
        """
        risk_score = 0.0
        
        # Base risk by action type
        action_base_risks = {
            'no_action': 0.0,
            'send_push_notification': 0.2,
            'recommend_content': 0.2,
            'send_achievement_hint': 0.1,
            'offer_tutorial': 0.1,
            'send_reengagement_email': 0.3,
            'send_discount_offer': 0.7,  # Costs money
            'adjust_difficulty': 0.6      # Affects game experience
        }
        
        risk_score = action_base_risks.get(action, 0.5)
        
        # Adjust based on confidence
        if confidence < 0.6:
            risk_score += 0.2  # Low confidence = higher risk
        elif confidence < 0.75:
            risk_score += 0.1
        
        # Adjust based on player context
        engagement = player_context.get('EngagementLevel', 'Medium')
        
        if engagement == 'Low' and action in ['adjust_difficulty', 'send_discount_offer']:
            risk_score += 0.1  # Risky to make big changes for at-risk players
        
        # Ensure risk is in [0, 1]
        risk_score = max(0.0, min(1.0, risk_score))
        
        return risk_score
    
    def _check_safety(self, action: str, player_context: Dict[str, Any],
                     risk_level: float) -> Tuple[bool, str]:
        """
        Check safety constraints
        
        Args:
            action: Recommended action
            player_context: Player information
            risk_level: Risk score
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Rule 1: Don't send discount offers to players who already purchase
        if action == 'send_discount_offer':
            if player_context.get('InGamePurchases', 0) == 1:
                return False, "Safety rule: Don't send discounts to paying players"
        
        # Rule 2: Don't increase difficulty for struggling players
        if action == 'adjust_difficulty':
            player_level = player_context.get('PlayerLevel', 50)
            difficulty = player_context.get('GameDifficulty', 'Medium')
            
            if player_level < 10 and difficulty == 'Hard':
                return False, "Safety rule: Don't adjust difficulty for low-level players on hard mode"
        
        # Rule 3: Don't spam low-engagement players
        if action in ['send_push_notification', 'send_reengagement_email']:
            engagement = player_context.get('EngagementLevel', 'Medium')
            sessions_per_week = player_context.get('SessionsPerWeek', 5)
            
            if engagement == 'Low' and sessions_per_week < 1:
                # Be careful with disengaged players
                logger.warning("Sending communication to highly disengaged player")
        
        # Rule 4: No high-risk actions for new players
        age_days = player_context.get('account_age_days', 30)
        if age_days < 7 and risk_level > 0.5:
            return False, f"Safety rule: No high-risk actions for new players (age: {age_days} days)"
        
        return True, "Safety checks passed"
    
    def _validate_business_rules(self, action: str, 
                                player_context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate business logic rules
        
        Args:
            action: Recommended action
            player_context: Player information
            
        Returns:
            Tuple of (is_valid, message)
        """
        warnings = []
        
        # Rule 1: Don't send multiple promotional actions too frequently
        if action in ['send_discount_offer', 'send_push_notification', 'send_reengagement_email']:
            actions_this_week = player_context.get('actions_this_week', 0)
            
            if actions_this_week >= self.max_actions_per_week:
                warnings.append(f"Player already received {actions_this_week} actions this week")
        
        # Rule 2: Respect action cooldowns
        days_since_last_action = player_context.get('days_since_last_action', 999)
        if days_since_last_action < self.min_days_between_actions:
            warnings.append(f"Last action was only {days_since_last_action} days ago")
        
        # Rule 3: Content recommendations should match genre preferences
        if action == 'recommend_content':
            player_genre = player_context.get('GameGenre', '')
            if not player_genre:
                warnings.append("No genre preference available for content recommendation")
        
        # Rule 4: Tutorials only for low-level players
        if action == 'offer_tutorial':
            player_level = player_context.get('PlayerLevel', 50)
            if player_level > 30:
                warnings.append(f"Offering tutorial to high-level player (level {player_level})")
        
        if warnings:
            return False, "; ".join(warnings)
        
        return True, "Business rules validation passed"
    
    def _check_appropriateness(self, action: str, player_context: Dict[str, Any],
                              prediction: str = None) -> Tuple[bool, str]:
        """
        Check if action is appropriate for player context
        
        Args:
            action: Recommended action
            player_context: Player information
            prediction: Predicted engagement level
            
        Returns:
            Tuple of (is_appropriate, message)
        """
        issues = []
        
        # Match action to engagement prediction
        if prediction == 'High':
            if action in ['send_discount_offer', 'send_reengagement_email']:
                issues.append("Aggressive action for high-engagement player")
        
        if prediction == 'Low':
            if action == 'no_action':
                issues.append("No action for at-risk low-engagement player")
        
        # Check action matches player needs
        sessions_per_week = player_context.get('SessionsPerWeek', 5)
        player_level = player_context.get('PlayerLevel', 50)
        
        if sessions_per_week < 2 and action == 'adjust_difficulty':
            issues.append("Difficulty adjustment unlikely to help infrequent player")
        
        if player_level < 5 and action == 'send_achievement_hint':
            issues.append("Achievement hints may be premature for new player")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Action is appropriate"
    
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
        report.append("ACTION VALIDATION REPORT (Layer 3)")
        report.append("=" * 60)
        report.append(f"Action Valid: {validation_details.get('action_valid', 'N/A')}")
        report.append(f"Safety Valid: {validation_details.get('safety_valid', 'N/A')}")
        report.append(f"Business Rules Valid: {validation_details.get('business_rules_valid', 'N/A')}")
        report.append(f"Risk Acceptable: {validation_details.get('risk_acceptable', 'N/A')}")
        report.append(f"Risk Level: {validation_details.get('risk_level', 0):.2f}")
        
        if validation_details.get('requires_human_review'):
            report.append("\n⚠️  FLAGGED FOR HUMAN REVIEW")
        
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
