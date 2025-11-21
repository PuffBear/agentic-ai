"""
LLM-Based Guardrails
Multi-layer validation using LLM reasoning
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger

load_dotenv()

class LLMGuardrail:
    """
    LLM-powered guardrail for validating agent decisions
    """
    
    def __init__(self, layer: str = "general"):
        """
        Initialize guardrail
        
        Args:
            layer: Which layer this guardrail operates on
        """
        self.layer = layer
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.warning("GROQ_API_KEY not found. Guardrails in mock mode.")
            self.llm = None
        else:
            self.llm = ChatGroq(
                model="llama-3.2-90b-text-preview",
                temperature=0.1,  # Low temperature for consistent validation
                api_key=api_key
            )
        
        logger.info(f"LLMGuardrail initialized for layer: {layer}")
    
    def check_input_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Layer 1: Validate input data quality
        
        Args:
            data: Input data to validate
        
        Returns:
            Validation result
        """
        if self.llm is None:
            return {'passed': True, 'issues': [], 'severity': 'none'}
        
        prompt = f"""You are validating input data for a gaming analytics system.

Input Data:
{data}

Check for:
1. Missing or invalid values
2. Unrealistic values (e.g., age > 120, negative playtime)
3. Inconsistent data (e.g., level 100 but 0 achievements)
4. Potential data quality issues

Respond in this format:
PASSED: YES or NO
ISSUES: [list issues, or "None"]
SEVERITY: CRITICAL, HIGH, MEDIUM, LOW, or NONE"""

        try:
            messages = [
                SystemMessage(content="You are a data quality validator."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            passed = 'YES' in content.split('PASSED:')[1].split('\n')[0].upper()
            issues_text = content.split('ISSUES:')[1].split('SEVERITY:')[0].strip()
            issues = [issues_text] if issues_text.lower() != 'none' else []
            severity = content.split('SEVERITY:')[1].strip().split('\n')[0].strip().upper()
            
            return {
                'passed': passed,
                'issues': issues,
                'severity': severity,
                'layer': 'input_validation'
            }
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return {'passed': True, 'issues': [str(e)], 'severity': 'UNKNOWN'}
    
    def check_prediction_quality(
        self,
        prediction: str,
        confidence: float,
        model_agreement: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Layer 2: Validate prediction quality (hallucination detection)
        
        Args:
            prediction: Predicted value
            confidence: Prediction confidence
            model_agreement: Model agreement rate
            context: Additional context
        
        Returns:
            Validation result
        """
        if self.llm is None:
            return {'passed': confidence > 0.6 and model_agreement > 0.7, 'risk': 'unknown'}
        
        prompt = f"""You are validating ML predictions for reliability.

Prediction: {prediction}
Confidence: {confidence:.1%}
Model Agreement: {model_agreement:.1%}
Context: {context}

Assess:
1. Is confidence sufficient? (threshold: 60%)
2. Is model agreement acceptable? (threshold: 70%)
3. Any signs of hallucination or unreliable prediction?
4. Does prediction make sense given context?

Respond in this format:
PASSED: YES or NO
RISK: HIGH, MEDIUM, or LOW
CONCERNS: [list concerns, or "None"]
RECOMMENDATION: [action to take]"""

        try:
            messages = [
                SystemMessage(content="You are a prediction quality validator detecting hallucinations."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            passed = 'YES' in content.split('PASSED:')[1].split('\n')[0].upper()
            risk = content.split('RISK:')[1].split('\n')[0].strip().upper()
            concerns_text = content.split('CONCERNS:')[1].split('RECOMMENDATION:')[0].strip()
            concerns = [concerns_text] if concerns_text.lower() != 'none' else []
            recommendation = content.split('RECOMMENDATION:')[1].strip()
            
            return {
                'passed': passed,
                'risk': risk,
                'concerns': concerns,
                'recommendation': recommendation,
                'layer': 'prediction_validation'
            }
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            return {'passed': False, 'risk': 'HIGH', 'concerns': [str(e)]}
    
    def check_action_safety(
        self,
        action: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Layer 3: Validate action safety and appropriateness
        
        Args:
            action: Proposed action
            player_context: Player context
        
        Returns:
            Validation result
        """
        if self.llm is None:
            return {'approved': True, 'risk': 'unknown'}
        
        prompt = f"""You are validating a proposed action for safety and appropriateness.

Proposed Action:
- Name: {action.get('name', 'Unknown')}
- Cost: ${action.get('cost', 0)}
- Description: {action.get('description', 'N/A')}

Player Context:
- Predicted Engagement: {player_context.get('prediction', 'Unknown')}
- Confidence: {player_context.get('confidence', 0):.1%}

Check:
1. Is action appropriate for player's engagement level?
2. Is cost justified?
3. Any safety concerns or risks?
4. Better alternatives?

Respond in this format:
APPROVED: YES or NO
RISK: HIGH, MEDIUM, or LOW
CONCERNS: [list, or "None"]
ALTERNATIVE: [if any, or "None"]"""

        try:
            messages = [
                SystemMessage(content="You are an action safety validator."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            approved = 'YES' in content.split('APPROVED:')[1].split('\n')[0].upper()
            risk = content.split('RISK:')[1].split('\n')[0].strip().upper()
            concerns_text = content.split('CONCERNS:')[1].split('ALTERNATIVE:')[0].strip()
            concerns = [concerns_text] if concerns_text.lower() != 'none' else []
            alternative = content.split('ALTERNATIVE:')[1].strip()
            
            return {
                'approved': approved,
                'risk': risk,
                'concerns': concerns,
                'alternative': alternative if alternative.lower() != 'none' else None,
                'layer': 'action_validation'
            }
            
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            return {'approved': False, 'risk': 'HIGH', 'concerns': [str(e)]}
