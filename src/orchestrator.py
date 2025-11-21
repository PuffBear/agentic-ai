"""
LLM-Powered Orchestrator
Coordinates all agents using LangChain and LLM reasoning
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger

# Load environment variables
load_dotenv()

class LLMOrchestrator:
    """
    LangChain-based orchestrator that coordinates agents with LLM reasoning
    """
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize orchestrator with agents
        
        Args:
            agents: Dictionary of agent instances
        """
        self.agents = agents
        
        # Initialize LLM
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.warning("GROQ_API_KEY not found. Using mock mode.")
            self.llm = None
        else:
            self.llm = ChatGroq(
                model="llama-3.2-90b-text-preview",
                temperature=0.3,
                api_key=api_key
            )
        
        logger.info("LLMOrchestrator initialized with LangChain")
    
    def explain_prediction(
        self,
        player_data: Dict[str, Any],
        prediction: str,
        confidence: float,
        model_agreement: float
    ) -> str:
        """
        Use LLM to explain prediction in natural language
        
        Args:
            player_data: Player features
            prediction: Predicted engagement level
            confidence: Prediction confidence
            model_agreement: Model agreement rate
        
        Returns:
            Natural language explanation
        """
        if self.llm is None:
            return "LLM not available. Please set GROQ_API_KEY."
        
        prompt = f"""You are an AI analyst explaining gaming behavior predictions.

Player Profile:
- Age: {player_data.get('age', 'N/A')}
- Playtime: {player_data.get('playtime_hours', 'N/A')} hours
- Sessions/Week: {player_data.get('sessions_per_week', 'N/A')}
- Player Level: {player_data.get('player_level', 'N/A')}
- Has Purchases: {player_data.get('has_purchases', 'N/A')}

Prediction Results:
- Predicted Engagement: {prediction}
- Confidence: {confidence:.1%}
- Model Agreement: {model_agreement:.1%}

Provide a clear, concise explanation (2-3 sentences) of:
1. Why this prediction makes sense given the player's behavior
2. Any concerns or risks with this prediction
3. Your assessment of prediction reliability

Be direct and professional."""

        try:
            messages = [
                SystemMessage(content="You are an expert gaming analytics AI."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return f"Prediction: {prediction} (Confidence: {confidence:.1%})"
    
    def recommend_action_with_reasoning(
        self,
        player_data: Dict[str, Any],
        prediction: str,
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to recommend action with reasoning
        
        Args:
            player_data: Player features
            prediction: Predicted engagement level
            available_actions: List of possible actions
        
        Returns:
            Recommended action with explanation
        """
        if self.llm is None:
            # Fallback to RL agent
            return {
                'action': available_actions[2],  # Default to notification
                'reasoning': 'LLM not available, using default action.'
            }
        
        actions_text = "\n".join([
            f"{i}. {a['name']} (Cost: ${a['cost']}) - {a['description']}"
            for i, a in enumerate(available_actions)
        ])
        
        prompt = f"""You are an AI strategist recommending player retention actions.

Player Profile:
- Predicted Engagement: {prediction}
- Age: {player_data.get('age', 'N/A')}
- Playtime: {player_data.get('playtime_hours', 'N/A')} hours
- Sessions/Week: {player_data.get('sessions_per_week', 'N/A')}
- Player Level: {player_data.get('player_level', 'N/A')}
- Has Purchases: {player_data.get('has_purchases', 'N/A')}

Available Actions:
{actions_text}

Which action would be most effective and why? Consider:
1. Player's current engagement level
2. Cost-effectiveness
3. Likelihood of success

Respond in this format:
RECOMMENDED ACTION: [action name]
REASONING: [2-3 sentences explaining why]"""

        try:
            messages = [
                SystemMessage(content="You are an expert in player retention strategies."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            # Parse response
            lines = content.split('\n')
            action_line = [l for l in lines if 'RECOMMENDED ACTION:' in l]
            reasoning_lines = [l for l in lines if 'REASONING:' in l]
            
            if action_line and reasoning_lines:
                recommended_name = action_line[0].split('RECOMMENDED ACTION:')[1].strip()
                reasoning = reasoning_lines[0].split('REASONING:')[1].strip()
                
                # Find matching action
                recommended_action = next(
                    (a for a in available_actions if a['name'].lower() in recommended_name.lower()),
                    available_actions[2]  # Default
                )
                
                return {
                    'action': recommended_action,
                    'reasoning': reasoning,
                    'llm_override': True
                }
            else:
                # Fallback
                return {
                    'action': available_actions[2],
                    'reasoning': content,
                    'llm_override': False
                }
                
        except Exception as e:
            logger.error(f"LLM action recommendation failed: {e}")
            return {
                'action': available_actions[2],
                'reasoning': f'LLM failed: {str(e)}',
                'llm_override': False
            }
    
    def validate_decision(
        self,
        decision_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to validate a decision (Guardrail Layer)
        
        Args:
            decision_context: Context about the decision
        
        Returns:
            Validation result with risk assessment
        """
        if self.llm is None:
            return {
                'validated': True,
                'risk_level': 'unknown',
                'concerns': [],
                'explanation': 'LLM validation not available'
            }
        
        prompt = f"""You are an AI safety validator checking gaming analytics decisions.

Decision Context:
- Predicted Engagement: {decision_context.get('prediction', 'N/A')}
- Confidence: {decision_context.get('confidence', 0):.1%}
- Model Agreement: {decision_context.get('model_agreement', 0):.1%}
- Recommended Action: {decision_context.get('action', 'N/A')}
- Action Cost: ${decision_context.get('action_cost', 0)}

Validate this decision by checking:
1. Is the confidence high enough to act?
2. Is model agreement sufficient?
3. Is the action appropriate for the prediction?
4. Are there any red flags or concerns?

Respond in this format:
VALIDATED: YES or NO
RISK LEVEL: LOW, MEDIUM, or HIGH
CONCERNS: [list any concerns, or "None"]
EXPLANATION: [1-2 sentences]"""

        try:
            messages = [
                SystemMessage(content="You are an expert AI safety validator."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            # Parse response
            validated = 'YES' in content.split('VALIDATED:')[1].split('\n')[0].upper() if 'VALIDATED:' in content else True
            
            risk_match = content.split('RISK LEVEL:')[1].split('\n')[0].strip() if 'RISK LEVEL:' in content else 'MEDIUM'
            risk_level = risk_match.upper() if risk_match.upper() in ['LOW', 'MEDIUM', 'HIGH'] else 'MEDIUM'
            
            concerns_section = content.split('CONCERNS:')[1].split('EXPLANATION:')[0].strip() if 'CONCERNS:' in content else 'None'
            concerns = [concerns_section] if concerns_section.lower() != 'none' else []
            
            explanation = content.split('EXPLANATION:')[1].strip() if 'EXPLANATION:' in content else content
            
            return {
                'validated': validated,
                'risk_level': risk_level,
                'concerns': concerns,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return {
                'validated': True,
                'risk_level': 'MEDIUM',
                'concerns': [f'Validation error: {str(e)}'],
                'explanation': 'Validation could not be completed'
            }
