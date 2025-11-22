"""
Execution Agent (Agent 4)
Simulates action execution and tracks outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from .base_agent import BaseAgent

class ExecutionAgent(BaseAgent):
    """Agent 4: Executes actions and tracks outcomes"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        super().__init__(
            agent_name="execution_agent",
            config_path=config_path
        )
        
        # Track all executions
        self.execution_history = []
        self.total_cost = 0.0
        self.total_revenue = 0.0
        
        # Success rates by action type (simulated)
        self.action_success_rates = {
            'discount_10': 0.35,    # 35% conversion rate
            'discount_20': 0.45,    # 45% conversion rate
            'notification': 0.25,   # 25% engagement boost
            'content_recommend': 0.30,  # 30% engagement boost
            'no_action': 0.0        # No change
        }
        
        self.logger.info("ExecutionAgent initialized")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input"""
        required_keys = ['mode']
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        mode = input_data['mode']
        if mode not in ['execute', 'simulate', 'get_history', 'get_roi']:
            self.logger.error(f"Invalid mode: {mode}")
            return False
        
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output"""
        return True
    
    def _simulate_outcome(
        self,
        action: Dict[str, Any],
        player_data: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Simulate the outcome of an action
        
        Args:
            action: Action to execute
            player_data: Player context
            prediction: Predicted engagement level
        
        Returns:
            Outcome with success, revenue, engagement change
        """
        action_name = action['name']
        action_cost = action['cost']
        
        # Get base success rate
        base_success_rate = self.action_success_rates.get(action_name, 0.0)
        
        # Adjust success rate based on prediction confidence and engagement
        if prediction == 'Low' and action_name in ['discount_10', 'discount_20']:
            success_rate = base_success_rate * 1.2  # More effective for low engagement
        elif prediction == 'Medium' and action_name in ['notification', 'content_recommend']:
            success_rate = base_success_rate * 1.1  # More effective for medium
        elif prediction == 'High' and action_name == 'no_action':
            success_rate = 1.0  # Always "successful" to not annoy
        else:
            success_rate = base_success_rate * 0.8  # Less effective if mismatched
        
        # Cap at 100%
        success_rate = min(success_rate, 1.0)
        
        # Simulate success
        success = np.random.random() < success_rate
        
        # Calculate revenue impact
        if success:
            if 'discount' in action_name:
                # Conversion: player makes purchase
                revenue = np.random.uniform(10, 50)  # $10-50 purchase
            elif action_name in ['notification', 'content_recommend']:
                # Engagement boost: potential future revenue
                revenue = np.random.uniform(2, 10)  # $2-10 expected value
            else:
                revenue = 0.0
        else:
            revenue = 0.0
        
        # Calculate engagement change
        if success:
            if prediction == 'Low':
                engagement_change = 1  # Low → Medium
            elif prediction == 'Medium':
                engagement_change = 1  # Medium → High
            else:
                engagement_change = 0  # Already high
        else:
            engagement_change = 0
        
        # Calculate ROI
        net_benefit = revenue - action_cost
        roi = (net_benefit / action_cost * 100) if action_cost > 0 else 0
        
        return {
            'success': success,
            'revenue': revenue,
            'cost': action_cost,
            'net_benefit': net_benefit,
            'roi': roi,
            'engagement_change': engagement_change,
            'success_rate': success_rate
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing logic"""
        mode = input_data['mode']
        
        if mode == 'execute' or mode == 'simulate':
            # Execute or simulate an action
            action = input_data['action']
            player_data = input_data.get('player_data', {})
            prediction = input_data.get('prediction', 'Medium')
            confidence = input_data.get('confidence', 0.8)
            
            # Simulate outcome
            outcome = self._simulate_outcome(action, player_data, prediction)
            
            # Record execution
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'action': action['name'],
                'action_cost': action['cost'],
                'player_data': player_data,
                'prediction': prediction,
                'confidence': confidence,
                **outcome
            }
            
            self.execution_history.append(execution_record)
            
            # Update totals
            self.total_cost += outcome['cost']
            self.total_revenue += outcome['revenue']
            
            self.logger.info(
                f"Executed {action['name']}: Success={outcome['success']}, "
                f"Revenue=${outcome['revenue']:.2f}, ROI={outcome['roi']:.1f}%"
            )
            
            return {
                'mode': mode,
                'execution_record': execution_record,
                'outcome': outcome,
                'total_executions': len(self.execution_history),
                'total_cost': self.total_cost,
                'total_revenue': self.total_revenue,
                'total_roi': ((self.total_revenue - self.total_cost) / self.total_cost * 100) 
                            if self.total_cost > 0 else 0
            }
        
        elif mode == 'get_history':
            # Return execution history
            return {
                'mode': 'get_history',
                'history': self.execution_history,
                'total_executions': len(self.execution_history)
            }
        
        elif mode == 'get_roi':
            # Calculate ROI statistics
            if not self.execution_history:
                return {
                    'mode': 'get_roi',
                    'message': 'No executions yet'
                }
            
            df = pd.DataFrame(self.execution_history)
            
            roi_stats = {
                'total_cost': self.total_cost,
                'total_revenue': self.total_revenue,
                'net_benefit': self.total_revenue - self.total_cost,
                'overall_roi': ((self.total_revenue - self.total_cost) / self.total_cost * 100)
                               if self.total_cost > 0 else 0,
                'success_rate': df['success'].mean(),
                'by_action': df.groupby('action').agg({
                    'success': 'mean',
                    'revenue': 'sum',
                    'cost': 'sum',
                    'roi': 'mean'
                }).to_dict(),
                'total_executions': len(self.execution_history)
            }
            
            return {
                'mode': 'get_roi',
                'roi_stats': roi_stats
            }
