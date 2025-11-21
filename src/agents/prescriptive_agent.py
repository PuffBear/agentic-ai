"""
Prescriptive Agent (Agent 3)
Recommends actions using contextual bandit
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.rl_bandit import ContextualBandit

class PrescriptiveAgent(BaseAgent):
    """Agent 3: Action recommendation with RL"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        super().__init__(
            agent_name="prescriptive_agent",
            config_path=config_path
        )
        
        # Define actions
        self.actions = [
            {'id': 0, 'name': 'discount_10', 'cost': 5, 'description': '10% discount offer'},
            {'id': 1, 'name': 'discount_20', 'cost': 10, 'description': '20% discount offer'},
            {'id': 2, 'name': 'notification', 'cost': 0.5, 'description': 'Engagement notification'},
            {'id': 3, 'name': 'content_recommend', 'cost': 1, 'description': 'Content recommendation'},
            {'id': 4, 'name': 'no_action', 'cost': 0, 'description': 'No action'}
        ]
        
        # Initialize bandit
        self.bandit = ContextualBandit(n_actions=len(self.actions), context_dim=6)
        
        # Get config
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        
        self.logger.info(f"PrescriptiveAgent initialized with {len(self.actions)} actions")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input"""
        required_keys = ['mode']
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        mode = input_data['mode']
        if mode not in ['recommend', 'update', 'stats']:
            self.logger.error(f"Invalid mode: {mode}")
            return False
        
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output"""
        return True
    
    def _create_context(self, player_data: Dict[str, Any]) -> np.ndarray:
        """
        Create context vector from player data
        
        Args:
            player_data: Dict with player features
        
        Returns:
            Context vector
        """
        # Extract key features for context
        context = np.array([
            player_data.get('age', 30) / 100,  # Normalized age
            player_data.get('playtime_hours', 50) / 1000,  # Normalized playtime
            player_data.get('sessions_per_week', 5) / 30,  # Normalized sessions
            player_data.get('player_level', 25) / 100,  # Normalized level
            1 if player_data.get('has_purchases', False) else 0,  # Binary purchases
            player_data.get('predicted_engagement', 1) / 2  # Normalized prediction
        ])
        
        return context
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing logic"""
        mode = input_data['mode']
        
        if mode == 'recommend':
            # Recommend action for a player
            player_data = input_data.get('player_data', {})
            context = self._create_context(player_data)
            
            # Select action
            action_id = self.bandit.select_action(context, self.exploration_rate)
            action = self.actions[action_id]
            
            self.logger.info(f"Recommended action: {action['name']}")
            
            return {
                'mode': 'recommend',
                'action_id': action_id,
                'action': action,
                'context': context.tolist()
            }
        
        elif mode == 'update':
            # Update bandit with feedback
            action_id = input_data['action_id']
            reward = input_data['reward']
            
            self.bandit.update(action_id, reward)
            
            self.logger.info(f"Updated action {action_id} with reward {reward}")
            
            return {
                'mode': 'update',
                'action_id': action_id,
                'reward': reward,
                'updated': True
            }
        
        elif mode == 'stats':
            # Get bandit statistics
            stats = self.bandit.get_action_stats()
            best_action = self.bandit.get_best_action()
            
            return {
                'mode': 'stats',
                'action_stats': stats,
                'best_action': best_action,
                'best_action_name': self.actions[best_action]['name'],
                'total_iterations': self.bandit.total_iterations
            }