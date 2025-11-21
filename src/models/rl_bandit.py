"""
Contextual Bandit for Action Recommendation
Thompson Sampling implementation
"""

import numpy as np
from typing import Dict, List, Tuple
from loguru import logger

class ContextualBandit:
    """
    Thompson Sampling Contextual Bandit
    Learns which actions work best for different player contexts
    """
    
    def __init__(self, n_actions: int = 5, context_dim: int = 6):
        """
        Initialize contextual bandit
        
        Args:
            n_actions: Number of possible actions
            context_dim: Dimension of context features
        """
        self.n_actions = n_actions
        self.context_dim = context_dim
        
        # Thompson Sampling: Beta distribution parameters
        # Each action has alpha (successes) and beta (failures)
        self.alpha = np.ones(n_actions)  # Start with Beta(1,1)
        self.beta = np.ones(n_actions)
        
        # Track statistics
        self.action_counts = np.zeros(n_actions)
        self.total_reward = np.zeros(n_actions)
        self.total_iterations = 0
        
        logger.info(f"ContextualBandit initialized: {n_actions} actions, {context_dim} context dims")
    
    def select_action(self, context: np.ndarray, exploration_rate: float = 0.1) -> int:
        """
        Select action using Thompson Sampling
        
        Args:
            context: Player context features (not used in simple version)
            exploration_rate: Probability of random exploration
        
        Returns:
            Selected action index
        """
        # With probability exploration_rate, explore randomly
        if np.random.random() < exploration_rate:
            action = np.random.randint(self.n_actions)
            logger.debug(f"Exploring: random action {action}")
            return action
        
        # Thompson Sampling: sample from Beta distribution for each action
        samples = np.random.beta(self.alpha, self.beta)
        action = np.argmax(samples)
        
        logger.debug(f"Exploiting: selected action {action} (theta={samples[action]:.3f})")
        return action
    
    def update(self, action: int, reward: float):
        """
        Update belief about action effectiveness
        
        Args:
            action: Action that was taken
            reward: Observed reward
        """
        # Update counts
        self.action_counts[action] += 1
        self.total_reward[action] += reward
        self.total_iterations += 1
        
        # Update Beta distribution parameters
        if reward > 0:
            self.alpha[action] += reward
        else:
            self.beta[action] += abs(reward)
        
        logger.debug(f"Updated action {action}: alpha={self.alpha[action]:.2f}, beta={self.beta[action]:.2f}, reward={reward}")
    
    def get_action_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each action"""
        stats = {}
        for i in range(self.n_actions):
            avg_reward = self.total_reward[i] / (self.action_counts[i] + 1e-6)
            theta = self.alpha[i] / (self.alpha[i] + self.beta[i])
            stats[i] = {
                'count': int(self.action_counts[i]),
                'total_reward': float(self.total_reward[i]),
                'avg_reward': float(avg_reward),
                'theta': float(theta),
                'alpha': float(self.alpha[i]),
                'beta': float(self.beta[i])
            }
        return stats
    
    def get_best_action(self) -> int:
        """Get current best action (highest theta)"""
        theta = self.alpha / (self.alpha + self.beta)
        return int(np.argmax(theta))