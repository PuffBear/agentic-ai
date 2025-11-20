"""
Base Agent Class
Abstract base class that all agents inherit from
Defines common interface and functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml
from pathlib import Path
from loguru import logger

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    
    All agents must implement:
    - process(): Main processing logic
    - validate_input(): Input validation
    - validate_output(): Output validation
    """
    
    def __init__(
        self,
        agent_name: str,
        config_path: str = "config/agent_config.yaml",
        enabled: bool = True
    ):
        """
        Initialize base agent
        
        Args:
            agent_name: Unique name for this agent
            config_path: Path to agent configuration file
            enabled: Whether agent is active
        """
        self.agent_name = agent_name
        self.enabled = enabled
        self.config = self._load_config(config_path)
        self.execution_history: List[Dict] = []
        self.logger = logger.bind(agent=agent_name)
        
        self.logger.info(f"{agent_name} initialized (enabled={enabled})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load agent configuration from YAML file
        
        Args:
            config_path: Path to config file
        
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract this agent's config
        agent_key = self.agent_name.lower().replace(" ", "_")
        agent_config = full_config.get('agents', {}).get(agent_key, {})
        
        return agent_config.get('config', {})
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing logic - must be implemented by each agent
        
        Args:
            input_data: Input data for processing
        
        Returns:
            Processed output data
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing
        
        Args:
            input_data: Input data to validate
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data after processing
        
        Args:
            output_data: Output data to validate
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with full validation and error handling
        
        Args:
            input_data: Input data for processing
        
        Returns:
            Output data with execution metadata
        """
        if not self.enabled:
            self.logger.warning(f"{self.agent_name} is disabled, skipping execution")
            return {
                'success': False,
                'error': 'Agent disabled',
                'agent': self.agent_name
            }
        
        execution_start = datetime.now()
        self.logger.info(f"{self.agent_name} execution started")
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Input validation failed")
            
            # Process
            output_data = self.process(input_data)
            
            # Validate output
            if not self.validate_output(output_data):
                raise ValueError("Output validation failed")
            
            # Add metadata
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()
            
            result = {
                'success': True,
                'agent': self.agent_name,
                'execution_time': execution_time,
                'timestamp': execution_end.isoformat(),
                'data': output_data
            }
            
            # Track execution
            self._track_execution(result)
            
            self.logger.info(
                f"{self.agent_name} execution completed successfully "
                f"in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()
            
            self.logger.error(f"{self.agent_name} execution failed: {str(e)}")
            
            result = {
                'success': False,
                'agent': self.agent_name,
                'execution_time': execution_time,
                'timestamp': execution_end.isoformat(),
                'error': str(e)
            }
            
            self._track_execution(result)
            
            return result
    
    def _track_execution(self, result: Dict[str, Any]) -> None:
        """
        Track execution history
        
        Args:
            result: Execution result to track
        """
        self.execution_history.append(result)
        
        # Keep only last 1000 executions to avoid memory issues
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about agent executions
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_history:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0
            }
        
        successful = [e for e in self.execution_history if e.get('success', False)]
        failed = [e for e in self.execution_history if not e.get('success', False)]
        
        execution_times = [e['execution_time'] for e in self.execution_history]
        
        stats = {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful),
            'failed_executions': len(failed),
            'success_rate': len(successful) / len(self.execution_history) if self.execution_history else 0.0,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0.0,
            'min_execution_time': min(execution_times) if execution_times else 0.0,
            'max_execution_time': max(execution_times) if execution_times else 0.0
        }
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update agent configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.logger.info(f"Configuration updated for {self.agent_name}")
    
    def reset_history(self) -> None:
        """Clear execution history"""
        self.execution_history = []
        self.logger.info(f"Execution history cleared for {self.agent_name}")
    
    def __repr__(self) -> str:
        """String representation of agent"""
        return f"{self.__class__.__name__}(name='{self.agent_name}', enabled={self.enabled})"