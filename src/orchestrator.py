"""
Orchestrator - Connects all 5 agents into a complete pipeline
"""

import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

from src.agents import (
    DataAgent,
    PredictionAgent,
    PrescriptiveAgent,
    ExecutionAgent,
    MonitoringAgent
)
from src.guardrails import GuardrailSystem

class AgenticOrchestrator:
    """
    Orchestrates the complete agentic AI pipeline
    
    Pipeline:
    1. DataAgent: Ingest & preprocess
    2. PredictionAgent: Ensemble ML prediction
    3. Guardrails Layer 2: Validate prediction
    4. PrescriptiveAgent: Recommend action
    5. Guardrails Layer 3: Validate action
    6. ExecutionAgent: Execute action
    7. MonitoringAgent: Track performance & drift
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AgenticOrchestrator")
        
        # Initialize all agents
        self.logger.info("Initializing 5-agent system...")
        self.data_agent = DataAgent()
        self.prediction_agent = PredictionAgent()
        self.prescriptive_agent = PrescriptiveAgent()
        self.execution_agent = ExecutionAgent()
        self.monitoring_agent = MonitoringAgent()
        
        # Initialize guardrails
        self.guardrails = GuardrailSystem()
        
        # Pipeline state
        self.is_trained = False
        self.baseline_set = False
        self.pipeline_history = []
        
        self.logger.info("✅ AgenticOrchestrator initialized with 5 agents + guardrails")
    
    def train_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train the prediction models
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Training results
        """
        self.logger.info("Training pipeline...")
        
        # Train Agent 2: Prediction
        train_result = self.prediction_agent.execute({
            'mode': 'train',
            'X_train': X_train,
            'y_train': y_train
        })
        
        # Set baseline for Agent 5: Monitoring
        self.monitoring_agent.execute({
            'mode': 'set_baseline',
            'X': X_train,
            'y': y_train
        })
        
        self.is_trained = True
        self.baseline_set = True
        
        self.logger.info("✅ Pipeline trained successfully")
        
        return {
            'training_metrics': train_result['data']['metrics'],
            'model_agreement': train_result['data']['model_agreement']
        }
    
    def process_player(
        self,
        player_features: np.ndarray,
        player_data: Dict[str, Any],
        execute_action: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline for a single player
        
        Args:
            player_features: Scaled feature vector
            player_data: Raw player data dict
            execute_action: Whether to actually execute recommended action
        
        Returns:
            Complete pipeline result
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train_pipeline() first.")
        
        timestamp = datetime.now().isoformat()
        pipeline_id = len(self.pipeline_history)
        
        self.logger.info(f"Processing player {pipeline_id}...")
        
        # ===== LAYER 1: INPUT VALIDATION =====
        input_valid, input_issues = self.guardrails.layer_1_input_validation(player_data)
        
        if not input_valid:
            self.logger.warning(f"❌ Input validation failed: {input_issues}")
            return {
                'pipeline_id': pipeline_id,
                'timestamp': timestamp,
                'status': 'BLOCKED_INPUT',
                'input_issues': input_issues
            }
        
        # ===== AGENT 2: PREDICTION =====
        pred_result = self.prediction_agent.execute({
            'mode': 'predict',
            'X': player_features.reshape(1, -1)
        })
        
        prediction = pred_result['data']['predictions'][0]
        confidence = pred_result['data']['confidence'][0]
        probabilities = pred_result['data']['probabilities'][0]
        model_agreement = pred_result['data']['model_agreement']
        hallucination = pred_result['data']['hallucination_mask'][0]
        
        # ===== LAYER 2: PREDICTION VALIDATION =====
        pred_valid, risk_level, pred_concerns = self.guardrails.layer_2_prediction_validation(
            prediction, confidence, model_agreement, probabilities
        )
        
        if not pred_valid:
            self.logger.warning(f"❌ Prediction validation failed: {pred_concerns}")
            return {
                'pipeline_id': pipeline_id,
                'timestamp': timestamp,
                'status': 'BLOCKED_PREDICTION',
                'prediction': prediction,
                'confidence': confidence,
                'risk_level': risk_level,
                'concerns': pred_concerns
            }
        
        # ===== AGENT 3: PRESCRIPTIVE ACTION =====
        # Pass player_data directly - prescriptive agent will extract what it needs
        presc_result = self.prescriptive_agent.execute({
            'mode': 'recommend',
            'player_data': player_data  # Pass full dict with correct field names
        })
        
        recommended_action = presc_result['data']['recommended_action']
        
        # ===== LAYER 3: ACTION VALIDATION =====
        action_valid, action_concerns = self.guardrails.layer_3_action_validation(
            recommended_action, prediction, confidence, player_data
        )
        
        if not action_valid:
            self.logger.warning(f"⚠️ Action validation concerns: {action_concerns}")
        
        # ===== FULL GUARDRAIL VALIDATION =====
        full_validation = self.guardrails.validate_full_pipeline(
            player_data,
            prediction,
            confidence,
            model_agreement,
            probabilities,
            recommended_action
        )
        
        # ===== AGENT 4: EXECUTION =====
        execution_result = None
        if execute_action and full_validation['approved']:
            execution_result = self.execution_agent.execute({
                'mode': 'execute',
                'action': recommended_action,
                'player_data': player_data,
                'prediction': prediction,
                'confidence': confidence
            })
            status = 'EXECUTED'
        elif not full_validation['approved']:
            status = 'BLOCKED_VALIDATION'
        else:
            status = 'SIMULATED'
        
        # Compile full result
        pipeline_result = {
            'pipeline_id': pipeline_id,
            'timestamp': timestamp,
            'status': status,
            'player_data': player_data,
            'prediction': {
                'label': prediction,
                'confidence': confidence,
                'model_agreement': model_agreement,
                'hallucination': hallucination,
                'probabilities': probabilities.tolist()
            },
            'recommended_action': recommended_action,
            'validation': full_validation,
            'execution': execution_result['data'] if execution_result else None
        }
        
        self.pipeline_history.append(pipeline_result)
        
        self.logger.info(
            f"✅ Pipeline {pipeline_id} complete: {status} - "
            f"{prediction} → {recommended_action['name']}"
        )
        
        return pipeline_result
    
    def monitor_system_health(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_baseline: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check system health with Agent 5
        
        Returns:
            System health report
        """
        self.logger.info("Checking system health...")
        
        # Check drift
        drift_result = self.monitoring_agent.execute({
            'mode': 'check_drift',
            'X': X_val,
            'X_baseline': X_baseline
        })
        
        # Monitor performance
        pred_result = self.prediction_agent.execute({
            'mode': 'predict',
            'X': X_val
        })
        
        perf_result = self.monitoring_agent.execute({
            'mode': 'monitor_performance',
            'y_true': y_val,
            'y_pred': pred_result['data']['predictions'],
            'confidence': pred_result['data']['confidence']
        })
        
        # Get alerts
        alerts_result = self.monitoring_agent.execute({
            'mode': 'get_alerts'
        })
        
        health_report = {
            'drift': drift_result['data'],
            'performance': perf_result['data'],
            'alerts': alerts_result['data']['alerts'],
            'recommend_retrain': perf_result['data']['recommend_retrain']
        }
        
        self.logger.info("✅ System health check complete")
        
        return health_report
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics"""
        if not self.pipeline_history:
            return {'message': 'No pipeline runs yet'}
        
        total = len(self.pipeline_history)
        statuses = [p['status'] for p in self.pipeline_history]
        
        # Execution stats
        exec_results = [p['execution'] for p in self.pipeline_history if p['execution']]
        total_cost = sum(e['total_cost'] for e in exec_results) if exec_results else 0
        total_revenue = sum(e['total_revenue'] for e in exec_results) if exec_results else 0
        
        # Guardrail stats
        guardrail_stats = self.guardrails.get_validation_stats()
        
        return {
            'total_runs': total,
            'status_distribution': {
                'EXECUTED': statuses.count('EXECUTED'),
                'BLOCKED_INPUT': statuses.count('BLOCKED_INPUT'),
                'BLOCKED_PREDICTION': statuses.count('BLOCKED_PREDICTION'),
                'BLOCKED_VALIDATION': statuses.count('BLOCKED_VALIDATION'),
                'SIMULATED': statuses.count('SIMULATED')
            },
            'execution_stats': {
                'total_cost': total_cost,
                'total_revenue': total_revenue,
                'net_benefit': total_revenue - total_cost,
                'roi': ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
            },
            'guardrail_stats': guardrail_stats
        }
