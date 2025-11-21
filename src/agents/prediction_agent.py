"""
Prediction Agent (Agent 2)
Ensemble predictions with Random Forest, XGBoost, and Neural Network
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from .base_agent import BaseAgent
from ..utils.feature_engineering import FeatureEngineer
from ..utils.metrics import MetricsCalculator

class PredictionAgent(BaseAgent):
    """Agent 2: Multi-model prediction with ensemble"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        super().__init__(
            agent_name="prediction_agent",
            config_path=config_path
        )
        
        # Initialize models
        self.rf_model = None
        self.xgb_model = None
        self.nn_model = None
        
        # Initialize utilities
        self.feature_engineer = FeatureEngineer()
        self.metrics_calculator = MetricsCalculator()
        
        # Get config
        self.models_config = self.config.get('models', [])
        self.ensemble_method = self.config.get('ensemble_method', 'voting')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        self.logger.info("PredictionAgent initialized with 3 models")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required data"""
        required_keys = ['mode']  # 'train' or 'predict'
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        mode = input_data['mode']
        
        if mode == 'train':
            if 'X_train' not in input_data or 'y_train' not in input_data:
                self.logger.error("Train mode requires X_train and y_train")
                return False
        elif mode == 'predict':
            if 'X' not in input_data:
                self.logger.error("Predict mode requires X")
                return False
        else:
            self.logger.error(f"Invalid mode: {mode}")
            return False
        
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output data"""
        if 'predictions' not in output_data and 'metrics' not in output_data:
            self.logger.error("Output missing predictions or metrics")
            return False
        return True
    
    def _initialize_models(self):
        """Initialize all three models with config"""
        self.logger.info("Initializing models...")
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Neural Network
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        self.logger.info("✓ All models initialized")
    
    def train_models(self, X_train, y_train):
        """Train all three models"""
        self.logger.info(f"Training models on {len(X_train)} samples...")
        
        # Initialize if needed
        if self.rf_model is None:
            self._initialize_models()
        
        # Train Random Forest
        self.logger.info("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        self.logger.info("Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        
        # Train Neural Network
        self.logger.info("Training Neural Network...")
        self.nn_model.fit(X_train, y_train)
        
        self.logger.info("✓ All models trained successfully")
        
        return {
            'rf_trained': True,
            'xgb_trained': True,
            'nn_trained': True
        }
    
    def predict_ensemble(self, X):
        """Get ensemble predictions from all models"""
        # Get predictions from each model
        pred_rf = self.rf_model.predict(X)
        pred_xgb = self.xgb_model.predict(X)
        pred_nn = self.nn_model.predict(X)
        
        # Get probabilities
        proba_rf = self.rf_model.predict_proba(X)
        proba_xgb = self.xgb_model.predict_proba(X)
        proba_nn = self.nn_model.predict_proba(X)
        
        # Average probabilities (soft voting)
        proba_ensemble = (proba_rf + proba_xgb + proba_nn) / 3
        pred_ensemble = self.rf_model.classes_[np.argmax(proba_ensemble, axis=1)]
        
        # Calculate confidence
        confidence = np.max(proba_ensemble, axis=1)
        
        # Detect hallucinations (model disagreement)
        agreement = (pred_rf == pred_xgb) & (pred_xgb == pred_nn)
        hallucination_mask = ~agreement
        
        return {
            'predictions': pred_ensemble,
            'probabilities': proba_ensemble,
            'confidence': confidence,
            'hallucination_mask': hallucination_mask,
            'individual_predictions': {
                'rf': pred_rf,
                'xgb': pred_xgb,
                'nn': pred_nn
            }
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing logic"""
        mode = input_data['mode']
        
        if mode == 'train':
            X_train = input_data['X_train']
            y_train = input_data['y_train']
            
            # Train models
            train_status = self.train_models(X_train, y_train)
            
            # Get training metrics
            predictions = self.predict_ensemble(X_train)
            metrics = self.metrics_calculator.calculate_classification_metrics(
                y_train,
                predictions['predictions'],
                predictions['probabilities']
            )
            
            return {
                'mode': 'train',
                'train_status': train_status,
                'predictions': predictions['predictions'],
                'metrics': metrics,
                'model_agreement': (~predictions['hallucination_mask']).mean()
            }
        
        elif mode == 'predict':
            X = input_data['X']
            
            # Make predictions
            predictions = self.predict_ensemble(X)
            
            return {
                'mode': 'predict',
                'predictions': predictions['predictions'],
                'probabilities': predictions['probabilities'],
                'confidence': predictions['confidence'],
                'hallucination_mask': predictions['hallucination_mask'],
                'model_agreement': (~predictions['hallucination_mask']).mean()
            }