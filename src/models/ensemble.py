"""
Ensemble Model - Combines RF, XGBoost, and NN
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining:
    - RandomForestClassifier
    - XGBoostClassifier
    - Neural Network (MLPClassifier)
    
    Uses weighted voting based on validation performance
    """
    
    def __init__(self, 
                 method: str = 'soft_voting',
                 voting_weights: List[float] = None):
        """
        Initialize ensemble
        
        Args:
            method: 'soft_voting', 'hard_voting', or 'stacking'
            voting_weights: Optional weights for each model [rf, xgb, nn]
        """
        self.method = method
        self.voting_weights = voting_weights or [0.4, 0.4, 0.2]  # Default weights
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        self.models = {
            'random_forest': self.rf_model,
            'xgboost': self.xgb_model,
            'neural_network': self.nn_model
        }
        
        self.is_fitted = False
        self.classes_ = None
        
        logger.info(f"Ensemble model initialized with {method}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Train all models in the ensemble
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        logger.info(f"Training ensemble on {len(X)} samples...")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        self.rf_model.fit(X, y)
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model.fit(X, y)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        self.nn_model.fit(X, y)
        
        self.classes_ = self.rf_model.classes_
        self.is_fitted = True
        
        logger.info("✓ Ensemble training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'soft_voting':
            probas = self.predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]
        
        elif self.method == 'hard_voting':
            # Get predictions from each model
            pred_rf = self.rf_model.predict(X)
            pred_xgb = self.xgb_model.predict(X)
            pred_nn = self.nn_model.predict(X)
            
            # Convert to numeric for voting
            from scipy import stats
            predictions = np.stack([pred_rf, pred_xgb, pred_nn], axis=1)
            
            # Majority vote
            mode_result = stats.mode(predictions, axis=1, keepdims=True)
            return mode_result.mode.flatten()
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get probabilities from each model
        proba_rf = self.rf_model.predict_proba(X)
        proba_xgb = self.xgb_model.predict_proba(X)
        proba_nn = self.nn_model.predict_proba(X)
        
        # Weighted average
        w1, w2, w3 = self.voting_weights
        proba_ensemble = (w1 * proba_rf + w2 * proba_xgb + w3 * proba_nn)
        
        return proba_ensemble
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models
        
        Returns:
            Dictionary with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'random_forest': self.rf_model.feature_importances_,
            'xgboost': self.xgb_model.feature_importances_
        }
    
    def get_model_agreement(self, X: np.ndarray) -> float:
        """
        Calculate agreement between models
        
        Args:
            X: Features to check agreement
            
        Returns:
            Fraction of samples where all models agree
        """
        pred_rf = self.rf_model.predict(X)
        pred_xgb = self.xgb_model.predict(X)
        pred_nn = self.nn_model.predict(X)
        
        agreement = (pred_rf == pred_xgb) & (pred_xgb == pred_nn)
        return agreement.mean()
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each model separately
        
        Args:
            X: Features to predict
            
        Returns:
            Dictionary of predictions from each model
        """
        return {
            'random_forest': self.rf_model.predict(X),
            'xgboost': self.xgb_model.predict(X),
            'neural_network': self.nn_model.predict(X)
        }
    
    def evaluate_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                   cv: int = 5) -> Dict[str, float]:
        """
        Evaluate each model using cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores for each model
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        scores = {}
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            scores[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            logger.info(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return scores
