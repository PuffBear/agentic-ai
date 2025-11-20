"""
Data Ingestion Agent (Agent 1)
Handles data loading, preprocessing, quality checks, and anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from .base_agent import BaseAgent
from ..utils.data_loader import DataLoader
from ..utils.feature_engineering import FeatureEngineer

class DataAgent(BaseAgent):
    """Agent 1: Autonomous data ingestion and preprocessing"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize Data Ingestion Agent"""
        super().__init__(
            agent_name="data_agent",
            config_path=config_path
        )
        
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.anomaly_detector = None
        
        # Get config values
        self.batch_size = self.config.get('batch_size', 1000)
        self.validation_rules = self.config.get('validation_rules', {})
        self.feature_engineering_steps = self.config.get('feature_engineering', [])
        self.anomaly_config = self.config.get('anomaly_detection', {})
        
        self.logger.info("DataAgent initialized with feature engineering and anomaly detection")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data structure
        
        Args:
            input_data: Input dictionary
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['mode']  # 'load', 'transform', or 'full_pipeline'
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        mode = input_data['mode']
        if mode not in ['load', 'transform', 'full_pipeline']:
            self.logger.error(f"Invalid mode: {mode}")
            return False
        
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data
        
        Args:
            output_data: Output dictionary
        
        Returns:
            True if valid, False otherwise
        """
        if 'data' not in output_data:
            self.logger.error("Output missing 'data' key")
            return False
        
        if not isinstance(output_data['data'], pd.DataFrame):
            self.logger.error("Output data is not a DataFrame")
            return False
        
        return True
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame schema and data quality
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = [
            'Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours',
            'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['passed'] = False
            validation_results['issues'].append(f"Missing columns: {missing_columns}")
        
        # Check data types and ranges from config
        age_range = self.validation_rules.get('age_range', [0, 100])
        if 'Age' in df.columns:
            invalid_ages = df[(df['Age'] < age_range[0]) | (df['Age'] > age_range[1])]
            if len(invalid_ages) > 0:
                validation_results['warnings'].append(
                    f"{len(invalid_ages)} rows with age outside [{age_range[0]}, {age_range[1]}]"
                )
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            validation_results['warnings'].append(
                f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}"
            )
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"{duplicates} duplicate rows found")
        
        self.logger.info(f"Schema validation: {validation_results['passed']}, "
                        f"{len(validation_results['issues'])} issues, "
                        f"{len(validation_results['warnings'])} warnings")
        
        return validation_results
    
    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in numerical features using Isolation Forest
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with anomaly scores
        """
        if not self.anomaly_config.get('method') == 'isolation_forest':
            self.logger.info("Anomaly detection disabled or method not configured")
            return df
        
        # Select numerical features for anomaly detection
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if present
        if 'EngagementLevel' in numerical_features:
            numerical_features.remove('EngagementLevel')
        
        if not numerical_features:
            self.logger.warning("No numerical features for anomaly detection")
            return df
        
        # Train Isolation Forest
        contamination = self.anomaly_config.get('contamination', 0.05)
        
        if self.anomaly_detector is None:
            self.anomaly_detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit on numerical features
            X = df[numerical_features].fillna(df[numerical_features].median())
            self.anomaly_detector.fit(X)
            self.logger.info(f"Trained Isolation Forest on {len(numerical_features)} features")
        
        # Predict anomalies
        X = df[numerical_features].fillna(df[numerical_features].median())
        anomaly_scores = self.anomaly_detector.score_samples(X)
        anomaly_labels = self.anomaly_detector.predict(X)
        
        # Add to DataFrame
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = (anomaly_labels == -1).astype(int)
        
        n_anomalies = df['is_anomaly'].sum()
        self.logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        
        return df
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing logic for data ingestion
        
        Args:
            input_data: Dictionary with mode and optional data_path
        
        Returns:
            Dictionary with processed data and metadata
        """
        mode = input_data['mode']
        self.logger.info(f"Processing data in mode: {mode}")
        
        if mode == 'load' or mode == 'full_pipeline':
            # Load data
            data_path = input_data.get('data_path', 'data/raw/online_gaming_behavior_dataset.csv')
            self.data_loader.data_path = data_path
            df = self.data_loader.load_data()
            
            # Validate schema
            validation_results = self._validate_schema(df)
            
            if not validation_results['passed']:
                return {
                    'data': df,
                    'validation': validation_results,
                    'stage': 'load_failed'
                }
            
            output = {
                'data': df,
                'validation': validation_results,
                'original_shape': df.shape,
                'stage': 'loaded'
            }
        
        else:
            # Transform mode - data should be provided
            df = input_data.get('data')
            if df is None:
                raise ValueError("Transform mode requires 'data' in input_data")
            
            output = {
                'data': df,
                'original_shape': df.shape,
                'stage': 'transform'
            }
        
        if mode == 'transform' or mode == 'full_pipeline':
            # Feature engineering
            if 'interaction_features' in self.feature_engineering_steps:
                df = self.feature_engineer.create_interaction_features(df)
            
            if 'temporal_features' in self.feature_engineering_steps:
                df = self.feature_engineer.create_temporal_features(df)
            
            if 'behavioral_scores' in self.feature_engineering_steps:
                df = self.feature_engineer.create_behavioral_scores(df)
            
            # Detect anomalies
            df = self._detect_anomalies(df)
            
            output['data'] = df
            output['engineered_features'] = [col for col in df.columns if col not in output.get('data', df).columns]
            output['final_shape'] = df.shape
            output['stage'] = 'transformed'
        
        self.logger.info(f"Data processing complete. Final shape: {df.shape}")
        
        return output