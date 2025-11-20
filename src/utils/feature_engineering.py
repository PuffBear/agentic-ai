"""
Feature Engineering
Creates derived features and transforms data for modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Tuple, Optional
from loguru import logger

class FeatureEngineer:
    """Handles feature engineering and transformations"""
    
    def __init__(self):
        """Initialize feature engineer with transformers"""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        
        logger.info("FeatureEngineer initialized")
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional interaction features
        """
        df = df.copy()
        
        # Playtime intensity: hours per session
        df['PlaytimeIntensity'] = df['PlayTimeHours'] / (df['SessionsPerWeek'] * df['AvgSessionDurationMinutes'] / 60 + 1e-6)
        
        # Achievement rate: achievements per level
        df['AchievementRate'] = df['AchievementsUnlocked'] / (df['PlayerLevel'] + 1)
        
        # Session engagement: sessions * duration
        df['SessionEngagement'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
        
        # Age-playtime interaction
        df['AgePlaytimeInteraction'] = df['Age'] * df['PlayTimeHours']
        
        # Purchase power: binary purchase * playtime
        df['PurchasePower'] = (df['InGamePurchases'] == 'Yes').astype(int) * df['PlayTimeHours']
        
        logger.info(f"Created {5} interaction features")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal/behavioral features
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        # Average session duration in hours
        df['AvgSessionDurationHours'] = df['AvgSessionDurationMinutes'] / 60
        
        # Total weekly playtime estimate
        df['EstimatedWeeklyPlaytime'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes'] / 60
        
        # Playtime per level (progression speed)
        df['PlaytimePerLevel'] = df['PlayTimeHours'] / (df['PlayerLevel'] + 1)
        
        logger.info(f"Created {3} temporal features")
        
        return df
    
    def create_behavioral_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite behavioral scores
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with behavioral scores
        """
        df = df.copy()
        
        # Engagement score (normalized composite)
        engagement_components = [
            df['SessionsPerWeek'] / df['SessionsPerWeek'].max(),
            df['PlayTimeHours'] / df['PlayTimeHours'].max(),
            df['AchievementsUnlocked'] / df['AchievementsUnlocked'].max(),
            df['PlayerLevel'] / df['PlayerLevel'].max()
        ]
        df['EngagementScore'] = np.mean(engagement_components, axis=0)
        
        # Commitment level (sessions * duration * level)
        df['CommitmentLevel'] = (
            (df['SessionsPerWeek'] / df['SessionsPerWeek'].max()) *
            (df['AvgSessionDurationMinutes'] / df['AvgSessionDurationMinutes'].max()) *
            (df['PlayerLevel'] / df['PlayerLevel'].max())
        )
        
        logger.info(f"Created {2} behavioral scores")
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_features: List[str],
        method: str = "onehot",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            categorical_features: List of categorical column names
            method: Encoding method ('onehot' or 'label')
            fit: Whether to fit encoder (True for training, False for test)
        
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for feature in categorical_features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in DataFrame")
                continue
            
            if method == "onehot":
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[feature]])
                    self.encoders[feature] = encoder
                else:
                    encoder = self.encoders[feature]
                    encoded = encoder.transform(df[[feature]])
                
                # Create column names
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=feature_names,
                    index=df.index
                )
                
                # Drop original and add encoded
                df = df.drop(feature, axis=1)
                df = pd.concat([df, encoded_df], axis=1)
                
            elif method == "label":
                if fit:
                    encoder = LabelEncoder()
                    df[f"{feature}_encoded"] = encoder.fit_transform(df[feature])
                    self.encoders[feature] = encoder
                else:
                    encoder = self.encoders[feature]
                    df[f"{feature}_encoded"] = encoder.transform(df[feature])
                
                df = df.drop(feature, axis=1)
        
        logger.info(f"Encoded {len(categorical_features)} categorical features using {method}")
        
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        method: str = "standard",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            numerical_features: List of numerical column names
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit scaler (True for training, False for test)
        
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Scaling method {method} not implemented")
        
        if fit:
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            self.scalers['standard'] = scaler
        else:
            scaler = self.scalers['standard']
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        logger.info(f"Scaled {len(numerical_features)} features using {method} scaling")
        
        return df
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        create_interactions: bool = True,
        create_temporal: bool = True,
        create_behavioral: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input DataFrame
            create_interactions: Whether to create interaction features
            create_temporal: Whether to create temporal features
            create_behavioral: Whether to create behavioral scores
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        logger.info("Starting feature engineering pipeline")
        
        if create_interactions:
            df = self.create_interaction_features(df)
        
        if create_temporal:
            df = self.create_temporal_features(df)
        
        if create_behavioral:
            df = self.create_behavioral_scores(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after transformation"""
        return self.feature_names