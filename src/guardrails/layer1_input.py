"""
Layer 1: Input Validation Guardrail
Validates data schema, ranges, and detects adversarial inputs
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field, ValidationError, field_validator
import logging

logger = logging.getLogger(__name__)


class PlayerInputSchema(BaseModel):
    """Pydantic schema for player input validation"""
    
    PlayerID: int = Field(ge=1, description="Player ID must be positive")
    Age: int = Field(ge=16, le=100, description="Age must be between 16 and 100")
    Gender: str = Field(pattern="^(Male|Female)$", description="Gender must be Male or Female")
    Location: str = Field(min_length=2, max_length=50, description="Location must be valid")
    GameGenre: str = Field(description="Game genre")
    PlayTimeHours: float = Field(ge=0, le=10000, description="PlayTime must be non-negative and reasonable")
    InGamePurchases: int = Field(ge=0, le=1, description="InGamePurchases must be 0 or 1")
    GameDifficulty: str = Field(pattern="^(Easy|Medium|Hard)$", description="Difficulty must be Easy, Medium, or Hard")
    SessionsPerWeek: int = Field(ge=0, le=100, description="Sessions per week must be reasonable")
    AvgSessionDurationMinutes: float = Field(ge=0, le=1440, description="Session duration must be reasonable (max 24 hours)")
    PlayerLevel: int = Field(ge=1, le=100, description="Player level must be between 1 and 100")
    AchievementsUnlocked: int = Field(ge=0, le=1000, description="Achievements must be non-negative and reasonable")
    
    @field_validator('GameGenre')
    @classmethod
    def validate_genre(cls, v):
        valid_genres = ['Action', 'Adventure', 'Strategy', 'Sports', 'RPG', 'Simulation']
        if v not in valid_genres:
            raise ValueError(f"Genre must be one of {valid_genres}")
        return v


class InputValidationGuardrail:
    """
    Layer 1: Input Validation
    
    Validates:
    - Schema compliance (Pydantic)
    - Range checks
    - Data type enforcement
    - Adversarial input detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize input validation guardrail
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.expected_features = [
            'PlayerID', 'Age', 'Gender', 'Location', 'GameGenre',
            'PlayTimeHours', 'InGamePurchases', 'GameDifficulty',
            'SessionsPerWeek', 'AvgSessionDurationMinutes',
            'PlayerLevel', 'AchievementsUnlocked'
        ]
        
        # Statistical thresholds for anomaly detection
        self.zscore_threshold = self.config.get('zscore_threshold', 5)
        
        logger.info("Layer 1: Input Validation Guardrail initialized")
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Main validation method
        
        Args:
            data: Input data dictionary or DataFrame
            
        Returns:
            Tuple of (is_valid, error_message, validation_details)
        """
        validation_details = {
            'schema_valid': False,
            'range_valid': False,
            'type_valid': False,
            'adversarial_detected': False,
            'errors': []
        }
        
        # Convert DataFrame to dict if needed
        if isinstance(data, pd.DataFrame):
            if len(data) == 1:
                data = data.iloc[0].to_dict()
            else:
                # Batch validation
                return self._validate_batch(data)
        
        # 1. Schema validation
        schema_valid, schema_msg = self._validate_schema(data)
        validation_details['schema_valid'] = schema_valid
        if not schema_valid:
            validation_details['errors'].append(schema_msg)
            return False, schema_msg, validation_details
        
        # 2. Range validation
        range_valid, range_msg = self._validate_ranges(data)
        validation_details['range_valid'] = range_valid
        if not range_valid:
            validation_details['errors'].append(range_msg)
            return False, range_msg, validation_details
        
        # 3. Data type validation
        type_valid, type_msg = self._validate_types(data)
        validation_details['type_valid'] = type_valid
        if not type_valid:
            validation_details['errors'].append(type_msg)
            return False, type_msg, validation_details
        
        # 4. Adversarial input detection
        adversarial, adv_msg = self._detect_adversarial(data)
        validation_details['adversarial_detected'] = adversarial
        if adversarial:
            validation_details['errors'].append(adv_msg)
            logger.warning(f"Adversarial input detected: {adv_msg}")
            return False, adv_msg, validation_details
        
        # All checks passed
        logger.info("✓ Input validation passed")
        return True, "Input validation passed", validation_details
    
    def _validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate input against Pydantic schema
        
        Args:
            data: Input dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate using Pydantic
            PlayerInputSchema(**data)
            return True, "Schema validation passed"
        except ValidationError as e:
            error_msg = f"Schema validation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected schema validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_ranges(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate numeric ranges
        
        Args:
            data: Input dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Define valid ranges
        ranges = {
            'Age': (16, 100),
            'PlayTimeHours': (0, 10000),
            'SessionsPerWeek': (0, 100),
            'AvgSessionDurationMinutes': (0, 1440),
            'PlayerLevel': (1, 100),
            'AchievementsUnlocked': (0, 1000)
        }
        
        for field, (min_val, max_val) in ranges.items():
            if field in data:
                value = data[field]
                if not (min_val <= value <= max_val):
                    error_msg = f"Range validation failed: {field}={value} not in [{min_val}, {max_val}]"
                    logger.error(error_msg)
                    return False, error_msg
        
        return True, "Range validation passed"
    
    def _validate_types(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate data types
        
        Args:
            data: Input dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        type_specs = {
            'PlayerID': (int, np.integer),
            'Age': (int, np.integer),
            'Gender': (str,),
            'Location': (str,),
            'GameGenre': (str,),
            'PlayTimeHours': (int, float, np.number),
            'InGamePurchases': (int, np.integer),
            'GameDifficulty': (str,),
            'SessionsPerWeek': (int, np.integer),
            'AvgSessionDurationMinutes': (int, float, np.number),
            'PlayerLevel': (int, np.integer),
            'AchievementsUnlocked': (int, np.integer)
        }
        
        for field, expected_types in type_specs.items():
            if field in data:
                value = data[field]
                if not isinstance(value, expected_types):
                    error_msg = f"Type validation failed: {field} is {type(value)}, expected {expected_types}"
                    logger.error(error_msg)
                    return False, error_msg
        
        return True, "Type validation passed"
    
    def _detect_adversarial(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Detect adversarial or anomalous inputs
        
        Args:
            data: Input dictionary
            
        Returns:
            Tuple of (is_adversarial, message)
        """
        # Check for extreme outliers (z-score based)
        numeric_fields = [
            'Age', 'PlayTimeHours', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'
        ]
        
        # Simple heuristic checks
        suspicious_patterns = []
        
        # 1. Impossible combinations
        if data.get('PlayerLevel', 0) > 90 and data.get('PlayTimeHours', 0) < 100:
            suspicious_patterns.append("High level with very low playtime")
        
        if data.get('AchievementsUnlocked', 0) > 500 and data.get('PlayTimeHours', 0) < 200:
            suspicious_patterns.append("Too many achievements for playtime")
        
        if data.get('SessionsPerWeek', 0) > 50 and data.get('AvgSessionDurationMinutes', 0) > 300:
            suspicious_patterns.append("Unrealistic session frequency and duration")
        
        # 2. Check for SQL injection attempts in string fields
        string_fields = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
        sql_keywords = ['SELECT', 'DROP', 'INSERT', 'DELETE', 'UPDATE', 'UNION', '--', ';']
        
        for field in string_fields:
            if field in data:
                value = str(data[field]).upper()
                if any(keyword in value for keyword in sql_keywords):
                    suspicious_patterns.append(f"Potential SQL injection in {field}")
        
        # 3. Check for script injection
        script_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=']
        for field in string_fields:
            if field in data:
                value = str(data[field]).lower()
                if any(pattern in value for pattern in script_patterns):
                    suspicious_patterns.append(f"Potential script injection in {field}")
        
        if suspicious_patterns:
            error_msg = f"Adversarial patterns detected: {', '.join(suspicious_patterns)}"
            return True, error_msg
        
        return False, "No adversarial patterns detected"
    
    def _validate_batch(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a batch of inputs
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_message, validation_details)
        """
        validation_details = {
            'total_rows': len(df),
            'valid_rows': 0,
            'invalid_rows': 0,
            'errors': []
        }
        
        invalid_indices = []
        
        for idx, row in df.iterrows():
            is_valid, error_msg, _ = self.validate(row.to_dict())
            if is_valid:
                validation_details['valid_rows'] += 1
            else:
                validation_details['invalid_rows'] += 1
                invalid_indices.append(idx)
                validation_details['errors'].append(f"Row {idx}: {error_msg}")
        
        if validation_details['invalid_rows'] > 0:
            summary = f"Batch validation failed: {validation_details['invalid_rows']}/{validation_details['total_rows']} rows invalid"
            logger.warning(summary)
            return False, summary, validation_details
        
        logger.info(f"✓ Batch validation passed: {validation_details['valid_rows']} rows")
        return True, "Batch validation passed", validation_details
    
    def get_validation_report(self, validation_details: Dict[str, Any]) -> str:
        """
        Generate human-readable validation report
        
        Args:
            validation_details: Validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("INPUT VALIDATION REPORT (Layer 1)")
        report.append("=" * 60)
        report.append(f"Schema Valid: {validation_details.get('schema_valid', 'N/A')}")
        report.append(f"Range Valid: {validation_details.get('range_valid', 'N/A')}")
        report.append(f"Type Valid: {validation_details.get('type_valid', 'N/A')}")
        report.append(f"Adversarial Detected: {validation_details.get('adversarial_detected', 'N/A')}")
        
        if validation_details.get('errors'):
            report.append("\nErrors:")
            for error in validation_details['errors'][:5]:  # Show first 5
                report.append(f"  - {error}")
        
        report.append("=" * 60)
        return "\n".join(report)
