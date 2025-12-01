"""
Data Loading and Management
Handles loading, splitting, and caching of gaming behavior dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from loguru import logger

class DataLoader:
    """Handles all data loading and initial preprocessing"""
    
    def __init__(self, data_path: str = "data/raw/online_gaming_behavior_dataset.csv"):
        """
        Initialize data loader
        
        Args:
            data_path: Path to the dataset CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info(f"DataLoader initialized with path: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV
        
        Returns:
            DataFrame with loaded data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download from Kaggle: "
                "https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset"
            )
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info(f"Target distribution:\n{self.df['EngagementLevel'].value_counts()}")
        
        return self.df
    
    def load_gaming_dataset(self) -> pd.DataFrame:
        """Alias for load_data() for compatibility"""
        return self.load_data()
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            self.load_data()
        
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "target_distribution": self.df['EngagementLevel'].value_counts().to_dict(),
            "numeric_stats": self.df.describe().to_dict(),
            "categorical_unique": {
                col: self.df[col].nunique() 
                for col in self.df.select_dtypes(include=['object']).columns
            }
        }
        
        return info
    
    def split_data(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion of data for test set
            validation_size: Proportion of training data for validation
            stratify: Whether to stratify split by target variable
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.df is None:
            self.load_data()
        
        # Separate features and target
        X = self.df.drop('EngagementLevel', axis=1)
        y = self.df['EngagementLevel']
        
        # First split: train+val vs test
        stratify_col = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_col = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_col,
            random_state=random_state
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Val: {X_val.shape[0]} samples")
        logger.info(f"  Test: {X_test.shape[0]} samples")
        logger.info(f"  Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"  Val target distribution: {y_val.value_counts().to_dict()}")
        logger.info(f"  Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        output_dir: str = "data/processed"
    ) -> None:
        """
        Save processed data splits to disk
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            y_train, y_val, y_test: Target Series
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        X_train.to_csv(output_path / "X_train.csv", index=False)
        X_val.to_csv(output_path / "X_val.csv", index=False)
        X_test.to_csv(output_path / "X_test.csv", index=False)
        
        y_train.to_csv(output_path / "y_train.csv", index=False)
        y_val.to_csv(output_path / "y_val.csv", index=False)
        y_test.to_csv(output_path / "y_test.csv", index=False)
        
        logger.info(f"Processed data saved to {output_path}")
    
    def load_processed_data(
        self,
        input_dir: str = "data/processed"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load previously processed data splits
        
        Args:
            input_dir: Directory containing processed data
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        input_path = Path(input_dir)
        
        X_train = pd.read_csv(input_path / "X_train.csv")
        X_val = pd.read_csv(input_path / "X_val.csv")
        X_test = pd.read_csv(input_path / "X_test.csv")
        
        y_train = pd.read_csv(input_path / "y_train.csv").squeeze()
        y_val = pd.read_csv(input_path / "y_val.csv").squeeze()
        y_test = pd.read_csv(input_path / "y_test.csv").squeeze()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Processed data loaded from {input_path}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test