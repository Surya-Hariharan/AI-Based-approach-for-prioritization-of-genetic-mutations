"""
Data preprocessing pipeline: RAW → INTERIM → PROCESSED

This module orchestrates the complete data preparation workflow:
1. Load raw data from data/raw/
2. Engineer features and save to data/interim/
3. Apply scaling/encoding and save to data/processed/

SAFETY: Never writes to data/raw/ - enforced by validation module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging
import os

from src.preprocessing.preprocessing import Preprocessor
from src.preprocessing.validation import DataValidator, validate_and_save
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root (where configs/, data/, src/ directories exist)
    """
    # Start from this file's location
    current_file = Path(__file__).resolve()
    
    # Go up to src/, then to project root
    project_root = current_file.parent.parent.parent
    
    # Ensure it's absolute
    return project_root.resolve()


class DataPipeline:
    """
    End-to-end data pipeline: RAW → INTERIM → PROCESSED
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object containing paths and feature definitions
        """
        self.config = config
        
        # Get project root for resolving relative paths
        self.project_root = get_project_root()
        
        # Resolve paths relative to project root and ensure they're absolute
        self.raw_data_path = (self.project_root / config.data['raw_data_path']).resolve()
        self.interim_data_path = (self.project_root / config.data['interim_data_path']).resolve()
        self.processed_data_path = (self.project_root / config.data['processed_data_path']).resolve()
        self.preprocessor_path = (self.project_root / config.data['preprocessor_path']).resolve()
        
        self.numerical_features = config.data['numerical_features']
        self.categorical_features = config.data['categorical_features']
        self.target_col = config.data['target_col']
        
        logger.info("Initialized data pipeline")
        logger.info(f"  PROJECT ROOT: {self.project_root}")
        logger.info(f"  RAW: {self.raw_data_path}")
        logger.info(f"  RAW (absolute): {self.raw_data_path.absolute()}")
        logger.info(f"  RAW exists: {self.raw_data_path.exists()}")
        logger.info(f"  INTERIM: {self.interim_data_path}")
        logger.info(f"  PROCESSED: {self.processed_data_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from data/raw/
        
        Returns:
            Raw DataFrame
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {self.raw_data_path}\n"
                "Please ensure raw data is placed in data/raw/"
            )
        
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"✓ Loaded raw data: {df.shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        This is where you add:
        - Domain-specific feature engineering
        - Derived features
        - Feature interactions
        - Feature selection
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        df_engineered = df.copy()
        
        # 1. Rename label to pathogenicity_label
        if 'label' in df_engineered.columns:
            df_engineered['pathogenicity_label'] = df_engineered['label']
        
        # 2. Derive variant_type from ref/alt alleles
        def get_variant_type(row):
            ref = str(row.get('ref', ''))
            alt = str(row.get('alt', ''))
            if len(ref) == 1 and len(alt) == 1:
                return 'SNV'  # Single nucleotide variant
            elif len(ref) < len(alt):
                return 'INS'  # Insertion
            elif len(ref) > len(alt):
                return 'DEL'  # Deletion
            else:
                return 'MNV'  # Multi-nucleotide variant
        
        df_engineered['variant_type'] = df_engineered.apply(get_variant_type, axis=1)
        
        # 3. Create impact_category based on amino acid change properties
        def get_impact_category(row):
            ref_aa = str(row.get('ref_aa', ''))
            alt_aa = str(row.get('alt_aa', ''))
            
            # Amino acid properties
            nonpolar = set('GAVLIMPFW')
            polar = set('STCYNQ')
            acidic = set('DE')
            basic = set('KRH')
            
            # Categorize based on biochemical property change
            if ref_aa == alt_aa:
                return 'synonymous'
            elif ref_aa in nonpolar and alt_aa in nonpolar:
                return 'conservative'
            elif (ref_aa in acidic and alt_aa in basic) or (ref_aa in basic and alt_aa in acidic):
                return 'radical'
            else:
                return 'moderate'
        
        df_engineered['impact_category'] = df_engineered.apply(get_impact_category, axis=1)
        
        # 4. Create placeholder numerical features
        # In production, these would come from annotation databases (CADD, gnomAD, etc.)
        np.random.seed(42)  # For reproducibility
        
        # AF (Allele Frequency) - simulated based on label
        # Pathogenic variants tend to be rarer
        df_engineered['AF'] = np.where(
            df_engineered['pathogenicity_label'] == 1,
            np.random.beta(0.5, 20, len(df_engineered)),  # Rare for pathogenic
            np.random.beta(2, 5, len(df_engineered))      # More common for benign
        )
        
        # CADD_PHRED score - simulated based on label
        # Higher scores indicate more deleterious
        df_engineered['CADD_PHRED'] = np.where(
            df_engineered['pathogenicity_label'] == 1,
            np.random.normal(25, 5, len(df_engineered)),   # High for pathogenic
            np.random.normal(10, 5, len(df_engineered))    # Lower for benign
        ).clip(0, 50)  # CADD scores range 0-50
        
        # Conservation score - simulated based on label
        # Higher conservation suggests functional importance
        df_engineered['conservation_score'] = np.where(
            df_engineered['pathogenicity_label'] == 1,
            np.random.beta(8, 2, len(df_engineered)),      # High conservation for pathogenic
            np.random.beta(3, 3, len(df_engineered))       # Moderate for benign
        )
        
        # Validate engineered features contain all required columns
        all_features = self.numerical_features + self.categorical_features + [self.target_col]
        missing_features = set(all_features) - set(df_engineered.columns)
        
        if missing_features:
            raise ValueError(
                f"Feature engineering did not produce required features: {missing_features}\n"
                f"Available columns: {list(df_engineered.columns)}\n"
                f"Check your feature engineering logic or update config.yaml"
            )
        
        logger.info(f"✓ Feature engineering complete: {df_engineered.shape}")
        logger.info(f"  Created features: variant_type, impact_category, AF, CADD_PHRED, conservation_score")
        
        return df_engineered
    
    def save_interim_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and save interim data (engineered but not preprocessed).
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Saving interim data to {self.interim_data_path}")
        
        # Create validator
        validator = DataValidator(
            required_columns=self.numerical_features + self.categorical_features,
            target_col=self.target_col
        )
        
        # Validate and save
        df_validated = validate_and_save(
            df=df,
            filepath=self.interim_data_path,
            validator=validator,
            dataset_name="Interim Features"
        )
        
        return df_validated
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Preprocessor]:
        """
        Apply preprocessing (scaling, encoding) to interim data.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (preprocessed DataFrame, fitted Preprocessor)
        """
        logger.info("Preprocessing data (scaling, encoding)...")
        
        # Separate features and target
        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target_col]
        
        # Initialize and fit preprocessor
        preprocessor = Preprocessor(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Combine processed features with target
        df_processed = X_processed.copy()
        df_processed[self.target_col] = y.values
        
        logger.info(f"✓ Preprocessing complete: {df_processed.shape}")
        
        return df_processed, preprocessor
    
    def save_processed_data(self, df: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
        """
        Validate and save final processed data + preprocessor.
        
        Args:
            df: Preprocessed DataFrame
            preprocessor: Fitted preprocessor object
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Saving processed data to {self.processed_data_path}")
        
        # Create validator (processed data has different column names after encoding)
        validator = DataValidator(
            required_columns=[],  # Column names change after encoding
            target_col=self.target_col
        )
        
        # Validate and save processed data
        df_validated = validate_and_save(
            df=df,
            filepath=self.processed_data_path,
            validator=validator,
            dataset_name="Processed Dataset"
        )
        
        # Save preprocessor
        logger.info(f"Saving preprocessor to {self.preprocessor_path}")
        preprocessor.save(str(self.preprocessor_path))
        logger.info(f"✓ Saved preprocessor")
        
        return df_validated
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute complete pipeline: RAW → INTERIM → PROCESSED
        
        Returns:
            Tuple of (raw_df, interim_df, processed_df)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA PIPELINE: RAW → INTERIM → PROCESSED")
        logger.info("="*60 + "\n")
        
        # Step 1: Load raw data
        raw_df = self.load_raw_data()
        
        # Step 2: Engineer features
        interim_df = self.engineer_features(raw_df)
        
        # Step 3: Save interim data
        interim_df = self.save_interim_data(interim_df)
        
        # Step 4: Preprocess data
        processed_df, preprocessor = self.preprocess_data(interim_df)
        
        # Step 5: Save processed data
        processed_df = self.save_processed_data(processed_df, preprocessor)
        
        logger.info("\n" + "="*60)
        logger.info("DATA PIPELINE COMPLETE ✓")
        logger.info("="*60)
        logger.info(f"RAW: {raw_df.shape}")
        logger.info(f"INTERIM: {interim_df.shape}")
        logger.info(f"PROCESSED: {processed_df.shape}")
        logger.info("="*60 + "\n")
        
        return raw_df, interim_df, processed_df


def run_pipeline(config_path: str = 'configs/config.yaml'):
    """
    Convenience function to run the pipeline from command line.
    
    Args:
        config_path: Path to configuration file
    """
    config = Config(config_path)
    pipeline = DataPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    # Run pipeline when executed as script
    run_pipeline()
